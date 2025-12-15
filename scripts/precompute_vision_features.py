import argparse
from pathlib import Path

import numpy as np
import pyrootutils
import torch
from PIL import Image
from tqdm import tqdm

pyrootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from libs.long_clip.model import longclip
from src.data.hicodet import HICODET
from src.utils import img_ops

MODELS = {
    "clip-vit_b_16": ("clip", "openai/clip-vit-base-patch16"),
    "clip-vit_l_14": ("clip", "openai/clip-vit-large-patch14"),
    "longclip_b": ("longclip", "longclip-B"),
    "longclip_l": ("longclip", "longclip-L"),
}

LONGCLIP_PATHS = {
    "longclip-B": "weights/longclip-B.pt",
    "longclip-L": "weights/longclip-L.pt",
}

OUTPUT_FILENAMES = {
    "clip-vit_b_16": "0_hicodet_train_openai-clip-vit-base-patch16_vision_features.pth",
    "clip-vit_l_14": "0_hicodet_train_openai-clip-vit-large-patch14_vision_features.pth",
    "longclip_b": "0_hicodet_train_longclip-B_vision_features.pth",
    "longclip_l": "0_hicodet_train_longclip-L_vision_features.pth",
}


def load_encoder(model_type, model_name, device):
    if model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)

        def preprocess(pil_images):
            pil_images = [img_ops.expand2square(img) for img in pil_images]
            inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            return inputs["pixel_values"].to(device)

        def encode(pixel_values):
            return model.get_image_features(pixel_values)

        return encode, preprocess
    else:
        model, transforms = longclip.load(LONGCLIP_PATHS[model_name], device=device)
        model.eval()

        def preprocess(pil_images):
            pil_images = [img_ops.expand2square(img) for img in pil_images]
            return torch.stack([transforms(img) for img in pil_images]).to(device)

        def encode(pixel_values):
            return model.encode_image(pixel_values)

        return encode, preprocess


def collect_crops_for_image(image_pil, humans_bbox, objects_bbox):
    box_crops = []
    interaction_crops = []

    for human_bbox, object_bbox in zip(humans_bbox, objects_bbox):
        box_crops.append(image_pil.crop(human_bbox.tolist()))
        box_crops.append(image_pil.crop(object_bbox.tolist()))

        union_bbox = torch.cat([
            torch.min(human_bbox[:2], object_bbox[:2]),
            torch.max(human_bbox[2:], object_bbox[2:]),
        ])
        interaction_crops.append(image_pil.crop(union_bbox.tolist()))

    return box_crops, interaction_crops


def encode_batched(encode_fn, preprocess_fn, pil_images, batch_size):
    if len(pil_images) == 0:
        return torch.empty(0)

    all_features = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        pixel_values = preprocess_fn(batch)
        features = encode_fn(pixel_values)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute vision features for HICODET")
    parser.add_argument("--data_root", type=str, default="data/hicodet")
    parser.add_argument("--output_dir", type=str, default="weights")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=64, help="Images per chunk (crops collected+encoded per chunk then freed)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dataset
    print("Loading HICODET train split...")
    dataset = HICODET(root_dir=args.data_root, split="train")
    dataset.setup()

    num_images = len(dataset)

    # Encode with each model
    for model_key in args.models:
        model_type, model_name = MODELS[model_key]
        print(f"\nEncoding with {model_key} ({model_name})...")

        encode_fn, preprocess_fn = load_encoder(model_type, model_name, device)

        features = {}
        # Process dataset in chunks: collect crops, encode, store results, free crops
        for chunk_start in tqdm(
            range(0, num_images, args.chunk_size),
            desc=f"  Encoding ({model_key})",
            total=(num_images + args.chunk_size - 1) // args.chunk_size,
        ):
            # Collect crops for this chunk only
            all_box_crops = []
            all_interaction_crops = []
            chunk_meta = []
            offsets_box = []
            offsets_int = []
            pos_box = 0
            pos_int = 0

            for idx in range(chunk_start, min(chunk_start + args.chunk_size, num_images)):
                sample = dataset[idx]
                if len(sample["verbs_id"]) == 0:
                    continue

                image_pil = sample["images_pil"]
                humans_bbox = sample["humans_bbox"]
                objects_bbox = sample["objects_bbox"]
                n = len(sample["verbs_id"])

                box_crops, interaction_crops = collect_crops_for_image(
                    image_pil, humans_bbox, objects_bbox
                )

                all_box_crops.extend(box_crops)
                all_interaction_crops.extend(interaction_crops)
                offsets_box.append((pos_box, pos_box + len(box_crops)))
                offsets_int.append((pos_int, pos_int + len(interaction_crops)))
                pos_box += len(box_crops)
                pos_int += len(interaction_crops)

                chunk_meta.append({
                    "filename": sample["images_filename"],
                    "verbs_id": sample["verbs_id"],
                    "objects_id": sample["objects_id"],
                    "humans_bbox": humans_bbox,
                    "objects_bbox": objects_bbox,
                    "n_interactions": n,
                })

            if len(chunk_meta) == 0:
                continue

            # Batch encode all crops from this chunk
            with torch.no_grad():
                box_feats = encode_batched(encode_fn, preprocess_fn, all_box_crops, args.batch_size)
                int_feats = encode_batched(encode_fn, preprocess_fn, all_interaction_crops, args.batch_size)

            del all_box_crops, all_interaction_crops

            # Split back per image
            for i, meta in enumerate(chunk_meta):
                n = meta["n_interactions"]
                b_start, b_end = offsets_box[i]
                i_start, i_end = offsets_int[i]

                pred_boxes = torch.cat([meta["humans_bbox"], meta["objects_bbox"]], dim=0)
                pred_labels = torch.cat([
                    torch.full((n,), dataset.person_idx, dtype=torch.long),
                    meta["objects_id"],
                ], dim=0)
                interactions_pairings = torch.tensor(
                    [[j, j + n] for j in range(n)], dtype=torch.long,
                )

                features[meta["filename"]] = {
                    "interactions_pairings": interactions_pairings,
                    "verbs_id": meta["verbs_id"],
                    "pred_boxes": pred_boxes,
                    "pred_labels": pred_labels,
                    "pred_boxes_image_features": box_feats[b_start:b_end],
                    "pred_interactions_image_features": int_feats[i_start:i_end],
                }

        out_dir = Path(args.output_dir) / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        output = {"features": features}

        out_path = out_dir / OUTPUT_FILENAMES[model_key]
        print(f"  Saving to {out_path}...")
        torch.save(output, out_path)
        print(f"  Saved {len(features)} images")

        del encode_fn, preprocess_fn
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
