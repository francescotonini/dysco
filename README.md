# Dynamic Scoring with Enhanced Semantics for Training-Free Human-Object Interaction Detection
![teaser](./assets/teaser.png)

## Installation
```bash
# Create environment
uv sync

# Or with pip
pip install -e .
```

## Data
Follow the process of [ADA-CM](https://github.com/ltttpku/ADA-CM). Then, link the downloaded dataset as follows:

```bash
ln -s /path/to/hico_20160224_det data/hicodet
```

Download the following to the `weights/` folder:
| File | Source |
|------|--------|
| `longclip-B.pt` | [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) |
| `longclip-L.pt` | [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) |
| `openai/clip-vit-base-patch16` | downloaded automatically from HuggingFace |
| `openai/clip-vit-large-patch14` | downloaded automatically from HuggingFace |

Download the pre-computed signatures and the DETR detections from [this link](https://drive.google.com/file/d/1JG9dQvlWEfHSTWddG0_znicyn7a74tg7/view?usp=sharing) and unzip it into `weights/`. It contains:
- `weights/<model>/prompts_X.pth`: pre-computed text features.
- `weights/detr/hicodet_test_bbox_R50.pt`: pre-computed DETR detections.

Generate vision features
```bash
python scripts/precompute_vision_features.py \
    --data_root data/hicodet \
    --output_dir weights \
    --models clip-vit_b_16 clip-vit_l_14 longclip_b longclip_l \
    --batch_size 256
```

After all the steps above, the `weights/` folder should look like:
```
weights/
├── longclip-B.pt
├── longclip-L.pt
├── detr/
│   └── hicodet_test_bbox_R50.pt
├── clip-vit_b_16/
│   ├── prompts_0.pth
│   ├── prompts_1.pth
│   └── 0_hicodet_train_openai-clip-vit-base-patch16_vision_features.pth
├── clip-vit_l_14/
│   ├── prompts_0.pth
│   ├── prompts_1.pth
│   └── 0_hicodet_train_openai-clip-vit-large-patch14_vision_features.pth
├── longclip_b/
│   ├── prompts_0.pth
│   ├── prompts_1.pth
│   └── 0_hicodet_train_longclip-B_vision_features.pth
└── longclip_l/
    ├── prompts_0.pth
    ├── prompts_1.pth
    └── 0_hicodet_train_longclip-L_vision_features.pth
```

## Experiments
Run any experiment with:

```bash
python src/main.py experiment=<config>
```

Available configurations:
| Config | VLM |
|--------|---------|
| `clip-vit_b_16` | CLIP ViT-B/16 |
| `clip-vit_l_14` | CLIP ViT-L/14 |
| `longclip_b` | LongCLIP-B |
| `longclip_l` | LongCLIP-L |

For example:
```bash
python src/main.py experiment=clip-vit_b_16
```

## Cite us!
If you find our paper and/or code helpful, please consider citing:
```bibtex
@inproceedings{tonini2025dynamic,
    title={Dynamic Scoring with Enhanced Semantics for Training-Free Human-Object Interaction Detection},
    author={Tonini, Francesco and Vaquero, Lorenzo and Conti, Alessandro and Beyan, Cigdem and Ricci, Elisa},
    booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
    pages={2801--2810},
    year={2025}
}
```

## Acknowledgement
We gratefully thank the authors from [ADA-CM](https://github.com/ltttpku/ADA-CM) and [Lightning Hydra](https://github.com/ashleve/lightning-hydra-template) for open-sourcing their code.