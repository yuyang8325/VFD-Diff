# VFD-Diff

Official PyTorch implementation of VFD-Diff described in the paper:

Yang Yu, Weiwei Gao, Zheng Song, Feng Xiao, "Vessel-aware frequency distillation diffusion with a teacher-student network for fundus fluorescein angiography synthesis."

*(Paper link will be added upon publication.)*

VFD-Diff translates color fundus photography (CFP) images into fundus fluorescein angiography (FFA) images with enhanced vascular detail. Built on the adversarial diffusion baseline [SynDiff](https://github.com/icon-lab/SynDiff), it introduces three vessel-aware modules:

- **VMG** — Adaptive vessel-mask-guided local enhancement: a parameter-adaptive Frangi filtering method extracts a vessel mask from the input CFP image (Eqs. (1)-(4)), on which mask-weighted L1 and SSIM losses are constructed to focus optimization on vascular regions (Eqs. (5)-(6)).
- **FDE** — Dual-stage frequency-domain enhancement: high-frequency-enhanced noise is used in the forward diffusion process (Eq. (7)), and Fourier-domain low-frequency suppression is applied to the generated image (Eq. (8)), jointly improving vessel edge sharpness and fine vessel generation.
- **DG** — Vessel-region distillation guidance: a teacher network that takes vessel-mask-weighted inputs is first trained and frozen; the student network then receives pixel-level and feature-level distillation losses on top of its global loss (Eq. (9)).

## Dependencies

```
python>=3.8
torch>=1.8.0
torchvision>=0.9.0
cuda>=11.2
numpy
scikit-image
h5py
pillow
```

## Installation
- Clone this repo:
```bash
git clone https://github.com/yuyang8325/VFD-Diff
cd VFD-Diff
```

## Dataset
You should structure your aligned dataset in the following way:

```
input_path/
  ├── tranA/                    # source modality (CFP)
  │   ├── train/
  │   │   └── 0_Normal/         
  │   │       └── 1_1.png
  │   ├── val/
  │   │   └── 43_1.png
  │   └── test/
  │       └── 7_1.png
  └── tranB/                    # target modality (FFA)
      ├── train/
      │   └── 0_Normal/
      │       └── 1_mask_1.png
      ├── val/
      │   └── 43_mask_1.png
      └── test/
          └── 7_mask_1.png
```

In the paper, experiments are conducted on the public **Isfahan MISP** and **MPOS** datasets
with the splits described in Section 3.1 of the manuscript. Dataset images are **not** included
in this repository and should be prepared by yourself following the structure above.

Pairing rule: `X_Y.png` in tranA corresponds to `X_mask_Y.png` in tranB. Images are loaded as
grayscale, resized to 512×512 and scaled to [-1, 1]. The train set may contain arbitrary class
subfolders (they only need to match between tranA and tranB); val/test images are placed
directly in the `val`/`test` folders. Vessel masks are computed from the CFP inputs on first
access and cached under `input_path/tranA_vmasks/`.

## Train

Training follows the two-stage protocol of Section 2.3 . Per the paper (Section 3.3), the Adam optimizer is used with an initial learning rate of 1.5e-4 that stays constant for the first 300 epochs and linearly decays to zero over the subsequent 300
epochs (`--lr_schedule paper`), on a single RTX 4090 GPU.

**Step 1 — Teacher stage** (recommended budget: half of the student stage, i.e. 300 epochs):

```
python3 train.py --image_size 512 --exp exp_ffa_generation --num_channels 2 --num_timesteps 4 --batch_size 1 --contrast1 tranA --contrast2 tranB --num_epoch 300 --input_path /input/path/for/data --output_path /output/for/results --ch_mult 1 1 2 2 4 4 --lambda_l1_loss 15 --lr_g 1.5e-4 --lr_schedule paper --save_content --save_ckpt_every 20 --gpu_chose 0 --vfd_stage teacher
```

**Step 2 — Student stage** (VFD-Diff full training, 600 epochs):

```
python3 train.py --image_size 512 --exp exp_ffa_generation --num_channels 2 --num_timesteps 4 --batch_size 1 --contrast1 tranA --contrast2 tranB --num_epoch 600 --input_path /input/path/for/data --output_path /output/for/results --ch_mult 1 1 2 2 4 4 --lambda_l1_loss 15 --lr_g 1.5e-4 --lr_schedule paper --save_content --save_ckpt_every 20 --gpu_chose 0 --vfd_stage student
```

## Test

```
python test.py --image_size 512 --exp exp_ffa_generation --num_channels 2 --num_timesteps 4 --batch_size 1 --embedding_type positional --z_emb_dim 512 --contrast1 tranA --contrast2 tranB --which_epoch 600 --gpu_chose 0 --input_path /input/path/for/data --output_path /output/for/results
```

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@ARTICLE{yu2026vfddiff,
  author={Yang Yu, Weiwei Gao, Zheng Song, Feng Xiao},
  journal={-- to be updated upon publication --},
  title={Vessel-Aware Frequency Distillation Diffusion with a Teacher-Student Network for Fundus Fluorescein Angiography Synthesis},
  year={2026}}
```

For any questions, comments and contributions, please contact Yang Yu (yuyang8325@163.com)

# License

This project is licensed under the [Apache License 2.0](./LICENSE). It is a derivative work of
[SynDiff](https://github.com/icon-lab/SynDiff) (MIT License); the required third-party notices
are reproduced in the appendix of the [LICENSE](./LICENSE) file and in [NOTICE](./NOTICE).

# Acknowledgements

This code uses libraries from [SynDiff](https://github.com/icon-lab/SynDiff), [pGAN-cGAN](https://github.com/icon-lab/pGAN-cGAN), [StyleGAN-2](https://github.com/NVlabs/stylegan2), and [DD-GAN](https://github.com/NVlabs/denoising-diffusion-gan) repositories.
