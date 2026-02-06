# DiT-101

A minimal, from-scratch implementation of [Diffusion Transformers (DiT)](https://www.wpeebles.com/DiT) for conditional image generation on the MNIST dataset. This project demonstrates how to combine Vision Transformers with DDPM diffusion to generate class-conditional handwritten digits.

## Model Architecture

The model patchifies 28x28 images into a sequence of 4x4 patches, processes them through transformer blocks with **Adaptive Layer Normalization (AdaLN-Zero)** conditioning, and reconstructs the full image.

![Model Architecture](dits.png)

| Parameter | Value |
|---|---|
| Image Size | 28 x 28 |
| Patch Size | 4 x 4 (49 patches) |
| Embedding Dim | 64 |
| DiT Blocks | 3 |
| Attention Heads | 4 |
| Diffusion Steps | 1000 |
| Classes | 10 (digits 0-9) |

## Project Structure

```
├── dit.py            # Full DiT model (patchify → transformer blocks → unpatchify)
├── dit_block.py      # Transformer block with AdaLN-Zero conditioning
├── time_emb.py       # Sinusoidal timestep embedding
├── diffusion.py      # DDPM forward process (noise scheduling)
├── dataset.py        # MNIST dataset wrapper
├── train.py          # Training loop
├── inference.py      # Image generation (reverse diffusion)
├── config.py         # Global config (T=1000 timesteps)
└── model.pth         # Pretrained checkpoint
```

## How It Works

**Training** — Sample a clean image, add noise at a random timestep via the forward diffusion process, and train the model to predict the added noise (L1 loss).

**Inference** — Start from pure Gaussian noise and iteratively denoise over 1000 steps, conditioned on a target digit class.

## Results

Each row shows one digit class (0-9) progressively denoised from random noise (left) to a clean image (right):

![Inference Results](inference.png)

## Getting Started

### Requirements

- Python 3
- PyTorch
- torchvision
- matplotlib

### Training

```bash
python train.py
```

Trains for 500 epochs with batch size 300. Checkpoints are saved to `model.pth` every 1000 iterations. A GPU is recommended.

### Inference

```bash
python inference.py
```

Generates one image per digit class (0-9) and saves the denoising progression to `inference.png`.

## Interactive Notebook
You can also try the interactive notebooks `diffusion.ipynb`, `train.ipynb` and `inference.ipynb` to interactively explore the diffusion process, training loop, and inference process.

## References

- [Scalable Diffusion Models with Transformers](https://www.wpeebles.com/DiT) (Peebles & Xie, 2023)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)