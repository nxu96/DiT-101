"""
DiT Inference Script — Flow Matching (ODE Sampling)

This script generates images using a DiT model trained with the flow matching
objective. Instead of reversing a Markov chain (DDPM) or using a non-Markovian
update (DDIM), flow matching solves an ODE from noise to data.

The ODE to solve:
    dx/dt = v_θ(x_t, t)

We integrate backwards from t=1 (pure noise) to t=0 (clean image) using the
Euler method:
    x_{t-dt} = x_t - dt * v_θ(x_t, t)

This is inherently deterministic (no random noise injected during sampling),
and the number of steps is freely adjustable.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""

import torch
import matplotlib.pyplot as plt

from dit import DiT


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Number of Euler steps for ODE integration (more steps = higher quality)
NUM_STEPS = 50

# Must match the scale used during training (see train_fm.py)
T_SCALE = 1000


# ==============================================================================
# ODE SAMPLING (EULER METHOD)
# ==============================================================================


def ode_sample(model, x, y, num_steps=50):
    """
    Generate images by solving the flow matching ODE from noise to data.

    Starting from pure noise x_1 ~ N(0, I), we integrate the learned velocity
    field v_θ backwards from t=1 to t=0 using the Euler method:

        x_{t-dt} = x_t - dt * v_θ(x_t, t)

    Args:
        model: Trained DiT model (trained with flow matching objective)
        x: Initial noise tensor, shape (batch, channel, height, width)
        y: Class labels for conditional generation, shape (batch,)
        num_steps: Number of Euler integration steps

    Returns:
        steps: List of tensors showing the generation progression.
               First element is the initial noise, last is the final image.
    """
    steps = [x.clone()]

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    dt = 1.0 / num_steps

    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            # Current time: goes from 1.0 down to dt
            t = 1.0 - i * dt

            # Scale t for the sinusoidal time embedding (must match training)
            t_batch = torch.full((x.size(0),), t * T_SCALE, device=DEVICE)

            # Model predicts velocity v_θ(x_t, t)
            velocity = model(x, t_batch, y)

            # Euler step: x_{t-dt} = x_t - dt * v_θ
            x = x - dt * velocity

            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x.clone())

    return steps


# ==============================================================================
# LOAD TRAINED MODEL
# ==============================================================================

model = DiT(
    img_size=28,
    patch_size=4,
    channel=1,
    emb_size=64,
    label_num=10,
    dit_num=3,
    head=4,
).to(DEVICE)

model.load_state_dict(torch.load("model_fm.pth"))
print("Flow matching model loaded successfully")


# ==============================================================================
# GENERATE IMAGES
# ==============================================================================

batch_size = 10

# Start from pure Gaussian noise (t=1 endpoint of the flow)
x = torch.randn(size=(batch_size, 1, 28, 28))

# Class labels: generate one of each digit (0, 1, 2, ..., 9)
y = torch.arange(start=0, end=10, dtype=torch.long)

print(f"Generating {batch_size} images...")
print(f"Class labels: {y.tolist()}")
print(f"Using Euler ODE sampler with {NUM_STEPS} steps")

steps = ode_sample(model, x, y, num_steps=NUM_STEPS)

num_generated_steps = len(steps) - 1
print(f"Generation complete! {num_generated_steps} ODE steps")


# ==============================================================================
# VISUALIZE RESULTS
# ==============================================================================

num_imgs = 20

plt.figure(figsize=(15, 15))

for b in range(batch_size):
    for i in range(num_imgs):
        idx = int(num_generated_steps / num_imgs) * (i + 1)
        idx = min(idx, num_generated_steps)

        # Convert from [-1, 1] back to [0, 1] for visualization
        final_img = (steps[idx][b].to("cpu") + 1) / 2
        final_img = final_img.permute(1, 2, 0)

        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img, cmap="gray")
        plt.axis("off")

plt.suptitle("Flow Matching: Noise → Generated Digit (Euler ODE)", fontsize=14)
plt.tight_layout()
plt.savefig("inference_fm.png", dpi=150, bbox_inches="tight")

print("Visualization saved to inference_fm.png")
