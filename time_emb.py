import math

import torch
from torch import nn

from config import T


# Time embedding, embed the time step into a vector.
class TimeEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        # sinusoidal positional embedding (like the one in the original transformer paper).
        # Create a geometric sequence of frequencies, ranging from 1.0 (highest)
        # to 1/10000.0 (lowest).
        self.half_emb_size = emb_size // 2
        half_emb = torch.exp(
            torch.arange(self.half_emb_size)
            * (-1 * math.log(10000) / (self.half_emb_size - 1))
        )
        # Register as a buffer, not a learnable parameter so it moves with the model to
        # the GPU and is not updated during training.
        self.register_buffer("half_emb", half_emb)

    def forward(self, t):
        t = t.view(t.size(0), 1)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
        # Multiply the timestep t by each frequency.
        half_emb_t = half_emb * t
        # Apply sine and cosine to get the final embedding.
        # Difference frequencies capture different "scales" of time - high frequencies
        # detect small changes low frequencies distinguish large time gaps.
        # This essentially gives the model a rich representation of where it is
        # in the diffusion process.
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t


if __name__ == "__main__":
    time_emb = TimeEmbedding(16)
    t = torch.randint(0, T, (3,))  # Timestamps for 3 images.
    print("Timesteps are ", t)
    embs = time_emb(t)
    print("Time embedding shape is ", embs.shape)  # [t, emb_size]
    print(embs)
    print(embs)
