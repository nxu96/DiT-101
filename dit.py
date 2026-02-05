"""
Diffusion Transformer (DiT) Model Implementation

This module implements the full DiT architecture for image generation.
The model takes a noisy image, a diffusion timestep, and a class label as input,
and predicts the noise to be removed (or the denoised image, depending on the
training objective).

Architecture Overview:
1. Patchify: Split image into non-overlapping patches and embed them
2. Add positional embeddings to patches
3. Create conditioning vector from timestep + class label
4. Process through multiple DiT blocks (transformer with AdaLN conditioning)
5. Unpatchify: Reconstruct the image from processed patches

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import torch
from torch import nn

from config import T  # Total number of diffusion timesteps
from dit_block import DiTBlock  # Transformer block with AdaLN conditioning
from time_emb import TimeEmbedding  # Sinusoidal timestep embedding


class DiT(nn.Module):
    """
    Diffusion Transformer model for conditional image generation.

    The model processes images as sequences of patches (similar to ViT),
    with conditioning information (timestep + class) injected via AdaLN
    in each transformer block.

    Args:
        img_size: Height/width of input images (assumes square images)
        patch_size: Size of each patch (e.g., 4 means 4x4 patches)
        channel: Number of image channels (1 for grayscale, 3 for RGB)
        emb_size: Dimension of token embeddings throughout the model
        label_num: Number of classes for conditional generation
        dit_num: Number of DiT blocks (transformer depth)
        head: Number of attention heads in each DiT block
    """

    def __init__(
        self, img_size, patch_size, channel, emb_size, label_num, dit_num, head
    ):
        super().__init__()

        # ==== Store Configuration ====
        self.patch_size = patch_size  # Size of each patch (e.g., 4x4)
        self.patch_count = img_size // self.patch_size  # Patches per row/column
        self.channel = channel  # Number of image channels

        # ==== Patchify Layers ====
        # Convert image into patch embeddings using a convolution.
        # The conv acts as a learned patch extractor: each patch_size x patch_size
        # region is projected to channel * patch_size^2 features.
        # Stride = patch_size ensures non-overlapping patches.
        self.conv = nn.Conv2d(
            in_channels=channel,  # Input channels (1 for MNIST)
            out_channels=channel * patch_size**2,  # Flattened patch features
            kernel_size=patch_size,  # Each patch is patch_size x patch_size
            padding=0,  # No padding
            stride=patch_size,  # Non-overlapping patches
        )

        # Project flattened patch features to embedding dimension
        self.patch_emb = nn.Linear(
            in_features=channel * patch_size**2,  # Flattened patch size
            out_features=emb_size,  # Embedding dimension
        )
        # NOTE: We use conv + linear to project each patch to a token.

        # Learnable positional embeddings for each patch position
        # Shape: (1, num_patches, emb_size) - 1 is for broadcasting across batch
        # Uniformly initialized positional embeddings.
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count**2, emb_size))

        # ==== Timestep Embedding ====
        # Convert scalar timestep to a rich embedding vector.
        # Uses sinusoidal encoding followed by an MLP for expressiveness.
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),  # Sinusoidal positional encoding
            nn.Linear(emb_size, emb_size),  # MLP layer 1
            nn.ReLU(),  # Non-linearity
            nn.Linear(emb_size, emb_size),  # MLP layer 2
        )

        # ==== Class Label Embedding ====
        # Lookup table that maps class indices to embedding vectors.
        # Each of the label_num classes gets its own learned emb_size vector.
        self.label_emb = nn.Embedding(
            num_embeddings=label_num,  # Number of classes (10 for MNIST)
            embedding_dim=emb_size,  # Embedding dimension
        )

        # ==== DiT Blocks (Transformer Backbone) ====
        # Stack of transformer blocks with AdaLN conditioning.
        # Each block processes the patch sequence with self-attention and FFN,
        # modulated by the conditioning vector (timestep + class).
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))

        # ==== Output Layers ====
        # Final layer norm before projection (standard in transformers)
        self.ln = nn.LayerNorm(emb_size)

        # Project embeddings back to patch pixel space
        # Output has same size as flattened patch: channel * patch_size^2
        self.linear = nn.Linear(emb_size, channel * patch_size**2)

    def forward(self, x, t, y):
        """
        Forward pass of the DiT model.

        Args:
            x: Noisy input images
               Shape: (batch, channel, height, width)
            t: Diffusion timesteps (which noise level each image is at)
               Shape: (batch,)
            y: Class labels for conditional generation
               Shape: (batch,)

        Returns:
            Predicted noise (or denoised image) with same shape as input:
            (batch, channel, height, width)
        """
        # ============================================================
        # STEP 1: Create Conditioning Vector
        # ============================================================

        # Embed class labels: lookup in embedding table
        y_emb = self.label_emb(y)  # Shape: (batch, emb_size)

        # Embed timesteps: sinusoidal encoding + MLP
        t_emb = self.time_emb(t)  # Shape: (batch, emb_size)

        # Combine into single conditioning vector (simple addition)
        # This will be used by AdaLN in each DiT block
        cond = y_emb + t_emb  # Shape: (batch, emb_size)

        # ============================================================
        # STEP 2: Patchify - Convert Image to Patch Sequence
        # ============================================================

        # Apply strided convolution to extract patch features
        # Input: (batch, channel, H, W) e.g., (batch, 1, 28, 28)
        # Output: (batch, channel*patch_size^2, patch_count, patch_count)
        #         e.g., (batch, 16, 7, 7) for patch_size=4
        x = self.conv(x)
        # Rearrange: (batch, C, h, w) -> (batch, h, w, C)
        # This prepares for flattening the spatial dimensions
        x = x.permute(0, 2, 3, 1)
        # Flatten spatial dimensions: (batch, h, w, C) -> (batch, h*w, C)
        # Now we have a sequence of patches, like tokens in NLP
        x = x.view(x.size(0), self.patch_count * self.patch_count, x.size(3))

        # Project patch features to embedding dimension
        x = self.patch_emb(x)  # Shape: (batch, num_patches, emb_size)

        # Add positional embeddings so the model knows patch locations
        # patch_pos_emb broadcasts: (1, num_patches, emb_size) + (batch, ...)
        # NOTE: Its just a simple addition, not concat or anything.
        # Key insight: learnable positional embeddings are SHARED across all images.
        # One thing that is consistent across all images at position i is the position itself.
        x = x + self.patch_pos_emb  # Shape: (batch, num_patches, emb_size)

        # ============================================================
        # STEP 3: Process Through DiT Blocks
        # ============================================================

        # Pass through each transformer block with conditioning
        for dit in self.dits:
            x = dit(x, cond)  # Each block uses cond for AdaLN modulation

        # ============================================================
        # STEP 4: Output Projection
        # ============================================================

        # Final layer normalization
        x = self.ln(x)  # Shape: (batch, num_patches, emb_size)

        # Project back to patch pixel space
        x = self.linear(x)  # Shape: (batch, num_patches, channel*patch_size^2)

        # ============================================================
        # STEP 5: Unpatchify - Reconstruct Image from Patches
        # ============================================================

        # Reshape to separate patch grid and patch pixels
        # From: (batch, num_patches, channel*patch_size^2)
        # To: (batch, patch_count, patch_count, channel, patch_size, patch_size)
        x = x.view(
            x.size(0),
            self.patch_count,  # Number of patches in height
            self.patch_count,  # Number of patches in width
            self.channel,  # Image channels
            self.patch_size,  # Patch height
            self.patch_size,  # Patch width
        )

        # Rearrange dimensions to group spatial dimensions together
        # From: (batch, pH, pW, C, psH, psW)
        # To: (batch, C, pH, pW, psH, psW)
        x = x.permute(0, 3, 1, 2, 4, 5)

        # Interleave patch grid with patch pixels
        # From: (batch, C, pH, pW, psH, psW)
        # To: (batch, C, pH, psH, pW, psW)
        x = x.permute(0, 1, 2, 4, 3, 5)

        # Final reshape to image dimensions
        # From: (batch, C, pH, psH, pW, psW)
        # To: (batch, C, pH*psH, pW*psW) = (batch, C, H, W)
        x = x.reshape(
            x.size(0),
            self.channel,
            self.patch_count * self.patch_size,  # Original height
            self.patch_count * self.patch_size,  # Original width
        )

        return x  # Shape: (batch, channel, height, width)


if __name__ == "__main__":
    # ==== Test the DiT Model ====

    # Initialize model for MNIST-like images:
    # - 28x28 grayscale images
    # - 4x4 patches (49 patches total in a 7x7 grid)
    # - 256-dimensional embeddings
    # - 10 classes (digits 0-9)
    # - 6 DiT blocks with 8 attention heads each
    dit = DiT(
        img_size=28,
        patch_size=4,
        channel=1,
        emb_size=256,
        label_num=10,
        dit_num=6,
        head=8,
    )

    # Create dummy inputs
    x = torch.rand(5, 1, 28, 28)  # 5 random 28x28 grayscale images
    t = torch.randint(0, T, (5,))  # 5 random timesteps in [0, T)
    y = torch.randint(0, 10, (5,))  # 5 random class labels in [0, 10)

    # Forward pass
    outputs = dit(x, t, y)

    # Verify output shape matches input
    assert outputs.shape == x.shape
    print(f"Input shape:  {x.shape}")  # Expected: (5, 1, 28, 28)
    print(f"Output shape: {outputs.shape}")  # Expected: (5, 1, 28, 28)
