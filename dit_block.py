"""
DiT Block Implementation

This module implements the Diffusion Transformer (DiT) block with AdaLN-Zero conditioning.
The key innovation in DiT is using Adaptive Layer Normalization (AdaLN) to inject
conditioning information (timestep and class) into the transformer block, rather than
adding it directly to the input tokens.

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    """
    A single DiT (Diffusion Transformer) block with AdaLN-Zero conditioning.

    The block consists of two sub-blocks:
    1. Multi-head self-attention with AdaLN conditioning
    2. Feed-forward network (MLP) with AdaLN conditioning

    Each sub-block uses:
    - gamma (γ): Scale factor for layer norm output (multiplicative modulation)
    - beta (β): Shift factor for layer norm output (additive modulation)
    - alpha (α): Scale factor for the output before residual (controls contribution)

    Args:
        emb_size: Dimension of token embeddings
        nhead: Number of attention heads
    """

    def __init__(self, emb_size, nhead):
        super().__init__()

        # Store dimensions for reshaping in forward pass
        self.emb_size = emb_size  # Dimension of each token embedding
        self.nhead = nhead  # Number of parallel attention heads

        # ==== AdaLN-Zero Conditioning Layers ====
        # These layers learn to predict modulation parameters from the condition vector.
        # In DiT-Zero, alpha layers are initialized to output zeros, making the block
        # start as an identity function during training.

        # Conditioning for the attention sub-block
        self.gamma1 = nn.Linear(emb_size, emb_size)  # Scale for layer norm output
        self.beta1 = nn.Linear(emb_size, emb_size)  # Shift for layer norm output
        self.alpha1 = nn.Linear(emb_size, emb_size)  # Scale for attention output

        # Conditioning for the feed-forward sub-block
        self.gamma2 = nn.Linear(emb_size, emb_size)  # Scale for layer norm output
        self.beta2 = nn.Linear(emb_size, emb_size)  # Shift for layer norm output
        self.alpha2 = nn.Linear(emb_size, emb_size)  # Scale for FFN output

        # ==== Layer Normalization ====
        # Standard layer norm applied before each sub-block (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(emb_size)  # Before attention
        self.ln2 = nn.LayerNorm(emb_size)  # Before feed-forward

        # ==== Multi-Head Self-Attention ====
        # Project input to Query, Key, Value for all heads simultaneously.
        # Output dimension is nhead * emb_size (each head gets emb_size dimensions).
        self.wq = nn.Linear(emb_size, nhead * emb_size)  # Query projection
        self.wk = nn.Linear(emb_size, nhead * emb_size)  # Key projection
        self.wv = nn.Linear(emb_size, nhead * emb_size)  # Value projection
        # Project concatenated head outputs back to emb_size
        self.lv = nn.Linear(nhead * emb_size, emb_size)  # Output projection

        # ==== Feed-Forward Network (MLP) ====
        # Standard transformer FFN: expand 4x -> non-linearity -> project back
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),  # Expand to 4x hidden dim
            nn.ReLU(),  # Non-linearity (GELU is also common)
            nn.Linear(emb_size * 4, emb_size),  # Project back to emb_size
        )

    def forward(self, x, cond):
        """
        Forward pass of the DiT block.

        Args:
            x: Input tokens (image patches, result after ViT tokenization)
               Shape: (bs, seq_len, emb_size)
            cond: Conditioning vector (combined timestep + class embedding)
                  Shape: (bs, emb_size)

        Returns:
            Output tokens with same shape as input: (bs, seq_len, emb_size)
        """
        # NOTE: cond is just a single embedding vector, not a sequence of embeddings.
        # ==== Compute All Conditioning Parameters ====
        # Transform the condition vector into modulation parameters.
        # Each parameter has shape (bs, emb_size).
        gamma1_val = self.gamma1(cond)  # Attention block scale
        beta1_val = self.beta1(cond)  # Attention block shift
        alpha1_val = self.alpha1(cond)  # Attention output scale
        gamma2_val = self.gamma2(cond)  # FFN block scale
        beta2_val = self.beta2(cond)  # FFN block shift
        alpha2_val = self.alpha2(cond)  # FFN output scale

        # ============================================================
        # SUB-BLOCK 1: Multi-Head Self-Attention with AdaLN
        # ============================================================

        # Step 1: Layer normalization
        y = self.ln1(x)  # Shape: (bs, seq_len, emb_size)

        # Step 2: AdaLN modulation (scale and shift)
        # Formula: y = y * (1 + gamma) + beta
        # unsqueeze(1) broadcasts (bs, emb_size) -> (bs, 1, emb_size)
        # to apply same modulation to all tokens in the sequence
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # Step 3: Compute Query, Key, Value projections
        q = self.wq(y)  # Shape: (bs, seq_len, nhead * emb_size)
        k = self.wk(y)  # Shape: (bs, seq_len, nhead * emb_size)
        v = self.wv(y)  # Shape: (bs, seq_len, nhead * emb_size)

        # Step 4: Reshape for multi-head attention
        # Split the last dimension into (nhead, emb_size) and rearrange
        # Q: (batch, seq_len, nhead, emb_size) -> (batch, nhead, seq_len, emb_size)
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        # K: Transposed differently for matrix multiplication with Q
        # (batch, seq_len, nhead, emb_size) -> (batch, nhead, emb_size, seq_len)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3, 1)
        # V: Same arrangement as Q
        # (batch, seq_len, nhead, emb_size) -> (batch, nhead, seq_len, emb_size)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)

        # Step 5: Scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        attn = (
            q @ k / math.sqrt(self.emb_size)
        )  # Shape: (batch, nhead, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)  # Normalize attention weights
        y = attn @ v  # Weighted sum of values, shape: (batch, nhead, seq_len, emb_size)

        # Step 6: Concatenate heads and project back
        # (batch, nhead, seq_len, emb_size) -> (batch, seq_len, nhead, emb_size)
        y = y.permute(0, 2, 1, 3)
        # Flatten heads: (batch, seq_len, nhead * emb_size)
        y = y.reshape(y.size(0), y.size(1), self.nhead * self.emb_size)
        # Project back to emb_size
        y = self.lv(y)  # Shape: (batch, seq_len, emb_size)

        # Step 7: Alpha scaling (controls residual contribution)
        # In DiT-Zero, alpha starts at 0, so block initially acts as identity
        # alpha1_val: (bs, emb_size) -> (bs, 1, emb_size)
        y = y * alpha1_val.unsqueeze(1)

        # Step 8: Residual connection
        y = x + y  # Add original input

        # ============================================================
        # SUB-BLOCK 2: Feed-Forward Network with AdaLN
        # ============================================================

        # Step 1: Layer normalization
        z = self.ln2(y)  # Shape: (bs, seq_len, emb_size)

        # Step 2: AdaLN modulation (scale and shift)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)

        # Step 3: Feed-forward network (MLP)
        z = self.ff(z)  # Shape: (bs, seq_len, emb_size)

        # Step 4: Alpha scaling
        z = z * alpha2_val.unsqueeze(1)

        # Step 5: Residual connection and return
        return y + z


if __name__ == "__main__":
    # ==== Test the DiT Block ====
    # Create a DiT block with 16-dimensional embeddings and 4 attention heads
    dit_block = DiTBlock(emb_size=256, nhead=8)

    # Create dummy input:
    # - 5 images in the batch
    # - 49 patches per image (e.g., 7x7 grid from a 28x28 image with 4x4 patches)
    # - 256-dimensional embedding per patch
    x = torch.rand((5, 49, 256))

    # Create dummy conditioning vector:
    # - 5 images in the batch
    # - 256-dimensional condition (combined timestep + class embedding)
    cond = torch.rand((5, 256))

    # Forward pass
    outputs = dit_block(x, cond)

    # Output shape should match input: (5, 49, 256)
    assert outputs.shape == (5, 49, 256)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {outputs.shape}")
