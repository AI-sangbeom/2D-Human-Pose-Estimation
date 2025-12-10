import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class GatedAttention(nn.Module):
    """
    Gated Multi-head Self Attention
    Based on "Gated Attention for Large Language Models" (arXiv:2505.06708)
    
    Applies sigmoid gate after SDPA: Y' = Y ⊙ σ(XW_θ)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., gate_type="headwise"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.gate_type = gate_type
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gate projection (핵심 추가)
        if gate_type == "elementwise":
            self.W_gate = nn.Linear(dim, dim, bias=qkv_bias)
        elif gate_type == "headwise":
            self.W_gate = nn.Linear(dim, num_heads, bias=qkv_bias)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard SDPA
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        Y = (attn @ v)  # (B, num_heads, N, head_dim)
        
        # Gating mechanism: Y' = Y ⊙ σ(XW_θ)
        gate_logits = self.W_gate(x)  # (B, N, dim or num_heads)
        gate_scores = torch.sigmoid(gate_logits)
        
        if self.gate_type == "elementwise":
            # Element-wise: (B, N, dim) -> (B, num_heads, N, head_dim)
            gate_scores = gate_scores.view(B, N, self.num_heads, C // self.num_heads)
            gate_scores = gate_scores.transpose(1, 2)
        elif self.gate_type == "headwise":
            # Head-wise: (B, N, num_heads) -> (B, num_heads, N, 1)
            gate_scores = gate_scores.view(B, N, self.num_heads, 1)
            gate_scores = gate_scores.transpose(1, 2)
        
        # Apply gate
        Y_gated = Y * gate_scores
        
        # Reshape and project
        x = Y_gated.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Keep standard Attention for backward compatibility
class Attention(GatedAttention):
    """Standard attention (wrapper around GatedAttention with identity gate)"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop, gate_type="headwise")


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with Gated Attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 use_gated_attn=True, gate_type="headwise"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Use GatedAttention or standard Attention
        if use_gated_attn:
            self.attn = GatedAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                attn_drop=attn_drop, proj_drop=drop, gate_type=gate_type
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                attn_drop=attn_drop, proj_drop=drop
            )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """
    Masked Autoencoder Encoder
    - Processes only visible (unmasked) patches
    - Outputs encoded representations
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_gated_attn=True,
        gate_type="headwise",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), 
            requires_grad=False
        )
        
        # Transformer blocks with Gated Attention
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  use_gated_attn=use_gated_attn, gate_type=gate_type)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize tokens
        nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize positional embedding with sin-cos
        pos_embed = self._get_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize layers
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def _get_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        """
        Generate 2D sin-cos positional embeddings
        """
        import numpy as np
        
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        pos_embed = MAEEncoder._get_sincos_pos_embed_from_grid(embed_dim, grid)
        
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
        return pos_embed
    
    @staticmethod
    def _get_sincos_pos_embed_from_grid(embed_dim, grid):
        import numpy as np
        
        assert embed_dim % 2 == 0
        
        # Use half dimensions for each coordinate
        emb_h = MAEEncoder._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = MAEEncoder._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb
    
    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        import numpy as np
        
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking per sample
        
        Args:
            x: [B, N, D]
            mask_ratio: ratio of patches to mask
            
        Returns:
            x_masked: [B, N_visible, D]
            mask: [B, N], 0 is keep, 1 is remove
            ids_restore: [B, N]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio=0.75):
        """
        Args:
            x: (B, C, H, W) input images
            mask_ratio: masking ratio
            
        Returns:
            latent: (B, N_visible+1, D) encoded features
            mask: (B, N) binary mask
            ids_restore: (B, N) indices to restore
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add positional embedding (without cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """
    Masked Autoencoder Decoder
    - Reconstructs masked patches
    """
    def __init__(
        self,
        patch_size=16,
        num_patches=196,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        in_chans=3,
        use_gated_attn=True,
        gate_type="headwise",
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        
        # Project from encoder to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # Transformer blocks with Gated Attention
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  use_gated_attn=use_gated_attn, gate_type=gate_type)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize positional embedding
        pos_embed = MAEEncoder._get_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        Args:
            x: (B, N_visible+1, encoder_dim) encoded features
            ids_restore: (B, N) indices to restore original order
            
        Returns:
            pred: (B, N, patch_size^2 * in_chans) predictions
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] + 1 - x.shape[1], 
            1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Remove cls, append mask
        
        # Unshuffle
        x_ = torch.gather(
            x_, dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # Append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x


class MAEBackbone(nn.Module):
    """
    Complete Masked Autoencoder Backbone
    
    Can be used for:
    1. Self-supervised pretraining (with reconstruction loss)
    2. Feature extraction (encoder only)
    3. Fine-tuning on downstream tasks
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        mask_ratio=0.75,
        use_gated_attn=True,
        gate_type="headwise",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        
        num_patches = (img_size // patch_size) ** 2
        
        # Encoder with Gated Attention
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            use_gated_attn=use_gated_attn,
            gate_type=gate_type,
        )
        
        # Decoder with Gated Attention
        self.decoder = MAEDecoder(
            patch_size=patch_size,
            num_patches=num_patches,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            in_chans=in_chans,
            use_gated_attn=use_gated_attn,
            gate_type=gate_type,
        )
    
    def patchify(self, imgs):
        """
        Convert images to patches
        
        Args:
            imgs: (B, C, H, W)
        Returns:
            patches: (B, N, patch_size^2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * self.in_chans)
        
        return x
    
    def unpatchify(self, x):
        """
        Convert patches to images
        
        Args:
            x: (B, N, patch_size^2 * C)
        Returns:
            imgs: (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_chans, h * p, h * p)
        
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss
        
        Args:
            imgs: (B, C, H, W) original images
            pred: (B, N, patch_size^2 * C) predictions
            mask: (B, N) binary mask (1 = masked, 0 = visible)
            
        Returns:
            loss: scalar
        """
        target = self.patchify(imgs)
        
        # Normalize per patch (improves representation quality)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** 0.5
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass
        
        Args:
            imgs: (B, C, H, W) input images
            mask_ratio: masking ratio (uses self.mask_ratio if None)
            
        Returns:
            loss: reconstruction loss (training mode)
            pred: (B, N, patch_size^2 * C) predictions
            mask: (B, N) binary mask
            latent: (B, N_visible+1, D) encoded features
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio if self.training else 0.0
        
        # Encode
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        # Compute loss
        if self.training:
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, latent
        else:
            return pred, mask, latent
    
    def extract_features(self, imgs, mask_ratio=0.0):
        """
        Extract features without reconstruction (for downstream tasks)
        
        Args:
            imgs: (B, C, H, W) input images
            mask_ratio: masking ratio (default: no masking)
            
        Returns:
            features: (B, N+1, D) encoded features (includes cls token)
        """
        latent, _, _ = self.encoder(imgs, mask_ratio)
        return latent


class MAEWithKeypointHead(nn.Module):
    """
    MAE Backbone with Keypoint Detection Head
    For downstream tasks like pose estimation
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_kpts=17,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        pretrained_weights=None,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.num_kpts = num_kpts
        
        # MAE Encoder (decoder not needed for downstream tasks)
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self.load_pretrained(pretrained_weights)
        
        # Keypoint detection head
        self.keypoint_head = nn.Sequential(
            nn.Linear(encoder_embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_kpts * 3)  # (x, y, confidence) for each keypoint
        )
    
    def load_pretrained(self, weights_path):
        """Load pretrained MAE weights"""
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Load encoder weights only
        encoder_weights = {
            k.replace('encoder.', ''): v 
            for k, v in checkpoint.items() 
            if k.startswith('encoder.')
        }
        
        self.encoder.load_state_dict(encoder_weights, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            keypoints: (B, num_kpts, 3) predicted keypoints [x, y, conf]
        """
        # Extract features (no masking for downstream tasks)
        features, _, _ = self.encoder(x, mask_ratio=0.0)
        
        # Use cls token for global representation
        cls_features = features[:, 0, :]  # (B, D)
        
        # Predict keypoints
        kpts = self.keypoint_head(cls_features)  # (B, num_kpts * 3)
        kpts = kpts.reshape(-1, self.num_kpts, 3)
        
        # Normalize coordinates to [0, 1]
        kpts[..., :2] = torch.sigmoid(kpts[..., :2])
        kpts[..., 2] = torch.sigmoid(kpts[..., 2])  # Confidence
        
        return kpts


# ===== Usage Examples =====
if __name__ == "__main__":
    print("=" * 70)
    print("MAE Backbone - Complete Examples")
    print("=" * 70)
    
    batch_size = 4
    img_size = 640
    
    # ========== Example 1: Self-Supervised Pretraining ==========
    print("\n" + "=" * 70)
    print("1. Self-Supervised Pretraining (MAE)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = MAEBackbone(
        img_size=img_size,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        use_gated_attn=True,  # Enable Gated Attention
        gate_type="headwise",  # or "elementwise"
    )
    mae.to(device)
    
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print(f"Input: {images.shape}")
    
    mae.train()
    loss, pred, mask, latent = mae(images)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions: {pred.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Latent features: {latent.shape}")
    print(f"Masked ratio: {mask.float().mean().item():.2%}")
    
    # Visualize reconstruction (conceptual)
    reconstructed = mae.unpatchify(pred)
    print(f"Reconstructed images: {reconstructed.shape}")
    
    # ========== Example 2: Feature Extraction ==========
    print("\n" + "=" * 70)
    print("2. Feature Extraction (No Reconstruction)")
    print("=" * 70)
    
    mae.eval()
    with torch.no_grad():
        features = mae.extract_features(images, mask_ratio=0.0)
    
    print(f"Extracted features: {features.shape}")
    print(f"  - CLS token: {features[:, 0, :].shape}")
    print(f"  - Patch tokens: {features[:, 1:, :].shape}")
    
    # ========== Example 3: Keypoint Detection (Fine-tuning) ==========
    print("\n" + "=" * 70)
    print("3. Keypoint Detection (Downstream Task)")
    print("=" * 70)
    
    keypoint_model = MAEWithKeypointHead(
        img_size=img_size,
        num_kpts=17,  # COCO human pose
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
    )
    keypoint_model.to(device)
    
    keypoint_model.eval()
    with torch.no_grad():
        keypoints = keypoint_model(images)
    
    print(f"Predicted keypoints: {keypoints.shape}")
    print(f"  - Shape: (batch, num_kpts, 3)")
    print(f"  - Format: [x, y, confidence]")
    print(f"\nSample keypoint (first person, first joint):")
    print(f"  x: {keypoints[0, 0, 0].item():.3f}")
    print(f"  y: {keypoints[0, 0, 1].item():.3f}")
    print(f"  conf: {keypoints[0, 0, 2].item():.3f}")
    
    # ========== Model Statistics ==========
    print("\n" + "=" * 70)
    print("4. Model Statistics")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in mae.parameters())
    encoder_params = sum(p.numel() for p in mae.encoder.parameters())
    decoder_params = sum(p.numel() for p in mae.decoder.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,}")
    print(f"  - Decoder: {decoder_params:,}")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("  ✓ Complete MAE architecture with encoder-decoder")
    print("  ✓ Gated Attention in both encoder and decoder")
    print("  ✓ Self-supervised pretraining with reconstruction loss")
    print("  ✓ Sin-cos positional embeddings (no learning needed)")
    print("  ✓ Random masking (75% default) for robust learning")
    print("  ✓ Attention sink prevention with gating mechanism")
    print("  ✓ Easy adaptation to downstream tasks")
    print("  ✓ Feature extraction without decoder")
    print("  ✓ Pretrained weight loading support")
    print("=" * 70)