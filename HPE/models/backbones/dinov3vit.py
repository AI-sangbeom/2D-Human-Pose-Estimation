import torch
from torch import Tensor
from typing import Optional, Union, List, Dict
from models.utils import convert_path_or_url_to_url, Weights

dinov3_vit = torch.hub.load(
    "thirdparty/dinov3", 
    'dinov3_vits16'
    , source='local',
)

VisionTransformer = dinov3_vit.__class__
dinov3_vit = None

class Dinov3ViT(VisionTransformer):
    def __init__(
            self,     
            *,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            pos_embed_rope_base: float = 100.0,
            pos_embed_rope_min_period: float | None = None,
            pos_embed_rope_max_period: float | None = None,
            pos_embed_rope_normalize_coords: str = "separate",
            pos_embed_rope_shift_coords: float | None = None,
            pos_embed_rope_jitter_coords: float | None = None,
            pos_embed_rope_rescale_coords: float | None = None,
            pos_embed_rope_dtype: str = "fp32",
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            ffn_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.0,
            layerscale_init: float | None = None,
            norm_layer: str = "layernorm",
            ffn_layer: str = "mlp",
            ffn_bias: bool = True,
            proj_bias: bool = True,
            n_storage_tokens: int = 0,
            mask_k_bias: bool = False,
            pretrained: bool = False,
            weights: Union[Weights, str] = Weights.LVD1689M,
            hash: Optional[str] = None,
            check_hash: bool = False,
            **kwargs
        ):
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
        )
        vit_kwargs.update(**kwargs)
        super(Dinov3ViT, self).__init__(**vit_kwargs)
        if pretrained:
            if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
                raise ValueError(f"Unsupported weights for the backbone: {weights}")
            url = convert_path_or_url_to_url(weights)
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
            self.load_state_dict(state_dict, strict=True)
        else:
            self.init_weights()

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        all_xes = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
            all_xes.append(x)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output, all_xes

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0][0]
        else:
            return self.forward_features_list(x, masks)[0]


    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        
vit_sizes = {
    "small": dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
    ),
    "base": dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
    ),
    "large": dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
    ),

}

vit_ckps = {
    'small': './checkpoints/dinov3/vit/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 
    'base': './checkpoints/dinov3/vit/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth', 
    'large': None
    }