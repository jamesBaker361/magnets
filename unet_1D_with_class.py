from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block
from diffusers.models.unets.unet_1d import UNet1DModel, UNet1DOutput

class Unet1DModelCLassy(UNet1DModel):
    def __init__(self, sample_size = 65536, sample_rate = None, in_channels = 2, out_channels = 2, extra_in_channels = 0, time_embedding_type = "fourier", flip_sin_to_cos = True, use_timestep_embedding = False, freq_shift = 0,
                  down_block_types : Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"), 
                  up_block_types:Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")
                  , mid_block_type = "UNetMidBlock1D",
                    out_block_type = None, 
                    block_out_channels :Tuple[int] = (32, 32, 64),
                      act_fn = "swish", 
                      norm_num_groups = 8, layers_per_block = 1, downsample_each_block = False,use_class_embedding=True,num_classes:int=-1):
        super().__init__(sample_size, sample_rate, in_channels, out_channels, extra_in_channels, time_embedding_type, flip_sin_to_cos, use_timestep_embedding, freq_shift, down_block_types, up_block_types, mid_block_type, out_block_type, block_out_channels, act_fn, norm_num_groups, layers_per_block, downsample_each_block)
        
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        self.use_class_embedding=use_class_embedding
        if use_timestep_embedding and use_class_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0]//2,
            )
            self.class_embedding=torch.nn.Embedding(num_classes,block_out_channels[0]//2)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        print(timestep_embed.shape)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed.to(sample.dtype))
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(sample.dtype)
            timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))

        if self.use_class_embedding:
            class_embed=self.class_embedding(class_labels)
            timestep_embed=torch.cat([timestep_embed,class_embed],dim=-1)

        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=timestep_embed)
            down_block_res_samples += res_samples

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=timestep_embed)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)