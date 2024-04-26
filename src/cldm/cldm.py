import einops
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision.transforms.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, predicted_mask=None, pose_control=None, **kwargs):
        if predicted_mask is not None:
            # remove from the computation graph
            if not isinstance(predicted_mask, list):
                predicted_mask = [predicted_mask]
            for i in range(len(predicted_mask)):
                predicted_mask[i] = predicted_mask[i].detach()

        hs = []

        if (hasattr(self, 'sd_encoder_locked') and not self.sd_encoder_locked):
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
        else:
            with torch.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                for module in self.input_blocks:
                    h = module(h, emb, context)
                    hs.append(h)
        if (hasattr(self, 'sd_encoder_locked') and not self.sd_encoder_locked):
            h = self.middle_block(h, emb, context)
        else:
            with torch.no_grad():
                h = self.middle_block(h, emb, context)

        if control is not None:
            for i, c in enumerate(control):
                feat_map = c.pop()
                if predicted_mask is not None:
                    mask = predicted_mask[i]
                    mask = F.resize(mask, (h.shape[-2:]))
                    feat_map = feat_map * ((mask > 0.5).float()) # makes hard mask 0/1
                h = h + feat_map
                if pose_control is not None:
                    pose_c = pose_control[i]
                    feat_map_pose = pose_c.pop()
                    h = h + feat_map_pose

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None or (control is None and pose_control is None):
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                hsp = hs.pop()

                if control is not None:
                    for i, c in enumerate(control):
                        feat_map = c.pop()

                        if predicted_mask is not None:
                            mask = predicted_mask[i]
                            mask = F.resize(mask, (h.shape[-2:]))
                            feat_map = feat_map * ((mask > 0.5).float()) # makes hard mask 0/1
                        hsp = hsp + feat_map

                        if pose_control is not None:
                            pose_c = pose_control[i]
                            feat_map_pose = pose_c.pop()
                            hsp = hsp + feat_map_pose

                out_h = hsp
                h = torch.cat([h, out_h], dim=1)

            h = module(h, emb, context)

        h = h.type(x.dtype)

        return self.out(h)


class CapFace(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            cond_mode=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.cond_mode = cond_mode
        self.mask_predict = nn.Sequential(conv_nd(2, 320, 1, 3, padding=1, stride=1),
                                           nn.Sigmoid())

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, output_mask=True, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        predicted_mask = self.mask_predict(guided_hint)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        if output_mask:
            return {'control': outs,
                    'mask': predicted_mask}
        else:
            return {'control': outs,
                    'mask': None}


class CapHuman(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, control_cond_stage_config, control_stage_key, cond_mode, context_dim, img_linear_dim, control_id_key, mask_loss_weight=1.0, drop_control_cond_t=0,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_cond_stage_model = instantiate_from_config(control_cond_stage_config)
        self.control_key = control_key
        self.control_stage_key =  control_stage_key
        self.control_id_key = control_id_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.cond_mode = cond_mode
        self.mask_loss_weight = mask_loss_weight
        self.drop_control_cond_t = drop_control_cond_t
        self.id_linear = nn.Linear(512, context_dim)
        self.img_linear = nn.Linear(img_linear_dim, context_dim)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if 'img' in self.cond_mode:
            control_cond = batch[self.control_stage_key]
        if 'id' in self.cond_mode:
            control_id_feat = batch[self.control_id_key]
        if bs is not None:
            control = control[:bs]
            if 'img' in self.cond_mode:
                control_cond = control_cond[:bs]
            if 'id' in self.cond_mode:
                control_id_feat = control_id_feat[:bs]

        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        return_dict = dict(c_crossattn=[c], c_concat=[control])

        if 'img' in self.cond_mode:
            control_cond = self.control_cond_stage_model.encode(control_cond)
            return_dict['c_crossattn_control'] = [control_cond]
        if 'id' in self.cond_mode:
            return_dict['c_crossattn_id'] = [control_id_feat]
        if 'full_mask' in batch:
            return_dict['full_mask'] = batch['full_mask']
        if 'data_weight' in batch:
            return_dict['data_weight'] = batch['data_weight']
        if 'fuse_weight' in batch:
            return_dict['fuse_weight'] = batch['fuse_weight']

        return x, return_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):

        diffusion_model = self.model.diffusion_model
        output_mask = self.mask_loss_weight != 0

        if not isinstance(cond, list):
            conds = [cond]
        else:
            conds = cond

        cond_imgs = []
        cond_ids = []
        for cond in conds:
            cond_txt = torch.cat(cond['c_crossattn'], 1)
            if 'img' in self.cond_mode and 'c_crossattn_control' in cond and cond['c_crossattn_control'] is not None:
                cond_img = torch.cat(cond['c_crossattn_control'], 1)
                cond_imgs.append(cond_img)
            if 'id' in self.cond_mode and 'c_crossattn_id' in cond and cond['c_crossattn_id'] is not None:
                cond_id = torch.cat(cond['c_crossattn_id'], 1)
                cond_ids.append(cond_id)

        control_conds = []
        for i in range(len(conds)):
            control_cond_list = []
            if 'img' in self.cond_mode:
                cond_img = cond_imgs[i]
                cond_img = self.img_linear(cond_img)
                control_cond_list.append(cond_img)
            if 'id' in self.cond_mode:
                cond_id = cond_ids[i]
                cond_id = self.id_linear(cond_id)
                control_cond_list.append(cond_id)

            control_cond = torch.cat(control_cond_list, dim=1)

            if self.drop_control_cond_t != 0:
                drop_or_not = (t <= self.drop_control_cond_t).float().unsqueeze(1).unsqueeze(1)
                control_cond = control_cond*drop_or_not

            control_conds.append(control_cond)

        control = []
        predicted_mask = {'predicted_mask': []}

        for cond, control_cond in zip(conds, control_conds):
            out = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=control_cond, output_mask=output_mask, **cond)
            control.append(out['control'])
            predicted_mask['predicted_mask'].append(out['mask'])
            cond['predicted_mask'] = out['mask']

        for i in range(len(control)):
            control[i] = [c * scale for c, scale in zip(control[i], self.control_scales)]

        pose_controls = None
        if hasattr(self, 'pose_control_model') and 'c_pose_concat' in conds[0]: # inject
            pose_controls = []
            for cond in conds:
                pose_control = self.pose_control_model(x=x_noisy, hint=torch.cat(cond['c_pose_concat'], 1), timesteps=t, context=cond_txt)
                pose_controls.append(pose_control)
        if pose_controls is not None:
            for i in range(len(pose_controls)):
                pose_controls[i] = [c*scale for c, scale in zip(pose_controls[i], self.pose_control_scales)]

        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control, pose_control=pose_controls, **predicted_mask)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        unet_lr = self.unet_lr
        params = []
        params += [{'params': x} for x in list(self.control_model.parameters())]
        if 'id' in self.cond_mode:
            params += [{'params': x} for x in list(self.id_linear.parameters())]
        if 'img' in self.cond_mode:
            params += [{'params': x} for x in list(self.img_linear.parameters())]
        if not self.sd_encoder_locked:
            params += [{'params': x, 'lr': unet_lr} for x in list(self.model.diffusion_model.input_blocks.parameters())]
            params += [{'params': x, 'lr': unet_lr} for x in list(self.model.diffusion_model.middle_block.parameters())]
        if not self.sd_locked:
            params += [{'params': x, 'lr': unet_lr} for x in list(self.model.diffusion_model.output_blocks.parameters())]
            params += [{'params': x, 'lr': unet_lr} for x in list(self.model.diffusion_model.out.parameters())]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss = self.get_loss(output, target, mean=False)
        loss_simple = loss.mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)

        # mask loss
        if self.mask_loss_weight != 0:
            gt_mask = torch.where((cond['full_mask'] == 0) | (cond['full_mask'] == 7) | (cond['full_mask'] == 8) | (cond['full_mask'] == 9) | (cond['full_mask'] == 14) | (cond['full_mask'] == 15) | (cond['full_mask'] == 16) | (cond['full_mask'] == 17) | (cond['full_mask'] == 18), torch.zeros_like(cond['full_mask']), torch.ones_like(cond['full_mask'])) # remove ear and hat
            gt_mask = F.resize(gt_mask, output.shape[-1])
            gt_mask = gt_mask.unsqueeze(1)

            loss_mask = self.mask_loss_weight * ((gt_mask - cond['predicted_mask'])**2).mean()
            loss_dict.update({f'{prefix}/loss_mask': loss_mask})

            loss += loss_mask

        loss_dict.update({f'{prefix}/loss': loss})


        return loss, loss_dict
