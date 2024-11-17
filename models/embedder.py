import os
from typing import Optional
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.models
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, AutoModel, AutoProcessor
from models.layer import TemporalBlock

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_1d_sincos_pos_embed_from_grid        


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        self.args = args
        self.num_context_steps = args.num_context_steps
        self.n_layers = args.n_layers

        if args.base_model_name == 'clip':
            self.base_model = CLIPVisionModel.from_pretrained(args.vision_encoder_path)
        else:
            raise NotImplementedError

        if args.freeze_base:
            print("Base model frozen.")
            self.freeze_base_model()

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def token_selection(self, x):
        bs, ts, n, d = x.shape
        topk = round(self.args.num_tokens * self.args.topk_ratio)
        x_select = torch.zeros((bs, ts, topk, d), device=x.device)
        
        for b in range(bs):
            differences = (x[b, :-1] - x[b, 1:]).abs_() # (bs, ts, n-1, d)
            for t in range(ts):
                if t < ts - 1:
                    diffs = differences[t].mean(dim=-1)
                else:
                    diffs = differences[t-1].mean(dim=-1)
                topk_idx = torch.topk(diffs, topk, largest=True).indices
                x_select[b, t] = x[b, t, topk_idx]
        return x_select, topk_idx

    def forward(self, x):
        # x: (bs, ts=32, 3, 224/168, 224/168)
        bs, ts, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = self.base_model(x).last_hidden_state  # (bs*ts, 3, 224, 224) -> (bs*ts, 50, 768) or (bs*ts, 196, 768)
        if 'clip' in self.args.base_model_name:
            x = x[:, 1:, :] # Take tokens without [cls] token   (bs*ts, 49, 768) for CLIP ViT-B/32
        _, n, d = x.shape
        x = x.reshape(bs, ts, n, d)
        x, topk_idx = self.token_selection(x)
        x = x.mean(dim=2)
        return x


class byov_encoder(nn.Module):

    def __init__(self, args):
        super(byov_encoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.embedding_size = args.embedding_size
        self.num_frames = args.num_frames
        # self.token_selection_attn = Block(args.hidden_dim, args.n_heads, qkv_bias=True, norm_layer=nn.LayerNorm)
        # self.gate_layer = nn.Sequential(
        #     nn.Linear(args.hidden_dim, 1),
        #     nn.Softmax(dim=-1)
        # )
        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.embedding_size)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, args.num_frames, args.hidden_dim), requires_grad=False)
        self.encoder = nn.ModuleList([
            Block(args.embedding_size, args.n_heads, args.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(args.n_layers)
        ])
        self.norm = nn.LayerNorm(args.embedding_size)

        self.initialize_weights()

    def initialize_weights(self):
        pos = np.array([i for i in range(self.num_frames)])
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def future_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        ids = torch.arange(0, L, device=x.device).unsqueeze(0)
        ids = torch.cat([ids] * N, dim=0)
        ids_restore = torch.argsort(ids, dim=1)
        ids_keep = ids[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device = x.device)
        mask[:, :len_keep] = 0
        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):        
        if x.shape[1] != self.num_frames:
            new_pos_embed = torch.nn.functional.interpolate(self.pos_embed.permute(0, 2, 1), size=(x.shape[1]), mode='linear').permute(0, 2, 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x_r1, mask_r1, ids_restore_r1 = self.random_masking(x, self.args.mask_ratio)    # small number masking 
        x_r2, mask_r2, ids_restore_r2 = self.random_masking(x, self.args.mask_ratio * 2)    # large number masking
        x = self.embedding_layer(x)
        x_r1 = self.embedding_layer(x_r1)
        x_r2 = self.embedding_layer(x_r2)
        x = self.norm(x)
        x_r1 = self.norm(x_r1)
        x_r2 = self.norm(x_r2)
        for blk in self.encoder:
            x = blk(x)
            x_r1 = blk(x_r1)
            x_r2 = blk(x_r2)
        return x, x_r1, x_r2, mask_r1, mask_r2, ids_restore_r1, ids_restore_r2


class byov_decoder(nn.Module):
    def __init__(self, args):
        super(byov_decoder, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.decoder_embedding_size = args.decoder_embedding_size
        self.decoder_output_size = args.hidden_dim
        self.num_frames = args.num_frames

        self.decoder_embed = nn.Sequential(
            nn.LayerNorm(args.embedding_size),
            nn.Linear(args.embedding_size, args.decoder_embedding_size, bias=True)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embedding_size))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, args.num_frames, args.decoder_embedding_size), requires_grad=False)
        self.decoder = nn.ModuleList([
            TemporalBlock(args.decoder_embedding_size, args.n_heads, args.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(args.n_layers_dec)
        ])
        self.decoder_pred = nn.Sequential(
            nn.LayerNorm(args.decoder_embedding_size),
            nn.Linear(args.decoder_embedding_size, args.hidden_dim, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        pos = np.array([i for i in range(self.num_frames)])

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], pos)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token,std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x1_r1, x1_r2, x2, x2_r1, x2_r2, ids_restore1_r1, ids_restore1_r2, ids_restore2_r1, ids_restore2_r2):
        x1 = self.decoder_embed(x1)
        x2 = self.decoder_embed(x2)
        x1_r1 = self.decoder_embed(x1_r1)   # small number masking
        x1_r2 = self.decoder_embed(x1_r2)   # large number masking
        x2_r1 = self.decoder_embed(x2_r1)   # small number masking
        x2_r2 = self.decoder_embed(x2_r2)   # large number masking

        x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed

        mask_token1_r1 = self.mask_token.repeat(x1_r1.shape[0], ids_restore1_r1.shape[1] + 1 - x1_r1.shape[1], 1)
        x1_r1_ = torch.cat([x1_r1, mask_token1_r1], dim=1)
        x1_r1 = torch.gather(x1_r1_, dim=1, index=ids_restore1_r1.unsqueeze(-1).repeat(1, 1, x1_r1.shape[2]))
        x1_r1 = x1_r1 + self.decoder_pos_embed

        mask_token1_r2 = self.mask_token.repeat(x1_r2.shape[0], ids_restore1_r2.shape[1] + 1 - x1_r2.shape[1], 1)
        x1_r2_ = torch.cat([x1_r2, mask_token1_r2], dim=1)
        x1_r2 = torch.gather(x1_r2_, dim=1, index=ids_restore1_r2.unsqueeze(-1).repeat(1, 1, x1_r2.shape[2]))
        x1_r2 = x1_r2 + self.decoder_pos_embed

        mask_token2_r1 = self.mask_token.repeat(x2_r1.shape[0], ids_restore2_r1.shape[1] + 1 - x2_r1.shape[1], 1)
        x2_r1_ = torch.cat([x2_r1, mask_token2_r1], dim=1)
        x2_r1 = torch.gather(x2_r1_, dim=1, index=ids_restore2_r1.unsqueeze(-1).repeat(1, 1, x2_r1.shape[2]))
        x2_r1 = x2_r1 + self.decoder_pos_embed

        mask_token2_r2 = self.mask_token.repeat(x2_r2.shape[0], ids_restore2_r2.shape[1] + 1 - x2_r2.shape[1], 1)
        x2_r2_ = torch.cat([x2_r2, mask_token2_r2], dim=1)
        x2_r2 = torch.gather(x2_r2_, dim=1, index=ids_restore2_r2.unsqueeze(-1).repeat(1, 1, x2_r2.shape[2]))
        x2_r2 = x2_r2 + self.decoder_pos_embed

        x1_pred = torch.cat([x1_r2, x2], dim=1)
        x2_pred = torch.cat([x1, x2_r2], dim=1)
        for blk in self.decoder:
            x1_r1 = blk(x1_r1, mask=True)
            x1_pred = blk(x1_pred, mask=False)
            x2_r1 = blk(x2_r1, mask=True)
            x2_pred = blk(x2_pred, mask=False)

        x1_r1 = self.decoder_pred(x1_r1)
        x2_r1 = self.decoder_pred(x2_r2)
        x1_pred = self.decoder_pred(x1_pred)
        x2_pred = self.decoder_pred(x2_pred)
        x_pred = torch.cat([x1_r1, x2_r1], dim=1)
        return x_pred, x1_pred, x2_pred