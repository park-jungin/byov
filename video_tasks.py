import copy
import os
import numpy as np
import torch
import random
from pytorch_lightning.core import LightningModule
from dataset.video_align_dataset import VideoAlignmentTrainDataset
from models.embedder import Embedder, byov_encoder, byov_decoder
from evaluation.evaluate_features import prepare_data_loader, extract_embedding, classification, frame_retrieval, compute_progression_value, kendalls_tau
from utils.pos_embed import interpolate_pos_embed

class VideoAlignment(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Embedder(args)
        self.encoder = byov_encoder(args)
        self.decoder = byov_decoder(args)
        self.checkpoint_metric = "train_loss"
        self.data_path = None
        self.ds_loader_train, self.ds_dataset_train = prepare_data_loader(args, 'train', batch_size=1)
        self.ds_loader_val, self.ds_dataset_val = prepare_data_loader(args, 'val', batch_size=1)
        print(f'constructing downstream loader train {len(self.ds_loader_train)} | val {len(self.ds_loader_val)}')


    def training_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 32, 3, 224, 224)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)
        x1 = self.model(x1).detach()    # [bs, ts, hidden_dim]
        x2 = self.model(x2).detach()    # [bs, ts, hidden_dim]
        
        embed_ref = torch.cat([x1, x2], dim=1)

        z1, embeds1_r1, embeds1_r2, mask1_r1, mask1_r2, ids_restore1_r1, ids_restore1_r2 = self.encoder(x1)
        z2, embeds2_r1, embeds2_r2, mask2_r1, mask2_r2, ids_restore2_r1, ids_restore2_r2 = self.encoder(x2)
        # embed_ref = torch.cat([embeds1.clone().detach(), embeds2.clone().detach()], dim=1)
        
        mask = torch.zeros((z1.shape[0], z1.shape[1]), device=z1.device)
        
        embed_pred, embed1_pred, embed2_pred = self.decoder(z1, embeds1_r1, embeds1_r2, z2, embeds2_r1, embeds2_r2,
                                                            ids_restore1_r1, ids_restore1_r2, ids_restore2_r1, ids_restore2_r2)
        
        mask1 = torch.cat([mask1_r1, mask2_r1], dim=1)
        loss1 = (embed_pred - embed_ref) ** 2
        loss1 = loss1.mean(dim=-1)
        loss1 = (loss1 * mask1).sum() / mask1.sum()
        
        mask2 = torch.cat([mask1_r2, mask], dim=1)
        loss2 = (embed1_pred - embed_ref) ** 2
        loss2 = loss2.mean(dim=-1)
        loss2 = (loss2 * mask2).sum() / mask2.sum()
        
        mask3 = torch.cat([mask, mask2_r2], dim=1)
        loss3 = (embed2_pred - embed_ref) ** 2
        loss3 = loss3.mean(dim=-1)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        # loss = loss1 #loss2 + loss3
        loss = loss1 + loss2 + loss3 
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 64, 3, 168, 168)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)
        x1 = self.model(x1).detach()    # [bs, ts, hidden_dim]
        x2 = self.model(x2).detach()    # [bs, ts, hidden_dim]
        embed_ref = torch.cat([x1, x2], dim=1)
        
        # Forwarding to encoder
        z1, embeds1_r1, embeds1_r2, mask1_r1, mask1_r2, ids_restore1_r1, ids_restore1_r2 = self.encoder(x1)
        z2, embeds2_r1, embeds2_r2, mask2_r1, mask2_r2, ids_restore2_r1, ids_restore2_r2 = self.encoder(x2)
                
        # Forwarding to decoder
        embed_pred, embed1_pred, embed2_pred = self.decoder(z1, embeds1_r1, embeds1_r2, z2, embeds2_r1, embeds2_r2,
                                                            ids_restore1_r1, ids_restore1_r2, ids_restore2_r1, ids_restore2_r2)

        mask = torch.zeros((z1.shape[0], z1.shape[1]), device=z1.device)

        mask1 = torch.cat([mask1_r1, mask2_r1], dim=1)
        loss1 = (embed_ref - embed_pred) ** 2
        loss1 = loss1.mean(dim=-1)
        loss1 = (loss1 * mask1).sum() / mask1.sum()
        
        mask2 = torch.cat([mask1_r2, mask], dim=1)
        loss2 = (embed_ref - embed1_pred) ** 2
        loss2 = loss2.mean(dim=-1)
        loss2 = (loss2 * mask2).sum() / mask2.sum()
        
        mask3 = torch.cat([mask, mask2_r2], dim=1)
        loss3 = (embed_ref - embed2_pred) ** 2
        loss3 = loss3.mean(dim=-1)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        loss = loss1 + loss2 + loss3 
        embeddings = torch.stack((z1, z2), dim=1)  # (bs, 2, 32, 256)

        # self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.evaluate_downstream(batch_idx, embeddings.device)
        return loss

    def evaluate_downstream(self, batch_idx, device):
        if self.args.ds_every_n_epoch > 0 and self.global_rank == 0 and batch_idx == 0 and (
                self.current_epoch + 1) % self.args.ds_every_n_epoch == 0:
            extract_embedding('train', self.ds_loader_train, self.model, self.encoder, self.args.output_dir, device)
            extract_embedding('val', self.ds_loader_val, self.model, self.encoder, self.args.output_dir, device)

            if '1' in self.args.eval_task:  # classification
                regular_f1, ego2exo_val_f1, exo2ego_val_f1 = classification(self.args.output_dir,
                                                                            self.ds_dataset_train.video_ego_id,
                                                                            self.ds_dataset_val.video_ego_id)
                self.log('regular_f1', regular_f1)
                self.log('ego2exo_val_f1', ego2exo_val_f1)
                self.log('exo2ego_val_f1', exo2ego_val_f1)

            if '2' in self.args.eval_task:  # retrieval
                regular_map10, ego2exo_val_map10, exo2ego_val_map10 = frame_retrieval(self.args.output_dir,
                                                                                      self.ds_dataset_val.video_len_list,
                                                                                      self.ds_dataset_val.video_paths1)
                self.log('regular_map10', float(regular_map10))
                self.log('ego2exo_val_map10', float(ego2exo_val_map10))
                self.log('exo2ego_val_map10', float(exo2ego_val_map10))

            if '3' in self.args.eval_task:  # event completion
                modify_embeddings = True if self.args.dataset == 'pour_liquid' else False  # augment embedding for pour_liquid
                train_score, val_score = compute_progression_value(self.args.output_dir, self.ds_dataset_train.video_len_list,
                                                                   self.ds_dataset_val.video_len_list, modify_embeddings)
                self.log('train_score', train_score)
                self.log('val_score', val_score)

            if '4' in self.args.eval_task:  # kendall's tau
                train_tau = kendalls_tau(self.args.output_dir, self.ds_dataset_train.video_len_list,
                                         self.ds_dataset_train.video_paths1, 'train', False)
                val_tau = kendalls_tau(self.args.output_dir, self.ds_dataset_val.video_len_list,
                                       self.ds_dataset_val.video_paths1, 'val', False)
                self.log('train_tau', train_tau)
                self.log('val_tau', val_tau)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'train')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'val')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

