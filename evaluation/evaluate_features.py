import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.config import argparser
from utils.load_model import load_ckpt
from models.embedder import Embedder, byov_encoder
from dataset.video_align_dataset import VideoAlignmentDownstreamDataset
from evaluation.kendalls_tau import kendalls_tau
from evaluation.frame_retrieval import frame_retrieval
from evaluation.event_completion import compute_progression_value
from evaluation.classification import classification


def prepare_data_loader(args, mode, batch_size=1, num_workers=0):
    dataset = VideoAlignmentDownstreamDataset(args, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    print(f'Data loader len {len(data_loader)}')
    return data_loader, dataset


def extract_embedding(mode, data_loader, base_model, encoder, save_path, device):
    embeds_list = []
    labels_list = []
    for batch in tqdm(data_loader):
        frame, frame_label, video_path = batch
        frame = frame.reshape(1, -1, *frame.shape[-3:])          # frame-(1, 64, 224, 224, 3)
        frame = frame.permute(0, 1, 4, 2, 3).float().to(device)  # (1, 64, 3, 224, 224)
        with torch.no_grad():
            embeds = base_model(frame)
            embeds, _, _, _, _, _, _ = encoder(embeds)  # [1, num frames, 256]
        # print(embeds.shape)
        embeds = embeds.squeeze().cpu().numpy()
        embeds_list.append(embeds)
        frame_label = [label.numpy() for label in frame_label]
        labels_list.append(np.array(frame_label))

    embeds = np.concatenate(embeds_list, axis=0)
    np.save(f'{save_path}/{mode}_embeds.npy', embeds)
    print(f'Saved {mode} embeds to {save_path}/{mode}_embeds.npy')
    labels = np.concatenate(labels_list, axis=0)
    labels = np.squeeze(labels)
    np.save(f'{save_path}/{mode}_label.npy', labels)
    print(f'Saved {mode} labels to {save_path}/{mode}_label.npy')


def main():
    device = torch.device("cuda:0")
    args = argparser.parse_args()
    assert args.eval_mode in ['val', 'test']

    # prepare data loader
    loader_train, dataset_train = prepare_data_loader(args, 'train', batch_size=1, num_workers=args.num_workers)
    loader_val, dataset_val = prepare_data_loader(args, args.eval_mode, batch_size=1, num_workers=args.num_workers)

    assert args.ckpt != ''
    print('*' * 10, f'Evaluating {args.ckpt} on {args.eval_mode} set', '*' * 10)
    save_path = args.ckpt.replace('.ckpt', '_eval')
    os.makedirs(save_path, exist_ok=True)
    if args.extract_embedding:
        base_model = Embedder(args).to(device)
        encoder = byov_encoder(args).to(device)
        base_model.eval()
        encoder.eval()
        load_ckpt(encoder, args.ckpt)
        extract_embedding('train', loader_train, base_model, encoder, save_path, device)
        extract_embedding('val', loader_val, base_model, encoder, save_path, device)
        print(f'Extracting embedding to {save_path}')
    else:
        print(f'Loading embedding from {save_path}')

    # evaluate embedding
    if '1' in args.eval_task:  # classification
        regular_f1, ego2exo_val_f1, exo2ego_val_f1 = classification(save_path, dataset_train.video_ego_id, dataset_val.video_ego_id)
        print(f'Classification (F1 score): regular={regular_f1:.4f} | ego2exo={ego2exo_val_f1:.4f} | exo2ego={exo2ego_val_f1:.4f}')
        print('-' * 50)

    if '2' in args.eval_task:  # retrieval
        regular_map10, ego2exo_val_map10, exo2ego_val_map10 = frame_retrieval(save_path, dataset_val.video_len_list, dataset_val.video_paths1)
        print(f'Frame retrieval (MAP@10): regular={regular_map10:.4f} | ego2exo={ego2exo_val_map10:.4f} | exo2ego={exo2ego_val_map10:.4f}')
        print('-' * 50)

    if '3' in args.eval_task:  # event completion
        modify_embeddings = True if args.dataset == 'pour_liquid' else False  # augment embedding for pour_liquid
        train_score, val_score = compute_progression_value(save_path, dataset_train.video_len_list, dataset_val.video_len_list, modify_embeddings)
        print(f'Phase progression score = {val_score:.4f}')
        print('-' * 50)

    if '4' in args.eval_task:  # kendall's tau
        train_tau = kendalls_tau(save_path, dataset_train.video_len_list, dataset_train.video_paths1, 'train', False)
        val_tau = kendalls_tau(save_path, dataset_val.video_len_list, dataset_val.video_paths1, 'val', False)
        print(f'Kendall\'s tau = {val_tau:.4f}')
        print('-' * 50)


if __name__ == '__main__':
    main()




