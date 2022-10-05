import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import argparse
import json

from utils import AverageMeter, emd_loss, calc_aesthetic_metrics
from dataset import CustomImageFolder, AVADataset

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--data_root', type=str, default= "/ssd1t/song/Datasets/AVA/shortEdge256/all_images")
    parser.add_argument('--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    model, preprocess = clip.load(cfg.arch, device=cfg.device)
    dataset = AVADataset(csv_file='/home/ps/JJ_Projects/FromSong/dsmAVA/csvFiles/val_mlsp.csv', root_dir='/ssd1t/song/Datasets/AVA/shortEdge256/all_images', transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=10, pin_memory=True)
    # prompts = ['An aesthetically negative picture.'] * 5 + ['An aesthetically positive picture.'] * 5
    prompts = [ f'An image with an aesthetic score of {score} on a scale of 1-10.' for score in ['cat', 'dog'] ]
    text_inputs = clip.tokenize(prompts).to(cfg.device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    _loss = AverageMeter()
    _acc = AverageMeter()
    _plcc_mean = AverageMeter()
    _srcc_mean = AverageMeter()
    _plcc_std = AverageMeter()
    _srcc_std = AverageMeter()


    with torch.no_grad():
        for data in tqdm(dataloader):
            image = data["image"].to(cfg.device)
            print(data['annotations'].shape)
            label = data["annotations"].to(cfg.device)
            bin_label = data["bin_cls"]

            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            output = (image_features @ text_features.T).softmax(dim=-1)
            print(output)

            loss = emd_loss(output, label)
            acc, plcc_mean, srcc_mean, plcc_std, srcc_std = calc_aesthetic_metrics(output, label, bin_label, output.device)

            _loss.update(loss.item())
            _plcc_mean.update(plcc_mean)
            _srcc_mean.update(srcc_mean)
            _plcc_std.update(plcc_std)
            _srcc_std.update(srcc_std)
            _acc.update(acc)

        print("emd_loss: {:.4f}, acc: {:.2f}, cc: {:.4f}, {:.4f}, {:.4f}, {:.4f}"
        .format( _loss.avg, _acc.avg, _plcc_mean.avg, _srcc_mean.avg, _plcc_std.avg, _srcc_std.avg))


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)
