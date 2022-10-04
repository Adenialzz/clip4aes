import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import argparse
import json

from dataset import CustomImageFolder

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--data_root', type=str, default= "/ssd1t/song/Datasets/AVA/shortEdge256/all_images")
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'])
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    model, preprocess = clip.load(cfg.clip_arch, device=cfg.device)
    dataset = CustomImageFolder(cfg.data_root, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=10, pin_memory=True)

    results = dict()
    print(f'Forwarding clip ({cfg.clip_arch}) on AVA ({len(dataset)}).')
    with torch.no_grad():
        for data in tqdm(dataloader):
            names = data['name']
            images = data['image'].to(cfg.device)
            image_features = model.encode_image(images).cpu().numpy().tolist()
            for name, feat in zip(names, image_features):
                results[name] = feat
    if not os.path.exists(cfg.out_path):
        os.mkdir(cfg.out_path)
    with open(os.path.join(cfg.out_path, f"{cfg.clip_arch.replace('/', '_')}_clip_forward_ava_results.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)
