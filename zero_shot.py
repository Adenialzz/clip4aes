import torch
import clip
from tqdm import tqdm
import argparse

from utils import AverageMeter, emd_loss, calc_aesthetic_metrics
from dataset import CustomImageFolder, AVADataset
from sklearn import metrics

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
    prompts = [ f'An image with an aesthetic score of {score} on a scale of 1-10.' for score in range(1, 11) ]
    # prompts = [ 'An aesthetically negative picture.', 'An aesthetically positive image.' ]
    assert len(prompts) == 2 or len(prompts) == 10  # for aesthetic binary classification and aesthetic distribution prediction
    text_inputs = clip.tokenize(prompts).to(cfg.device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # metric loggers 
    _acc = AverageMeter()
    if len(prompts) == 10:
        _plcc_mean = AverageMeter()
        _srcc_mean = AverageMeter()
        _plcc_std = AverageMeter()
        _srcc_std = AverageMeter()
        _loss = AverageMeter()


    with torch.no_grad():
        for data in tqdm(dataloader):
            image = data["image"].to(cfg.device)
            label = data["annotations"].to(cfg.device)
            bin_label = data["bin_cls"]

            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            output = (image_features @ text_features.T).softmax(dim=-1)

            if len(prompts) == 10: 
                acc, plcc_mean, srcc_mean, plcc_std, srcc_std = calc_aesthetic_metrics(output, label, bin_label)
                loss = emd_loss(output, label)

                _loss.update(loss.item())
                _plcc_mean.update(plcc_mean)
                _srcc_mean.update(srcc_mean)
                _plcc_std.update(plcc_std)
                _srcc_std.update(srcc_std)
            else:
                emd_class_pred = torch.zeros((image.shape[0]))
                for idx in range(image.shape[0]):
                    if output[idx][0] > 0.5: # negative
                        emd_class_pred[idx] = 0.0
                    else:
                        emd_class_pred[idx] = 1.0
                acc = metrics.accuracy_score(bin_label, emd_class_pred)
            _acc.update(acc)

        if len(prompts) == 10:
            print("emd_loss: {:.4f}, acc: {:.2f}, cc: {:.4f}, {:.4f}, {:.4f}, {:.4f}"
            .format( _loss.avg, _acc.avg, _plcc_mean.avg, _srcc_mean.avg, _plcc_std.avg, _srcc_std.avg))
        else:
            print(f"acc: {_acc.avg:.2f}")


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)
