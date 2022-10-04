import torchvision.transforms as transforms
from dataset import AVADataset
from trainers import AesTrainer
from models import CLIP4AesFormer
from utils import freeze_weights
import timm
import clip
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # training procedure
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--resume", type=str, default=None)

    # optimizing procedure
    parser.add_argument("--optimizer-type", type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument("--fine-tune", action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_type", type=str, default=None, choices=["none", "lambdalr"])
    parser.add_argument("--weight-decay", type=float, default=0.)

    # save logs and results
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--log-level", type=str, choices=['INFO', 'DEBUG'], default='DEBUG')
    parser.add_argument("--out-path", type=str, default="./out_path")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--arch", type=str, default='resnet50', choices=timm.list_models(pretrained=True)+clip.available_models())
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    if cfg.arch in clip.available_models():
        model = CLIP4AesFormer(cfg.arch, cfg.device, out_len=10)
        if not cfg.fine_tune:
            freeze_weights(model, ['aes_fc'], None)
        pipeline = transforms.Compose([transforms.RandomHorizontalFlip(), model.preprocess])
    else:
        model = timm.create_model(cfg.arch, pretrained=True, num_classes=10)
        if not cfg.fine_tune:
            freeze_weights(model, ['head'], 2)
        pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    train_set = AVADataset(csv_file="/home/ps/JJ_Projects/FromSong/dsmAVA/csvFiles/train_mlsp.csv", root_dir="/ssd1t/song/Datasets/AVA/shortEdge256/all_images", transform=pipeline)
    val_set = AVADataset(csv_file="/home/ps/JJ_Projects/FromSong/dsmAVA/csvFiles/val_mlsp.csv", root_dir="/ssd1t/song/Datasets/AVA/shortEdge256/all_images", transform=pipeline)

    metrics_list = ['loss', 'acc', 'plcc_mean', 'srcc_mean', 'plcc_std', 'srcc_std']
    trainer = AesTrainer(cfg, model, [train_set, val_set], metrics_list)
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main(cfg)
