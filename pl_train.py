import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os.path as osp
import argparse
import clip
from lavis.models import load_model_and_preprocess
import timm

from utils import emd_loss, calc_aesthetic_metrics
from dataset import AVADataset

def get_args():
    parser = argparse.ArgumentParser()
    # training procedure
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume", type=str, default=None)

    # optimizing procedure
    parser.add_argument("--optimizer-type", type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_type", type=str, default=None, choices=["none", "lambdalr"])
    parser.add_argument("--weight-decay", type=float, default=0.)

    # model arch
    parser.add_argument("--fine-tune", action='store_true')
    parser.add_argument("--fc-bias", action='store_true')

    # save logs and results
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--log-level", type=str, choices=['INFO', 'DEBUG'], default='INFO')
    parser.add_argument("--data-root", type=str, default="/ssd1t/song/Datasets/AVA/")
    parser.add_argument("--out-dir", type=str, default=None)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--arch", type=str, default='ViT-B/32', choices=timm.list_models(pretrained=True)+clip.available_models()+['blip'])
    cfg = parser.parse_args()
    return cfg


class AesModel(pl.LightningModule):
    def __init__(self, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.pipeline = self.get_backbone()
        self.fc = torch.nn.Linear(feat_dim, 10, bias=self.cfg.fc_bias)
        self.softmax = torch.nn.Softmax(dim=1)
        if not self.cfg.fine_tune:
            self.freeze_backbone()
        self.save_hyperparameters()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_backbone(self):
        if self.cfg.arch in clip.available_models():
            clip_model, preprocessor = clip.load(self.cfg.arch, self.cfg.device)
            self.backbone_data_type = clip_model.dtype
            backbone = clip_model.visual
            pipeline = transforms.Compose(([transforms.RandomHorizontalFlip(), preprocessor]))
        elif self.cfg.arch == 'blip':
            backbone, preprocessors, _ = load_model_and_preprocess('blip_feature_extractor', model_type='base', is_eval=True, device=self.cfg.device)
            pipeline = transforms.Compose([transforms.RandomHorizontalFlip(), preprocessors['eval']])
        else:
            backbone = timm.create_model(self.cfg.arch, pretrained=True, num_classes=10)
            pipeline = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        return backbone, pipeline

    def configure_optimizers(self):
        params = []
        if not self.cfg.fine_tune:
            params.append( { 'params': self.backbone.parameters() } )
        params.append( { 'params': self.fc.parameters() } )

        if self.cfg.optimizer_type == 'adam':
            return torch.optim.Adam(params, lr=self.cfg.lr)
        elif self.cfg.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)

    def forward(self, x):
        x = x.type(self.backbone_data_type)
        feat = self.backbone(x)
        feat = feat.type(torch.float32)
        out = self.fc(feat)
        return out

    def make_logs(self, acc, plcc_mean, srcc_mean, plcc_std, srcc_std, loss, prefix):
        self.log(f'{prefix}_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_plcc_mean', plcc_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_srcc_mean', srcc_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_plcc_std', plcc_std, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_srcc_std', srcc_std, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_loss', loss, sync_dist=True)

    def _step(self, batch, batch_idx, split):
        image = batch['image']
        label = batch['annotations']
        outputs = self.softmax(self(image))
        loss = emd_loss(outputs, label)
        acc, plcc_mean, srcc_mean, plcc_std, srcc_std = calc_aesthetic_metrics(outputs, label)
        self.make_logs(acc, plcc_mean, srcc_mean, plcc_std, srcc_std, loss.item(), prefix=split)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, 'validation')
        return loss

    def train_dataloader(self):
        dset = AVADataset(csv_file=osp.join(self.cfg.data_root, 'csvFiles/trainval_mlsp.csv'), root_dir=osp.join(self.cfg.data_root, 'shortEdge256/all_images/'), transform=self.pipeline)
        loader = DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=10)
        return loader

    def val_dataloader(self):
        dset = AVADataset(csv_file=osp.join(self.cfg.data_root, 'csvFiles/test_mlsp.csv'), root_dir=osp.join(self.cfg.data_root, 'shortEdge256/all_images/'), transform=self.pipeline)
        loader = DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10)
        return loader

    def test_dataloader(self):
        dset = AVADataset(csv_file=osp.join(self.cfg.data_root, 'csvFiles/test_mlsp.csv'), root_dir=osp.join(self.cfg.data_root, 'shortEdge256/all_images/'), transform=self.pipeline)
        loader = DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10)
        return loader

def main(cfg):
    model = AesModel(512, cfg)
    trainer = pl.Trainer(
        default_root_dir=cfg.out_dir,
        accelerator='auto',
        devices='auto',
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        max_epochs=50
    )
    trainer.fit(model)

if __name__ == '__main__':
    cfg = get_args()
    main(cfg)




