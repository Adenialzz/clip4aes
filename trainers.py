import torch
import sys
import os
import os.path as osp
import logging
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from utils import AverageMeter, get_score, emd_loss, accuracy, calc_aesthetic_metrics
from sklearn import metrics
from tensorboardX import SummaryWriter
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class BaseTrainer:
    def __init__(self, cfg, model, dataset_list, metrics_list):

        self.cfg = cfg
        if not osp.exists(self.cfg.out_path):
            os.mkdir(self.cfg.out_path)
        self.cfg.summary_path = osp.join(self.cfg.out_path, 'summaries')
        self.cfg.model_path = osp.join(self.cfg.out_path, 'weights')
        self.init_writer()
        self.init_logger()
        self.init_device()
        self.init_loss_func()
        self.init_model(model)
        self.init_optimizer()
        self.init_lr_scheduler()

        train_set, val_set = dataset_list
        self.init_dataloader(train_set, val_set)

        self.metrics_list = metrics_list

    def init_logger(self):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file_handler = logging.FileHandler(osp.join(self.cfg.summary_path, 'training.log'), 'a', encoding='utf-8')
        std_out_handler = logging.StreamHandler(sys.stdout)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(eval(f"logging.{self.cfg.log_level}"))

        log_file_handler.setFormatter(formatter)
        std_out_handler.setFormatter(formatter)

        self.logger.addHandler(log_file_handler)
        self.logger.addHandler(std_out_handler)

    def init_writer(self):
        self.writer = SummaryWriter(self.cfg.summary_path)

    def init_device(self):
        self.device = torch.device(self.cfg.device)

    def init_model(self, model):
        self.model = model.to(self.device)

    def init_dataloader(self, train_set, val_set):
        self.train_loader = DataLoader(
                train_set, 
                batch_size=self.cfg.batchSize, 
                num_workers=self.cfg.num_workers, 
                pin_memory=True if self.cfg.num_workers > 0 else False,
                shuffle=True
                )
        self.val_loader = DataLoader(
                val_set, 
                batch_size=self.cfg.batchSize, 
                num_workers=self.cfg.num_workers, 
                pin_memory=True if self.cfg.num_workers > 0 else False,
                shuffle=False)
    
    def print_configs(self):
        print('*'*21, " - Configs - ", '*'*21)
        for k, v in vars(self.cfg).items():
            if v is None:
                v = "None"
            print(f"{k}: {v}")
        print('*'*56)

    def write_configs(self):
        save_path = osp.join(self.cfg.summary_path, 'configs.yml')
        with open(save_path, 'w') as f:
            for k, v in vars(self.cfg).items():
                f.writelines(f"{k}: {v}\n")
    
    def write_configs_to_txt(self):
        with open(self.cfg.summary_path, 'w') as f:
            f.writelines('*'*21, " - Configs - ", '*'*21+'\n')
            for k, v in vars(self.cfg).items():
                if v is None:
                    v = "None"
                f.writelines(k, ':', v+'\n')
            f.writelines('*'*56 + '\n')

    def init_lr_scheduler(self):
        if self.cfg.lr_scheduler_type is None:
            self.lr_scheduler = None
        elif self.cfg.lr_scheduler_type == "lambdalr":
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch+1))

    def init_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.cfg.optimizer_type == "sgd":
            self.optimizer = SGD(parameters, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_type == "adam":
            self.optimizer = Adam(parameters, lr=self.cfg.lr)
    
    def init_loss_func(self): 
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def resume(self):
        ckpt = torch.load(self.cfg.resume, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["epoch"]
    
    def epoch_forward(self, isTrain, epoch):
        raise NotImplementedError

    def plot_epoch_metric(self, epoch, train_dict, val_dict):
        for metric in self.metrics_list:
            self.writer.add_scalars(metric, {"train " + metric: train_dict[metric], "val " + metric: val_dict[metric]}, epoch)

    def save_model(self, epoch):
        if not osp.isdir(self.cfg.model_path):
            os.mkdir(self.cfg.model_path)
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, osp.join(self.cfg.model_path, f"model_{epoch}.pth"))
    
    def update_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def forward(self):
        self.print_configs()
        self.write_configs()
        if self.cfg.resume is not None:
            start_epoch = self.resume() + 1
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.cfg.epochs):
            self.logger.info(f"Training Epoch = {epoch}")
            train_metrics_dict = self.epoch_forward(isTrain=True, epoch=epoch)
            with torch.no_grad():
                self.logger.info(f"Validating Epoch = {epoch}")
                val_metrics_dict = self.epoch_forward(isTrain=False, epoch=epoch)
            self.plot_epoch_metric(epoch, train_metrics_dict, val_metrics_dict)
            self.save_model(epoch)
            self.update_lr()


class AesTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super().__init__(cfg, model, dataset_list, metrics_list)
        self.softmax = torch.nn.Softmax(dim=1)

    def init_loss_func(self):
        self.loss_func = emd_loss
    
    def epoch_forward(self, isTrain, epoch):
        if isTrain:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
        
        _loss = AverageMeter()
        _acc = AverageMeter()
        _plcc_mean = AverageMeter()
        _srcc_mean = AverageMeter()
        _plcc_std = AverageMeter()
        _srcc_std = AverageMeter()
        for batch_idx, data in enumerate(loader):
            image = data["image"].to(self.device)
            label = data["annotations"].to(self.device)
            bin_label = data["bin_cls"]

            if isTrain:
                self.optimizer.zero_grad()
            output = self.model(image)
            output = self.softmax(output)

            loss = self.loss_func(output, label)
            if isTrain:
                loss.backward()
                self.optimizer.step()

            acc, plcc_mean, srcc_mean, plcc_std, srcc_std = calc_aesthetic_metrics(output, label, bin_label)

            _loss.update(loss.item())
            _plcc_mean.update(plcc_mean)
            _srcc_mean.update(srcc_mean)
            _plcc_std.update(plcc_std)
            _srcc_std.update(srcc_std)
            _acc.update(acc)

            if  batch_idx % self.cfg.log_freq == 0:
                self.logger.info("epoch: {}, step: {} | loss: {:.4f}, acc: {:.2f}, cc: {:.4f}, {:.4f}, {:.4f}, {:.4f}"
                .format(epoch, batch_idx, _loss.avg, _acc.avg, _plcc_mean.avg, _srcc_mean.avg, _plcc_std.avg, _srcc_std.avg))

            # break
        metrics_result = {}
        for metric in self.metrics_list:
            metrics_result[metric] = eval('_' + metric).avg

        return metrics_result

