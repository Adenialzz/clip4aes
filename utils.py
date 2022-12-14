import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import metrics

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Num of Total Parameters': total_num, 'Num of Trainable Parameters': trainable_num}

def freeze_weights(model, remain_keys, n_remain_last_layer=None):
    if n_remain_last_layer is not None:
        loop = list(model.named_parameters())[: - n_remain_last_layer]
    else:
        loop = model.named_parameters()
    for name, params in loop:
        if_remain = False
        for k in remain_keys:
            if k in name:
                if_remain = True
                break
        params.requires_grad = if_remain

def get_bin_result(score_np):
    bs = score_np.shape[0]
    bin_result = torch.zeros((bs))
    for idx in range(bs):
        if score_np[idx] < 5:
            bin_result[idx] = 0.0
        elif score_np[idx] >= 5:
            bin_result[idx] = 1.0
    return bin_result

def calc_aesthetic_metrics(output, label):
    # calculate the cc of mean score
    pscore_np = get_score(output).cpu().detach().numpy()
    tscore_np = get_score(label).cpu().detach().numpy()

    plcc_mean = pearsonr(pscore_np, tscore_np)[0]
    srcc_mean = spearmanr(pscore_np, tscore_np)[0]

    # calculate the cc of std.dev
    pstd_np = torch.std(output, dim=1).cpu().detach().numpy()
    tstd_np = torch.std(label, dim=1).cpu().detach().numpy()

    plcc_std = pearsonr(pstd_np, tstd_np)[0]
    srcc_std = spearmanr(pstd_np, tstd_np)[0]

    # calculate the classification result of emd
    bin_label = get_bin_result(tscore_np)
    bin_pred = get_bin_result(pscore_np)

    acc = metrics.accuracy_score(bin_label, bin_pred)
    return acc, plcc_mean, srcc_mean, plcc_std, srcc_std

class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):       
    # params: output.shape (bs, num_classes), target.shape (bs, )
    # returns: res: list
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return [acc.item() for acc in res]

def get_score(score_distri):
    '''
    score_distri shape:  batch_size * 10
    '''
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor).to(score_distri.device)
    w_batch = w.repeat(score_distri.size(0), 1)

    score = (score_distri * w_batch).sum(dim=1)

    return score

def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes ???~W 1
        q: estimated distribution of shape num_classes ???~W 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size ???~W num_classes ???~W 1
        q: estimated distribution of shape mini_batch_size ???~W num_classes ???~W 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size
