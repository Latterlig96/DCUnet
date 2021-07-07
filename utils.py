import torch
import numpy as np

def sum_tensor(inp: torch.Tensor, 
               axes: torch.Tensor, 
               keepdim: bool=False):
        # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
        axes = np.unique(axes).astype(int)
        if keepdim:
            for ax in axes:
                inp = inp.sum(int(ax), keepdim=True)
        else:
            for ax in sorted(axes, reverse=True):
                inp = inp.sum(int(ax))
        return inp

def get_tp_fp_fn(net_output: torch.Tensor, 
                 gt: torch.Tensor, 
                 axes: bool=None, 
                 mask: bool=None, 
                 square: bool=False):
        """
        net_output must be (b, c, x, y(, z)))
        gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
        if mask is provided it must have shape (b, 1, x, y(, z)))
        :param net_output:
        :param gt:
        :param axes:
        :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
        :param square: if True then fp, tp and fn will be squared before summation
        :return:
        """
        if axes is None:
            axes = tuple(range(2, len(net_output.size())))

        shp_x = net_output.shape
        shp_y = gt.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        tp = net_output * y_onehot
        fp = net_output * (1 - y_onehot)
        fn = (1 - net_output) * y_onehot

        if mask is not None:
            tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
            fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
            fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

        if square:
            tp = tp ** 2
            fp = fp ** 2
            fn = fn ** 2

        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)

        return tp, fp, fn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class AverageMeter:
    def __init__(self,
                 name: str, 
                 fmt: str =':f'):
        self.name = name
        self.fmt = fmt
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
    
    def get_avg_loss(self) -> float:
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
