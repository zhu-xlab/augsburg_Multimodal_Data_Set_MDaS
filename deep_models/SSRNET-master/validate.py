import torch
from torch import nn
from utils import *
import pdb
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def validate(test_list, arch, model, epoch, n_epochs, cuda_dev):
    test_ref, test_lr, test_hr = test_list
    model.eval()
    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = test_ref
        lr = test_lr.to(cuda_dev,dtype=torch.float)
        hr = test_hr.to(cuda_dev,dtype=torch.float)
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)

        ref = ref.numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)

        with open('ConSSFCNN.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')

    return psnr
