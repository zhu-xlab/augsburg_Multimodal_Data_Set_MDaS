import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from augsburg_data_loader import build_datasets
from validate import validate
from train import train
import pdb
from torch.nn import functional as F
from datetime import datetime

class arguments():
    def __init__(self):
        self.n_bands = 242
        self.arch = 'ResTFNet'
        self.scale_ratio = 3
        self.n_bands_hyper = 242
        self.n_bands_multi = 4
        self.lr = 1e-4
        self.model_path = '../trained_model/arch'
        self.n_epochs = 10000
        self.image_size = {}
        self.image_size['height'] = 100
        self.image_size['width'] = 120
        self.triger_time = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        print([self.image_size['height'],self.image_size['width']])


cuda_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(cuda_dev)

def main():
    args = arguments()
    # Custom dataloader
    train_list, test_list, valid_list = build_datasets()

    # Build the models
    if args.arch == 'ResTFNet':
        model = ResTFNet(args.scale_ratio, 
                       args.n_bands_multi, 
                       args.n_bands_hyper).to(cuda_dev)
    elif args.arch == 'SSRNET':
        model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_bands_multi, 
                     args.n_bands_hyper).to(cuda_dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load the trained model parameters
    model_path = args.model_path.replace('arch', args.arch+args.triger_time) 

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(valid_list, 
                                args.arch,
                                model,
                                0,
                                args.n_epochs,
                                cuda_dev)
        print ('psnr: ', recent_psnr)

    # Loss and Optimizer
    criterion = nn.MSELoss().cuda()


    best_psnr = 0
    best_psnr = validate(valid_list,
                          args.arch, 
                          model,
                          0,
                          args.n_epochs,
                          cuda_dev)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.n_epochs):
        # One epoch's traininginceptionv3
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list, 
              args.image_size,
              args.scale_ratio,
              args.arch,
              model, 
              optimizer, 
              criterion, 
              epoch, 
              args.n_epochs)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(valid_list, 
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs,
                                cuda_dev)
        print ('psnr: ', recent_psnr)

        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
            torch.save(model.state_dict(), model_path)
            print ('Saved!')
            print ('')

    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
