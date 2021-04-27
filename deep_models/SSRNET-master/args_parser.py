import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='SSRNET',
                            choices=[# these four models are used for ablation experiments
                                     'SSRNET', 
                                     # these five models are used for comparison experiments
                                     'ResTFNet' 
                                     ])

    parser.add_argument('-root', type=str, default='./data')    
    parser.add_argument('--scale_ratio', type=float, default=3)
    parser.add_argument('--n_bands_hyper', type=int, default=242)
    parser.add_argument('--n_bands_multi', type=int, default=4)

    parser.add_argument('--model_path', type=str, 
                            default='../trained_model/arch.pkl',
                            help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',  
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',  
                            help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
 
    args = parser.parse_args()
    return args
 
