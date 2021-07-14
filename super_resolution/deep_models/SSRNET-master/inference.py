import torch
import numpy as np
from torch import nn
from osgeo import gdal
from augsburg_data_loader import build_datasets
from models.SSRNET import SSRNET
from models.TFNet import ResTFNet


class arguments():
    def __init__(self):
        self.model = 'ResTFNet_2021-03-24_16-44-09'
        print('Model used for prediction: ', self.model)
        self.arch  = self.model.split('_')[0]
        print('Model architecture type: ', self.arch)
        self.model_path = '../trained_model/'+self.model
        print('Directory to the model: ', self.model_path)


        self.ref_tiff_directory = '../../data/sub_area_1/EeteS_EnMAP_10m_sub_area1.tif'
        self.out_tiff_directory = '../../data/sub_area_1/'+self.model+'_fusion.tif'

        
        self.scale_ratio = 3
        self.n_bands_hyper = 242
        self.n_bands_multi = 4



def predict(test_list, arch, model, cuda_dev):
    _, test_lr, test_hr = test_list
    model.eval()
    with torch.no_grad():
        # Set mini-batch dataset
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

        out = out.detach().cpu().numpy()

    return out


def saveResultAsGeotiff(result, ref_tiff, out_tiff):
    # this function save the fused image as a geotiff that has the same geographic information as ref file
    try:
        # load geo parameters from ref geotiff    
        fid = gdal.Open(ref_tiff)
        row = fid.RasterYSize
        col = fid.RasterXSize
        bnd = fid.RasterCount
        proj = fid.GetProjection()
        geoInfo = fid.GetGeoTransform()
        del(fid)

        # intial output geotiff file and geo parameters
        out_driver = gdal.GetDriverByName('GTiff')
        out_f = out_driver.Create(out_tiff, col, row, bnd, gdal.GDT_UInt16)
    
        out_f.SetProjection(proj)
        out_f.SetGeoTransform(geoInfo)

        for idx_bnd in range(bnd):
            outBand = out_f.GetRasterBand(idx_bnd+1)
            outBand.WriteArray(result[idx_bnd,:,:])
            outBand.FlushCache()
        print('Prediction successfully saved as geotiff file')
        return 0
    except:
        print('Error during saving prediction')
        return 1




def main():
    # set argument
    args = arguments()
    # load data set
    _, test_list, _ = build_datasets()

    # gpu
    cuda_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    # load weights of trained model
    model.load_state_dict(torch.load(args.model_path), strict=False)
    # inferencing
    prediction = predict(test_list, args.arch, model, cuda_dev)

    # save output as a geotiff
    prediction = np.squeeze(prediction,axis=0)
    prediction = (prediction*1e4).astype(np.int)
    saveResultAsGeotiff(prediction, args.ref_tiff_directory, args.out_tiff_directory)




if __name__ =='__main__':
    main()


