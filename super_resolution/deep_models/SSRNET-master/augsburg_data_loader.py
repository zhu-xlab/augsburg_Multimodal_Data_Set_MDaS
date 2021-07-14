from osgeo import gdal
import torch

def tif_2_torch(tif_data_dir):
    # load data
    f = gdal.Open(tif_data_dir)
    dat = f.ReadAsArray()
    # scaling
    dat = dat/1e4
    # convert to torch
    dat = torch.from_numpy(dat).unsqueeze(dim=0)
    return dat 


def build_datasets():
    data_dir = '../../../Augsburg_data_4_publication/'
    train_ref = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_EnMAP_10m_deep_train.tif')
    train_lr  = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_EnMAP_30m_deep_train.tif')
    train_hr  = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_Sentinel_2_10m_deep_train.tif')

    test_ref  = tif_2_torch(data_dir+'sub_area_1/EeteS_EnMAP_10m_sub_area1.tif')
    test_lr   = tif_2_torch(data_dir+'sub_area_1/EeteS_EnMAP_30m_sub_area1.tif')
    test_hr   = tif_2_torch(data_dir+'sub_area_1/EeteS_Sentinel_2_10m_sub_area1.tif')

    valid_ref = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif')
    valid_lr  = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif')
    valid_hr  = tif_2_torch(data_dir+'sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif')

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr], [valid_ref, valid_lr, valid_hr]
