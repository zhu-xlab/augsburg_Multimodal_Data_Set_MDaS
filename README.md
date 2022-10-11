# MDAS: A New Multimodal Benchmark Dataset for Remote Sensing

In Earth observation, multimodal data fusion is an intuitive strategy to break the limitation of individual data. Complementary physical contents of data sources allow comprehensive and precise information retrieve. With current satellite missions, such as ESA Copernicus programme, various data will be accessible at an affordable cost. Future applications will have many options on data sources. Such privilege can be beneficial only if algorithms are ready to work with various data sources. However, current data fusion studies mostly focus on the fusion of two data sources. There are two reasons, first, different combinations of data sources face different scientific challenges. For example, the fusion of synthetic aperture radar (SAR) data and optical images needs to handle the geometric difference, while the fusion of hyperspectral and multispectral images deals with different resolutions on spatial and spectral domains. Second, nowadays, it is still both financially and labour expensive to acquire multiple data sources for the same region at the same time. In this paper, we provide the community a benchmark multimodal data set, MDAS, for the city of Augsburg, Germany. MDAS includes synthetic aperture radar (SAR) data, multispectral image, hyperspectral image, digital surface model (DSM), and geographic information system (GIS) data. All these data are collected on the same date, 7th May 2018. MDAS is a new benchmark data set that provides researchers rich options on data selections. In this paper, we run experiments for three typical remote sensing applications, namely, resolution enhancement, spectral unmixing, and land cover classification, on MDAS data set. Our experiments demonstrate the performance of representative state-of-the-art algorithms whose outcomes can sever as baselines for further studies.

The data is publicly available at [10.14459/2022mp1657312](https://doi.org/10.14459/2022mp1657312). If you use this data set, please cite our [paper](https://essd.copernicus.org/preprints/essd-2022-155/essd-2022-155.pdf).

```
@article{hu2022mdas,
  title={MDAS: A New Multimodal Benchmark Dataset for Remote Sensing},
  author={Hu, Jingliang and Liu, Rong and Hong, Danfeng and Camero, Andr{\'e}s and Yao, Jing and Schneider, Mathias and Kurz, Franz and Segl, Karl and Zhu, Xiao Xiang},
  journal={Earth System Science Data Discussions},
  pages={1--26},
  year={2022},
  publisher={Copernicus GmbH},
  doi={10.5194/essd-2022-155}
}
```

The code of the super-resolution experiments can be found [here](https://github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS/tree/main/super_resolution). Please, follow the links to get the original implementations used for the unmixing, namely [NMF-QMV](https://github.com/LinaZhuang/NMF-QMV_demo), [SeCoDe](https://github.com/danfenghong/IEEE_TGRS_SeCoDe), and [GMM](https://github.com/zhouyuanzxcv/Hyperspectral).



