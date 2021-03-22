What is it?
-----------

This toolbox includes nine algorithms for fusing hyperspectral and multispectral data to obtain high-resolution hyperspectral data. The toolbox was used for the comparisons in

N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and multispectral data fusion: a comparative review of the recent literature," IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 2, pp. 29-56, June 2017.


Algorithms
-----------

A. GSA (Gram-Schmidt adaptive)
Original code: http://www.openremotesensing.net/index.php/codes/11-pansharpening/6-hyperspectral-pansharpening-a-review
Note: The original code was adapted for hyperspectral and multispectral data fusion.

B. SFIM-HS (Smoothing filtered-based intensity modulation with hypersharpening)
Note: The code was implemented for the toolbox.

C. GLP-HS (Generalized Laplacian pyramid with hypersharpening)
Original code: http://www.openremotesensing.net/index.php/codes/11-pansharpening/6-hyperspectral-pansharpening-a-review
Note: The original code was adapted for hyperspectral and multispectral data fusion.

D. CNMF (Coupled nonnegative matrix factorization)
Original code: http://naotoyokoya.com/Download.html
Key parameters: M (the number of endmembers) in "CNMF_fusion.m".

E. ECCV'14 (Akhtar's method)
Original code: http://staffhome.ecm.uwa.edu.au/~00053650/code.html
Note: The original code "superResolution.m" was modified for the toolbox as "superResolution2.m". SPAMS (SPArse Modeling Software), which is an open source tool for sparse modeling, needs to be installed. SPAMS is available at http://spams-devel.gforge.inria.fr
Key parameters: param.L (Atoms selected in each iteration of G-SOMP+), param.k (Number of dictionary atoms), and param.eta (Modeling error)in "superResolution2.m".

F. ICCV'15 (Lanaras's method)
Original code: https://github.com/lanha/SupResPALM
Key parameters: p (the number of material spectra) in "SupResPALM.m".

G. HySure (Hyperspectral superresolution)
Original code: https://github.com/alfaiate/HySure
Note: The original code was used with a slight modification in the estimation of relative spectral response function.
Key parameters: p (the number of endmembers) in "HySure_wrapper.m".

H. MAP-SMM (Eismann's method)
Original code: Available in "M. T. Eismann, Resolution enhancement of hyperspectral imagery
using maximum a posteriori estimation with a stochastic mixing model, Ph.D. dissertation, Univ. Daton, Dayton, OH, May 2004."
Key parameters: npure (the number of endmembers) in "MAPSMM_fusion.m".

I. FUSE (Fast fusion based on Sylvester equation)
Original code: http://wei.perso.enseeiht.fr/publications.html
Key parameters: nb_sub (the subspace dimension) in "idHSsub.m".


Release date
------------

June 26, 2017


How to use it?
--------------

1) Set the MATLAB path to all folders in the toolbox
2) In ./Demo/HSMSFusionToolbox_Demo.m, please change "toolbox_path" to where you put the toolbox
3) In ./Demo/HSMSFusionToolbox_Demo.m, please change "method" to a method that you want to test
4) Run ./Demo/HSMSFusionToolbox_Demo.m

The toolbox includes four folders:

'Data': a sample data set of Hyperspec-VNIR Chikusei data.
'Demo': a demo program using Hyperspec-VNIR Chikusei data.
'Methods': source codes for nine algorithms of hyperspectral and multispectral data fusion.
'Quality_Indices': source codes for quality indices.


System-specific notes
---------------------

The code was tested using MATLAB R2015b on Windows 7 machines.
