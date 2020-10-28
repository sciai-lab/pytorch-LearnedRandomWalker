# Evaluation script

## Download data
In order to reproduce the results reported in 
[End-To-End Learned Random Walker for Seeded Image Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.pdf)
please download the `cvpr_2109_lrw_final.h5` `baseline_lrw.h5` from:
https://heibox.uni-heidelberg.de/published/cvpr2019_lrw/

## installation
```
conda env create -f ./environment.yml 
conda activate py36_cremi
pip install git+https://github.com/lorenzocerrone/cremi_python.git
```

## Run Learned RW evaluation
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./cvpr_2109_lrw_final.h5 
```

## Run baseline evaluation
standard RandomWalker algorithm
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./baseline_lrw.h5 --segdataset segmentation_stRW
```
standard Watershed algorithm
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./baseline_lrw.h5 --segdataset segmentation_stWS
```
 

