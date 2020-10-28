# Evaluation script

## Download data

## installation
```
conda env create -f ./environment.yml 
conda activate py36_cremi
pip install git+https://github.com/lorenzocerrone/cremi_python.git
```

## Learned RW evaluation
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./cvpr_2109_lrw_final.h5 
```
## Baseline evaluation
standard RandomWalker algorithm
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./baseline_lrw.h5 --segdataset segmentation_stRW
```
standard Watershed algorithm
```
python evaluation_test_cremi.py --gtpath ./cvpr_2109_lrw_final.h5 --segpath ./baseline_lrw.h5 --segdataset segmentation_stWS
```
 

