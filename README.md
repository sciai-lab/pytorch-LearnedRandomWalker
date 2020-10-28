# pytorch-LearnedRandomWalker
Implementation of the LearnedRandomWalker module as described in:
* Paper: [End-To-End Learned Random Walker for Seeded Image Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.pdf)  
* [Supplementary Material](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Cerrone_End-To-End_Learned_Random_CVPR_2019_supplemental.pdf)  
* [CVPR2019 Poster](./data/cvpr19_LRW_poster.pdf)

## Data processing:
The results reported in the paper are based on a modified version of the [CREMI](https://cremi.org/) challenge dataset.

The following processing have been performed:
* The `raw` and `labels` have been cropped to by 2 pixels in the `x/y` plane to to avoid potential
misalignments during the upsampling and downsampling.  
* The slice `0` of the `labels` is ignored because the UNet used in the experiments uses 3 z-slices as input. 
* Some instances are connected in 3D but are not visually connected in 2D, therefore a slice-by-slice relabeling is 
performed.
* Groundtruth segments smaller than `64` pixels in the `x/y` plane are merged with the surrounding segments using the 
watershed algorithm. 
* Groundtruth slices corrupted or with extreme artifacts are ignored in the testing and training. 
The following slices are removed from the test set: **Cremi B**: 44, 45, 15, 16, **Cremi C**: 14.
* The first 50 **valid** slices from each CREMI volume (A, B, C) are used for testing. The remaining **valid** slices 
are used for training.

The final 150 test set slices (with all above mentioned modifications):
https://heibox.uni-heidelberg.de/published/cvpr2019_lrw/

Additionally the repository contains seeds, learned RW segmentation, standard WS segmentation, standard RW segmentation
and the CNN predictions. 

## Evaluation:
In the [evaluation](./evaluation) directory you can find all instruction to reproduce the results in the manuscript
and the evaluation script used. 


## Cite:
```
@inproceedings{cerrone2019,
  title={End-to-end learned random walker for seeded image segmentation},
  author={Cerrone, Lorenzo and Zeilmann, Alexander and Hamprecht, Fred A},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12559--12568},
  year={2019}
}
```

