# SGIEL_VIReID  (CVPR 2023)  
Official PyTorch Implementation of "Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification" (CVPR'23) 



### Datasets
We follow [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) to preprocess SYSU-MM01 dataset.

For VCM-HITSZ, please refer to [its official repository](https://github.com/VCM-project233/MITML).


### Body Shape Data

We borrowed pre-trained Self-Correction Human Parsing ([SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)) model (pretrained on Pascal-Person-Part dataset) to segment body shape from background. Given a pixel of a visible or infrared image, we directly summed the probabilities of being a part of the head, torso, or limbs, predicted by SCHP, to create the body-shape map.  


### Dependencies

* python 3.7.9
* pytorch >1.0 (>1.7 recommended)
* torchvision 0.8.2
* cudatoolkit 11.0

### Training and Model

To reproduce our results on SYSU-MM01, just run (after the dataset path declared)
```
bash run.sh
```

We also uploaded a trained [model](https://drive.google.com/file/d/19LtgJ3p1xtCD4zU-_3OG8hrvMh2vKr_m/view?usp=sharing) using SGIEL.

### Acknowledge  

Thanks for the great code base from the open-sourced [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline).
