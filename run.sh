#!/bin/bash

python traindistill.py --dataset sysu --lr 0.1 --method step2060_p0.2intercutmix_bothcegkl_distillcosboth --gpu 0 --date 8.16 --gradclip 11 --seed 3 --gpuversion 3090

