#!/bin/bash

RUN_LOCAL=${1:-"1"}

#python -m demo/main --data_path src/data/cifar10 --load_model_from src/checkpoints/06-26/cifar10.ckpt --n_modules 5 --n_kerns 128 --eps .1
python -m demo/main --data_path src/data/facescrub_d128 --load_model_from src/checkpoints/07-10/facescrub_d128.ckpt --n_modules 7 --n_kerns 128 --eps .1 --run_local $RUN_LOCAL
