#!/bin/bash
#
#SBATCH -t 1-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=35000
#SBATCH -o logs/sbatch.out
#SBATCH -e logs/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

SCRATCH=/n/regal/rush_lab/awang/
#python codebase/utils/process_lfw.py --data_path data/raw/lfw/lfw-deepfunneled --save_data_to data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64
#python codebase/utils/process_lfw.py --load_data_from data/raw/lfw/lfw-deepfunneled.hdf5 --out_path data/lfw_test --resize 64

#python src/codebase_pytorch/utils/process_facescrub.py --data_path /n/regal/rush_lab/awang/facescrub/raw/ --out_path /n/regal/rush_lab/awang/data/facescrub_full/ --save_data_to /n/regal/rush_lab/awang/facescrub/dim224_n50.hdf5 --n_classes 50 --im_size 224 --n_channels 3 --normalize 1
#python src/codebase_pytorch/utils/quick_process.py --data_path /n/regal/rush_lab/awang/raw_data/imagenet/images_2/ --out_path /n/regal/rush_lab/awang/processed_data/imagenet_test/ --n_class_ims 2
#python src/codebase_pytorch/utils/song_data_process.py --data_list /n/home09/wangalexc/transferability-advdnn-pub/data/ilsvrc12/val.txt --data_path /n/home09/wangalexc/transferability-advdnn-pub/data/test_data --out_path /n/regal/rush_lab/awang/processed_data/song_small/ --n_ims 10

python src/codebase_pytorch/utils/ilsvrc_data_process.py --targ_file ${SCRATCH}/raw_data/imagenet/ilsvrc2012_val/ILSVRC2012_validation_ground_truth.txt --dict_file ${SCRATCH}/raw_data/imagenet/old2new.dict.pkl --data_path ${SCRATCH}/raw_data/imagenet/ilsvrc2012_val/ --out_path /n/regal/rush_lab/awang/processed_data/imNet_correct/tr.hdf5 --save_data_to ${SCRATCH}/raw_data/imagenet/ilsvrc2012_val/read_images.hdf5 --n_ims 1000
