#!/bin/bash
#
#SBATCH -t 1-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o src/logs/sbatch.out
#SBATCH -e src/logs/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

EXP_NAME="imagenet"
EXP_DIR="/n/regal/rush_lab/awang/processed_data/imagenet_test"
DATE="$(date +%m-%d)"
LOG_PATH="src/logs/$DATE"
OUT_PATH="src/outs/$DATE/${EXP_NAME}"
CKPT_PATH="src/checkpoints/$DATE"
mkdir -p $LOG_PATH
mkdir -p $OUT_PATH
mkdir -p $CKPT_PATH

TRAIN_NEW=${1:-"0"}
MODEL_ACTION="load_model_from"
#MODEL_PATH="src/checkpoints/07-27/$EXP_NAME.ckpt"
#MODEL_PATH="/n/regal/rush_lab/awang/models/resnet152.ckpt"

MODEL=ensemble

N_MODULES=7
N_KERNELS=128
INIT_SCALE=.1

TRAIN=0
OPTIMIZER=adagrad
N_EPOCHS=0
LR=.1
MOMENTUM=.9
WEIGHT_DECAY=.0002
NESTEROV=true
BATCH_SIZE=50

GENERATE=1
GENERATOR=optimization
N_GEN_STEPS=100
TARGET='next'

GEN_EPS=.1
GEN_ALPHA=0.0

GEN_INIT_OPT_CONST=100
GEN_OPTIMIZER=adam
GEN_LR=.01
GEN_BATCH_SIZE=10
N_BINARY_SEARCH_STEPS=1

USE_MWU=0
N_MWU_STEPS=10
MWU_PENALTY=.1

if [ ! -f "$MODEL_PATH" ] || [ $TRAIN_NEW -eq "1" ]; then
    # Train a good model and save it
    echo "Training a model from scratch"
    MODEL_ACTION="save_model_to"
    MODEL_PATH="$CKPT_PATH/$EXP_NAME.ckpt"
else
    # Load a saved model
    echo "Loading a model"
    TRAIN=0
fi
CMD="python -m src/codebase_pytorch/main --data_path $EXP_DIR --log_file $LOG_PATH/${EXP_NAME}.log --im_file $EXP_DIR/te.hdf5 --out_file $OUT_PATH/$EXP_NAME.hdf5 --out_path $OUT_PATH --$MODEL_ACTION $MODEL_PATH --optimizer $OPTIMIZER --n_epochs $N_EPOCHS --init_scale .1 --n_kerns $N_KERNELS --lr $LR --n_modules $N_MODULES --batch_size $BATCH_SIZE --generator $GENERATOR --alpha $GEN_ALPHA --eps $GEN_EPS --n_generator_steps $N_GEN_STEPS --target $TARGET --generator_optimizer $GEN_OPTIMIZER --generator_lr $GEN_LR --generator_init_opt_const $GEN_INIT_OPT_CONST --model $MODEL --generate $GENERATE --train $TRAIN --n_binary_search_steps $N_BINARY_SEARCH_STEPS --generator_batch_size $GEN_BATCH_SIZE --n_mwu_steps $N_MWU_STEPS --mwu_penalty $MWU_PENALTY --mwu_ensemble_weights $USE_MWU"
eval $CMD
