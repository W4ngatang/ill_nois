#!/usr/bin/env bash
#SBATCH -t 1-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o sbatch.out
#SBATCH -e sbatch.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

EXP_NAME="mwuTest"
EXP_DIR="/n/regal/ysinger_group/jcperdomo/data/"
DATE="$(date +%m-%d)"
LOG_PATH="logs/$DATE"
OUT_PATH="outs/$DATE/${EXP_NAME}"
CKPT_PATH="checkpoints/$DATE"
mkdir -p $LOG_PATH
mkdir -p $OUT_PATH
mkdir -p $CKPT_PATH


MODEL=ensemble


WEIGHTS_DICT="/n/home00/jcperdomo/juankyWeightLocations.pickle"
ENSEMBLE_HOLDOUT='vgg19bn'

TRAIN=0

GENERATE=1
GENERATOR=optimization
N_GEN_STEPS=10
TARGET='none'

GEN_EPS=.1
GEN_ALPHA=0.0

GEN_INIT_OPT_CONST=1000.
GEN_OPTIMIZER='adam'
GEN_LR=.01
GEN_BATCH_SIZE=5
N_BINARY_SEARCH_STEPS=1

USE_MWU=0
N_MWU_STEPS=5
MWU_PENALTY=.1

CMD="python -m src/codebase_pytorch/main --data_path $EXP_DIR --log_file $LOG_PATH/${EXP_NAME}.log --im_file $EXP_DIR/te.hdf5 --out_file $OUT_PATH/$EXP_NAME.hdf5 --out_path $OUT_PATH --weights_dict $WEIGHTS_DICT  --generator $GENERATOR --alpha $GEN_ALPHA --eps $GEN_EPS --n_generator_steps $N_GEN_STEPS --target $TARGET --generator_optimizer $GEN_OPTIMIZER --generator_lr $GEN_LR --generator_init_opt_const $GEN_INIT_OPT_CONST --model $MODEL --generate $GENERATE --train $TRAIN --n_binary_search_steps $N_BINARY_SEARCH_STEPS --generator_batch_size $GEN_BATCH_SIZE --n_mwu_steps $N_MWU_STEPS --mwu_penalty $MWU_PENALTY --mwu_ensemble_weights $USE_MWU --ensemble_holdout $ENSEMBLE_HOLDOUT"
eval $CMD
