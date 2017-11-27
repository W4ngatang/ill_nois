#!/bin/bash
#
#SBATCH -t 2-00:00
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o src/logs/sbatch.out
#SBATCH -e src/logs/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

EXP_NAME=$1
EXP_DIR="/n/regal/rush_lab/awang/processed_data/imNet_correct"

DATE="$(date +%m-%d)"
LOG_PATH="/n/holylfs/LABS/rush_lab/users/wangalexc/logs/$DATE"
OUT_PATH="/n/holylfs/LABS/rush_lab/users/wangalexc/outs/$DATE/${EXP_NAME}"
mkdir -p $LOG_PATH
mkdir -p $OUT_PATH

MODEL=ensemble
ENSEMBLE_HOLDOUT=$4

TRAIN=0

GENERATE=1
GENERATOR=ensemble
N_GEN_STEPS=$2
TARGET=$5
TARGET_FILE=/n/home09/wangalexc/transferability-advdnn-pub/data/test_labels.txt

GEN_OPTIMIZER=adam
GEN_LR=$7
GEN_BATCH_SIZE=10
GEN_INIT_OPT_CONST=$6
N_BINARY_SEARCH_STEPS=$3
STOCHASTIC_GENERATE=0

USE_MWU=$8
N_MWU_STEPS=${9}
MWU_PENALTY=${10}

CMD="python -m src/codebase_pytorch/main --data_path $EXP_DIR --log_file $LOG_PATH/${EXP_NAME}.log --im_file $EXP_DIR/te.hdf5 --out_file $OUT_PATH/$EXP_NAME.hdf5 --out_path $OUT_PATH --train $TRAIN --model $MODEL --ensemble_holdout $ENSEMBLE_HOLDOUT --generate $GENERATE --generator $GENERATOR --n_generator_steps $N_GEN_STEPS --target $TARGET --generator_optimizer $GEN_OPTIMIZER --generator_lr $GEN_LR --generator_init_opt_const $GEN_INIT_OPT_CONST --n_binary_search_steps $N_BINARY_SEARCH_STEPS --generator_batch_size $GEN_BATCH_SIZE --n_mwu_steps $N_MWU_STEPS --mwu_penalty $MWU_PENALTY --mwu_ensemble_weights $USE_MWU --stochastic_generate $STOCHASTIC_GENERATE --target_file $TARGET_FILE"
eval $CMD
