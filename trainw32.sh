#!/bin/bash

#SBATCH --chdir /home/yju/HRNet-Semantic-Segmentation
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 2
#SBATCH --mem 192G
#SBATCH --time 20:00:00
#SBATCH --gres gpu:2
#SBATCH --account=vita

echo "izar $HOSTNAME"

module load gcc/8.4.0-cuda python/3.7.7
module load mvapich2 

source /home/yju/venvs/hrnet/bin/activate

echo STARTING AT `date`

GPU_NUM=2
PY_CMD="python -m torch.distributed.launch --nproc_per_node=$GPU_NUM"

CONFIG="seg_hrnet_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold1"
$PY_CMD tools/train.py --cfg experiments/swiss_okutama/$CONFIG.yaml
python tools/test.py --cfg experiments/swiss_okutama/$CONFIG.yaml \
                DATASET.TEST_SET /work/vita/datasets/Okutama-Swiss-dataset/crossval1/test.lst \
                TEST.MODEL_FILE /scratch/izar/yju/hrnet/output/swiss_okutama/$CONFIG/final_state.pth

CONFIG="seg_hrnet_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2"
$PY_CMD tools/train.py --cfg experiments/swiss_okutama/$CONFIG.yaml
python tools/test.py --cfg experiments/swiss_okutama/$CONFIG.yaml \
                DATASET.TEST_SET /work/vita/datasets/Okutama-Swiss-dataset/crossval2/test.lst \
                TEST.MODEL_FILE /scratch/izar/yju/hrnet/output/swiss_okutama/$CONFIG/final_state.pth
                
CONFIG="seg_hrnet_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold3"
$PY_CMD tools/train.py --cfg experiments/swiss_okutama/$CONFIG.yaml
python tools/test.py --cfg experiments/swiss_okutama/$CONFIG.yaml \
                DATASET.TEST_SET /work/vita/datasets/Okutama-Swiss-dataset/crossval3/test.lst \
                TEST.MODEL_FILE /scratch/izar/yju/hrnet/output/swiss_okutama/$CONFIG/final_state.pth

                
CONFIG="seg_hrnet_ocr_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2"
$PY_CMD tools/train.py --cfg experiments/swiss_okutama/$CONFIG.yaml
python tools/test.py --cfg experiments/swiss_okutama/$CONFIG.yaml \
                DATASET.TEST_SET /work/vita/datasets/Okutama-Swiss-dataset/crossval2/test.lst \
                TEST.MODEL_FILE /scratch/izar/yju/hrnet/output/swiss_okutama/$CONFIG/final_state.pth   
                
CONFIG="seg_hrnet_ocr_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold3"
$PY_CMD tools/train.py --cfg experiments/swiss_okutama/$CONFIG.yaml
python tools/test.py --cfg experiments/swiss_okutama/$CONFIG.yaml \
                DATASET.TEST_SET /work/vita/datasets/Okutama-Swiss-dataset/crossval3/test.lst \
                TEST.MODEL_FILE /scratch/izar/yju/hrnet/output/swiss_okutama/$CONFIG/final_state.pth                                     
                   
echo FINISHED at `date`
