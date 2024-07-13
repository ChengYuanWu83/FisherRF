DATASET_PATH=$1
EXP_PATH=$2

echo python train_extend.py -s $DATASET_PATH -m ${EXP_PATH} --iterations 2000 --white_background
python train_extend.py -s $DATASET_PATH -m ${EXP_PATH} --iterations 2000 --white_background