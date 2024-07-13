DATASET_PATH=$1
EXP_PATH=$2

echo python train.py -s $DATASET_PATH -m ${EXP_PATH} --iterations 30000 --white_background
python train.py -s $DATASET_PATH -m ${EXP_PATH} --iterations 30000 --white_background