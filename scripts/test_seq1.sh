DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema score_test --iterations 30000  --white_background
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema score_test --iterations 30000  --white_background