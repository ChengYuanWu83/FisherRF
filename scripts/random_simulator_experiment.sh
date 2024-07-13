DATASET_PATH=$1
EXP_PATH=$2

echo python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} --eval --schema score_test --iterations 30000  --white_background
python simulator_experiment.py -s $DATASET_PATH -m ${EXP_PATH} --eval --schema score_test --iterations 30000  --white_background 