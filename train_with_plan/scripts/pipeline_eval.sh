exp_name=$1
DATASET_PATH=/home/nmsl/nbv_simulator_data/$exp_name
EXP_PATH=/home/nmsl/FisherRF/exp_results/$exp_name

#eval
python render.py -s $DATASET_PATH -m $EXP_PATH/final --eval
python metrics.py -m $EXP_PATH/final

