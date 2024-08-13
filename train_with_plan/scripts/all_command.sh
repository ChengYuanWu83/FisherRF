python ./train_with_plan/active_train_nbv.py -s /home/nmsl/nbv_simulator_data/test -m ./exp_results/test1\
        --iterations 3000 --eval --method=H_reg --schema test\
        --num_init_views 1 --interval_epochs 0 --maximum_view 2\
        --add_view 1 --iteration_base 2000 --white_background --save_ply_each_time


        python ./train_with_plan/active_train_bvs.py -s /home/nmsl/nbv_simulator_data/circular_75s -m ./exp_results/circular_75s_bvs\
        --iterations 30000 --eval --method=H_reg --schema test\
        --num_init_views 1 --interval_epochs 300 --maximum_view 15\
        --add_view 1 --iteration_base 0 --white_background --save_ply_each_time

        python ./train_with_plan/active_train_bvs.py -s /home/nmsl/nbv_simulator_data/circular2 -m ./exp_results/test\
                --iterations 5000 --eval --method=rand --schema test\
                --num_init_views 1 --interval_epochs 500 --maximum_view 3\
                --add_view 1 --iteration_base 0 --white_background 



python ./train_with_plan/nbv_planner.py --experiment_path /home/nmsl/nbv_simulator_data/test --radius 3



python ./train_with_plan/train_extend.py -s /home/nmsl/nbv_simulator_data/test -m ./exp_results/test1 --iterations 10000 --white_background --eval

#capture
python ./train_with_plan/circular_capture.py --experiment_path /home/nmsl/nbv_simulator_data/circular_50s --record --radius_start 3 --radius_end 10 --test_set --time_budget 50

python ./train_with_plan/random_capture.py --experiment_path /home/nmsl/nbv_simulator_data/test --set_type train --radius_start 3 --radius_end 10 --candidate_views_num 50


python ./

#eval
python render.py -s /home/nmsl/nbv_simulator_data/test -m ./exp_results/test/final --eval
python metrics.py -m ./exp_results/test/final