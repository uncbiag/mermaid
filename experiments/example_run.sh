
#! /usr/bin/env bash

#source activate p27

python generic_2d_experiment_driver.py --cuda_device 0 --dataset_config dataset.json --config test2d_025.json --move_to_directory test_sweep --multi_gaussian_weights_stage_0 [0.,0.,0.,1.0] --sweep_value_name_a model.registration_model.forward_model.smoother.deep_smoother.total_variation_weight_penalty --sweep_values_a 0.01,0.1,1.0,10 --sweep_value_name_b model.registration_model.forward_model.smoother.omt_weight_penalty --sweep_values_b 1.,10.,100. --config_kvs optimizer.sgd.individual.lr=0.01\;model.optimizer.sgd.shared.lr=0.0





