RDMM example on synthetic data
==============================

This is a demo for the synthetic data experiments of the NeurIPS 2019 paper on Region-specific Diffeomorphic Metric Mapping.


Generate synthetic data
^^^^^^^^^^^^^^^^^^^^^^^

To start off we first show how to generate the synthetic data.

.. code:: shell

    cd demos/rdmm_synth_data_generation
    # by default uses 20 threads for data generation
    python demo_for_generation.py

The optional settings in *demo_for_generation.py* are as follows:

..  code:: shell

    Creates synthetic registration examples for RDMM related experiments

    optional arguments:
      -h, --help            show this help message and exit
      -dp DATA_SAVING_PATH, --data_saving_path DATA_SAVING_PATH
                            path of the folder saving synthesis data
      -di DATA_TASK_PATH, --data_task_path DATA_TASK_PATH
                            path of the folder recording data info for
                            registration tasks



RDMM Registration
^^^^^^^^^^^^^^^^^

The data generation may take minutes. Once the data are prepared, we can run RDMM registration by

.. code:: shell

    cd ..
    python example_2d_synth.py

The optional settings in *example_2d_synth.py* are as follows:

.. code:: shell

    Registration demo for 2d synthetic data

    optional arguments:
      -h, --help            show this help message and exit
      --expr_name EXPR_NAME
                            the name of the experiment
      --data_task_path DATA_TASK_PATH
                            the path of data task folder
      --model_name MODEL_NAME
                            non-parametric method, vsvf/lddmm/rdmm are currently
                            supported in this demo
      --use_predefined_weight_in_rdmm
                            this flag is only for RDMM model, if set true, the
                            predefined regularizer mask will be loaded and only
                            the momentum will be optimized; if set false, both
                            weight and momenutm will be jointly optimized
      --mermaid_setting_path MERMAID_SETTING_PATH
                            path of mermaid setting json

Once the registrations are done, we can check the results at the default data_task_path *./rdmm_synth_data_generation/data_task*.
