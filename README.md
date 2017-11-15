# Image registration using pyTorch

CPU/GPU verwion of mermaid.

# Basic install
  * See mermaid.yaml for all the packages that need to be installed. This is easiest with anaconda.
  * conda install pytorch torchvision cuda80 -c soumith
  * conda install cffi
  * conda install -c simpleitk simpleitk
  * conda install sphinx
  * pip install pytorch-fft

After everything is installed, compile the documentation (there are also more detailed installation instructions)

      * cd mermaid
      * cd docs
      * make html


# Setup

* To compile the spatial transformer code simply go to the pyreg/libraries directory \
  for gpu user: execute 'sh make_cuda.sh'
  for cpu user: execute 'sh make_cpu.sh'
* pip install pytorch-fft
* go config to set CUDA_ON

# How to run
* run testRegistrationGeneric.py/testRegistrationMultiscale.py 
* settings are put together in config.txt
* use float32 in most cases !!!, float16 is not stable
* !!!!!!!!!!!!  most part of the codes have been examined. In case of failure, contact zyshen021@gmail.com
    
# Lastest Modification
  * 11.13    fix to adpat device
  * 11.12    optimize, fix bugs, 
  * 10.31:   unitest stn is added, see test_stn_cpu.py and test_stn_gpu.py
  * 10.29:   1D, 2D and 3D cuda STN have been implemented. see pyreg.libraries.functions.stn_nd
    *        stn = STNFunction_ND_BCXYZ(n_dim)
             stn.forward_stn(inputImage,inputGrids, output, n_dim, device_c, use_cuda=True)
             stn.backward_stn(inputImage, inputGrids, grad_input, grad_grids, grad_output, n_dim, device_c,use_cuda=True)


  * Important changes before 10.21:
    * add dataWarper.py, define some Variable Warppers, two are most common use
      *     # AdaptVal(x): Adaptive Warper: used to adapt the data type, implemented on the existed Tensor/Variable
                
            # MyTensor:  a adpative Tensor to create gpu, cpu and float16 Tensor

    * Changes are made in vector organization, batch is no longer calculated in loop
    * Memory efficient structure modification in "optimize" function
    * Real FFT is implemented for both speed and memory consideration, which needs the smoothing kernel should be symmetric 


# To Do
  * make svf_quasi_momentum stable
  * add smoothing unittest
  * modify other testRegistration*
  * add Tensorboard
  * schedule the main function
  
