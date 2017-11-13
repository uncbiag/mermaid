<<<<<<< HEAD
=======
# Image registration using pyTorch

Gpu version of proj mermaid

# Setup

* To compile the spatial transformer code simply go to the pyreg/libraries directory and exectute 'sh make_cuda.sh'
* To run the code import set_pyreg_paths 

# How to run
* run testRegistrationGeneric.py/testRegistrationMultiscale.py settings are put together in config.txt.
* use float32 in most cases !!!, float16 is not stable
* !!!!!!!!!!!!  most part of the codes has been examined. if fails, contact zyshen021@gmail.com
    
# Lastest Modification
  * 11.12    optimize, fix bugs, 
  * 10.31:   unitest stn is added, see test_stn_cpu.py and test_stn_gpu.py
  * 10.29:   1D, 2D and 3D cuda STN have been implemented. see pyreg.libraries.functions.stn_nd
    *        stn = STNFunction_ND_BCXYZ(n_dim)
             stn.forward_stn(inputImage,inputGrids, output, n_dim, device_c, use_cuda=True)
             stn.backward_stn(inputImage, inputGrids, grad_input, grad_grids, grad_output, n_dim, device_c,use_cuda=True)



  * Important changes before 10.21:
    * add dataWapper.py, define some Variable Warppers, two are most common use
      *     # AdpatVal(x): Adaptive Warper: used to adapt the data type, implemented on the existed Tensor/Variable
                
            # MyTensor:  a adpative Tensor to create gpu, cpu and float16 Tensor

    * Changes are made in vector organization, batch is no longer calculated in loop
    * Memory efficient structure modification in "optimize" function
    * Real FFT is implemented for both speed and memory consideration, which needs the smoothing kernel should be symmetric 

  
  
    
# To Do
  * add smoothing unittest
  * modify other testRegistration*
  * add Tensorboard
  * schedule the main function
  
  
>>>>>>> master
