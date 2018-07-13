
 <pre>
                                      _     _ 
                                     (_)   | |
  _ __ ___   ___ _ __ _ __ ___   __ _ _  __| |
 | '_ ` _ \ / _ \ '__| '_ ` _ \ / _` | |/ _` |
 | | | | | |  __/ |  | | | | | | (_| | | (_| |
 |_| |_| |_|\___|_|  |_| |_| |_|\__,_|_|\__,_|
                                                                                      
 </pre>                                       

# Image registration using pyTorch

Mermaid is a registration toolkit making use of automatic differentiation for rapid prototyping. It runs on the CPU and the GPU, though GPU acceleration only becomes obvious for large images or 3D volumes. 

# Basic installation

```diff
- We just changed the code to pytorch 0.4.0. So expect some issues. If you still want to run the old version, you can check out tag v0.2 (which works with pytorch 0.3.0 and python 2.7 or 3). Going forward all development will be based on python 3 as well.
```

A basic installation requires the installation of a few python packages. Most of these packages can be installed via conda, but a few do not have conda installers, but can be installed via pip. 

  * conda install pytorch torchvision -c pytorch
  * [for CPU only: conda install pytorch-cpu torchvision -c pytorch]
  * conda install future
  * conda install -c conda-forge itk
  * conda install sphinx
  * conda install pandas
  * conda install matplotlib
  * conda install scipy
  * conda install openpyxl
  * conda install -c anaconda cython
  * conda install -c anaconda scikit-image
  * pip install pytorch-fft
  * pip install pynrrd

IMPORTANT: pytorch 0.4.0 is required. The code will no longer run on 0.2.0 or 0.3.0

 The mermaid.yaml file in the install directory also contains information for all the packages that need to be installed. We will provide an anaconda installer in the (hopefully) near future.

# An alterative way to install everything

*WARNING: This does not work yet and has not been updated for pytorch 0.4.0

It is of course also possible to install everything into a virtual conda environment. For this, do the following:

   * conda create -n mermaid-py27 python=2.7 pip
   * source activate mermaid-py27
   * cd install
   * pip install -r requirements_python2_osx.txt  [Pick the right one for your operating system]
   * pip install -r requirements_pytorch_fft.txt

We are working on making the code compatible with python3, which is currently *not supported*. Once it is supported it will be possible to install doing the following:

   * conda create -n mermaid-py36 python=3.6 pip
   * source activate mermaid-py36
   * cd install
   * pip install -r requirements_python3_osx.txt  [Pick the right one for your operating system]
   * pip install -r requirements_pytorch_fft.txt

# After the installation

After everything is installed, compile the documentation (there are also more detailed installation instructions) by doing the following.

      * cd mermaid
      * cd docs
      * make html

Lastly, to compile the spatial transformer code simply go to the pyreg/libraries directory \
  for gpu user: execute 'sh make_cuda.sh'
  for cpu user: execute 'sh make_cpu.sh'

CUDA can be enabled by setting CUDA_ON to True in settings/compute_settings.json. All settings are contained in json configuration files in the settings directory. 

# How to run registrations

There are a few example registration implementations in the demos directory. Actual applications are contained in the apps directory.

# Things to know

* Use float32 in most cases, float16 is not stable
* Most parts of the codes have been examined. In case of failure of the GPU code contact zyshen021@gmail.com; for all other failures contact mn@cs.unc.edu or open an issue in the github issue tracker.
    
# Lastest modifications
  * 12.30: added diffusion, curvature, and total variation registration models.
  * 12.29: added multi Gaussian smoother
  * 12.19\
             add result_analysis: an excel result analysis\
             1. detailed results: saved in fig_save_path/results.xlsx\
             ** Sheet1:  #total_filename * #metrics  recorded by iteraton\
             ** batch_0: #batch_filename * #metric_by_label    recorded by iteration\
             ** batch_1: same as batch_0\
             ** ......\
             2. task results: saved in ../data/summary.xlsx\
             ** Sheet1:  task_name * #metrics  recorded by iteration




  * 12.16\
             add cuda nearest neighbour interpolation, and many fixes
  * 12.14\
             add evalution method iou, recall, precison(overlapping), dice
  * 12.13\
             dataManger into the apps/mm-reg, the system works through\
             transfer to fileio when reading image in data_pool\
             add adaptive_padding in fileio.py, to be favourable by the guassian fft
  * 12.12\
             add dataManager, Registration net, dataloader for typical dataset. see sample in prepare data\
             add 'data_pool.py', a generalized dataset pool class\
             support inter intra personal dataset likeo asis, support inter-personal like lpba, IBSR, CUMU dataset\
             support 2D and 3D dataset\
             support labeled and unlabeled dataset\
             support 2D img, label slicing
  * 12.11\
             support oasis inter intra registration\
             support saving results, /data/saved_result/taskname,  settings can be found in commandline parsers\
             support adaptiveNet (can be founded in smoother class) in llddmm mapping(tested),  which embedding networks between m and v\
             &nbsp;&nbsp;&nbsp;  (other possible combinations like [I_t, m and v], [I_0,m and v]  also implemented but still underdevelop)\
             add m_1 into loss for avoid explosion ( maybe useful in small time step, cannot avoid explosion in large time_step)

  * 12.05:   Added various momentum-based SVF models (SVFScalarMomentumImageNet, SVFScalarMomentumMapNet, SVFVectorMomentumImageNet, SVFVectorMomentumMapNet )
  * 12.05:   Added tests for the registration algorithms
  * 12.02:   Simplified interface. Petter parameters. Low-res capability (requires new arguent mapLowResFactor to be specified when calling the functions).
  * 11.26:   Created a separate IO class; use this for all file input/output
  * 11.13:   fix to adpat device
  * 11.12:   optimize, fix bugs, 
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


# Todo
  * make svf_quasi_momentum stable
  * add smoothing unittest
  * modify other testRegistration*
  * add Tensorboard
  * schedule the main function
  
