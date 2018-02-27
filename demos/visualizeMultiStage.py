import set_pyreg_paths
import torch
from torch.autograd import Variable

import pyreg.smoother_factory as SF
import pyreg.deep_smoothers as DS
import pyreg.utils as utils
import pyreg.image_sampling as IS

import pyreg.module_parameters as pars

import numpy as np

import matplotlib.pyplot as plt

import os

json_file = '../'
output_dir = '../experiments/test_out'
individual_dir = os.path.join(output_dir,'individual')
shared_dir = os.path.join(output_dir,'shared')


