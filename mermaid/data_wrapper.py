from __future__ import absolute_import
import torch
from mermaid.config_parser import CUDA_ON, USE_FLOAT16

# ----------------- global setting ----------------------------------------
USE_CUDA = CUDA_ON and torch.cuda.is_available()

# --------------------   My Tensor -------------------------
# a warped version of Tensor to adapt gpu, cpu and float16
if USE_CUDA:
    MyLongTensor = torch.cuda.LongTensor
    if not USE_FLOAT16:
        MyTensor = torch.cuda.FloatTensor
    else:
        MyTensor = torch.cuda.HalfTensor
else:
    MyTensor = torch.FloatTensor
    MyLongTensor = torch.LongTensor

# ------------------  ApdatVal --------------------------
# Adaptive Warper: used to adapt the data type, implemented on the existed Tensor/Variable
def AdaptVal(x):
    """ adapt float32/16, gpu/cpu, float 16 is not recommended to use for it is not stable"""
    if USE_CUDA:
        if not USE_FLOAT16:
            return x.cuda()
        else:
            return x.cuda().half()
    else:
        return x


# -------------------- STN ------------------------------
# specific to the STN Function

if USE_CUDA:
    STNTensor = torch.cuda.FloatTensor
else:
    STNTensor = torch.FloatTensor


def STNVal(x, ini):
    """
    the cuda version of stn is writing in float32
    so the input would first be converted into float32,
    the output would be converted to adaptive type
    """
    if USE_CUDA:
        if USE_FLOAT16:
            if ini == 1:
                return x.float()
            elif ini == -1:
                return x.half()
            else:
                raise ValueError('ini should be 1 or -1')
        else:
            return x
    else:
        return x


# ------------------ FFT ----------------------------
# specific to FFT Function
# do same thing as  the STNVal

FFTVal = STNVal
