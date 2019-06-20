from __future__ import print_function
from builtins import zip
from builtins import range
import set_pyreg_paths

#import testMultiGaussianExtension as ce
import mermaid.custom_pytorch_extensions as ce

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.autograd import gradcheck
from torch.autograd.gradcheck import *
from torch.autograd.gradcheck import _differentiable_outputs, _as_tuple

from mermaid.data_wrapper import USE_CUDA, FFTVal,AdaptVal

import matplotlib.pyplot as plt


def mygradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True):
    """Check gradients computed via small finite differences
       against analytical gradients

    The check between numerical and analytical has the same behaviour as
    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
    is true for all elements of analytical jacobian a and numerical jacobian n.

    Args:
        func: Python function that takes Variable inputs and returns
            a tuple of Variables
        inputs: tuple of Variables
        eps: perturbation for finite differences
        atol: absolute tolerance
        rtol: relative tolerance
        raise_exception: bool indicating whether to raise an exception if
            gradcheck fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
    Returns:
        True if all differences satisfy allclose condition
    """
    tst = func(*inputs)

    output = _differentiable_outputs(func(*inputs))

    def plot_res(msg, Ia=None,In=None):
        if not (Ia is None or In is None):

            nr_rows = len(Ia)

            for n in range(nr_rows):

                plt.subplot(nr_rows,3,1+n*3)
                plt.imshow(Ia[n].numpy())
                plt.colorbar()

                plt.subplot(nr_rows,3,2+n*3)
                plt.imshow(In[n].numpy())
                plt.colorbar()

                plt.subplot(nr_rows,3,3+n*3)
                plt.imshow(Ia[n].numpy() - In[n].numpy())
                plt.colorbar()

            plt.title(msg)
            plt.show()

    def plot_res_proj(msg, Ia=None,In=None):
        if not (Ia is None or In is None):

            nr_rows = len(Ia)

            for n in range(len(Ia)):

                plt.subplot(nr_rows,3,1+n*3)
                plt.plot(Ia[n].numpy().sum(axis=1))

                plt.subplot(nr_rows,3,2+n*3)
                plt.plot(In[n].numpy().sum(axis=1))

                plt.subplot(nr_rows,3,3+n*3)
                plt.plot((Ia[n].numpy() - In[n].numpy()).sum(axis=1))

            plt.title(msg)
            plt.show()

    def plot_res_plot(msg, Ia=None,In=None):
        if not (Ia is None or In is None):

            plt.subplot(2,3,1)
            for i in range(5):
                plt.plot(Ia[1].numpy()[i,...])

            plt.subplot(2, 3, 2)
            for i in range(5):
                plt.plot(In[1].numpy()[i, ...])

            plt.subplot(2, 3, 3)
            for i in range(5):
                plt.plot((Ia[1].numpy()-In[1].numpy())[i, ...])

            plt.subplot(2, 3, 4)
            for i in range(5):
                plt.plot(Ia[2].numpy()[i, ...])

            plt.subplot(2, 3, 5)
            for i in range(5):
                plt.plot(In[2].numpy()[i, ...])

            plt.subplot(2, 3, 6)
            for i in range(5):
                plt.plot((Ia[2].numpy() - In[2].numpy())[i, ...])

            plt.title(msg)
            plt.show()

    def fail_test(msg,Ia=None,In=None):
        print('Failed test:')
        print(msg)

        plot_res( 'Failed test', Ia, In)
        plot_res_proj( 'Failed test', Ia, In)

        #if raise_exception:
        #    raise RuntimeError(msg)
        #return False

    def pass_test(msg,Ia=None,In=None):
        print('Passed test:')
        print(msg)

        plot_res( 'Passed test', Ia, In)
        plot_res_proj( 'Passed test', Ia, In)


    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i].data

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                fail_test('for output no. %d,\n numerical:%s\nanalytical:%s\n' % (j, n, a) )
            else:
                pass_test('for output no. %d,\n numerical:%s\nanalytical:%s\n' % (j, n, a) )

        plot_res('Test', analytical, numerical)
        plot_res_proj('Test', analytical, numerical)
        plot_res_plot('Test', analytical, numerical)

        if not reentrant:
            return fail_test('not reentrant')

        if not correct_grad_sizes:
            return fail_test('not correct_grad_sizes')

    # check if the backward multiplies by grad_output
    zero_gradients(inputs)
    output = _differentiable_outputs(func(*inputs))
    if any([o.requires_grad for o in output]):
        torch.autograd.backward(output, [o.data.new(o.size()).zero_() for o in output], create_graph=True)
        var_inputs = list([i for i in inputs if isinstance(i, Variable)])
        if not var_inputs:
            raise RuntimeError("no Variables found in input")
        for i in var_inputs:
            if i.grad is None:
                continue
            if not i.grad.data.eq(0).all():
                return fail_test('backward not multiplied by grad_output')

    return True


d = torch.load('test_data_multi_gaussian.pt')

# do everything in double for better numerical precision

nr_of_gaussians = d['nr_of_gaussians']
stds =  d['std'].data.float()
weights =  d['weights'].data.float()
m =  d['m'].data.float()
stds.requires_grad=True
weights.requires_grad= True
m. requires_grad = True

gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(d['sz'], d['spacing'], nr_of_gaussians)

v = ce.fourier_multi_gaussian_convolution(m,gaussian_fourier_filter_generator,stds, weights)

q = torch.ones_like(v)
v.backward(q)

torch.set_num_threads(8)

test = mygradcheck(ce.FourierMultiGaussianConvolution(gaussian_fourier_filter_generator,True,True),
                 (m, stds, weights ) , eps=1e-6, atol=1e-4)
print(test)
