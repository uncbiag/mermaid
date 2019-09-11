"""
Similarity measures for the registration methods and factory to create similarity measures.
"""
from __future__ import absolute_import

from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod
import torch
from torch.autograd import Function
from .data_wrapper import AdaptVal
from . import utils
from math import floor
from numpy import log
from numpy import shape as numpy_shape
from . import forward_models as FM


class OTSimilarityHelper(Function):
    """Implements the pytorch function of optimal mass transport.
    """
    @staticmethod
    def forward(ctx, phi,I0,I1,multiplier0,multiplier1,spacing,nr_iterations_sinkhorn,std_sink):
        shape = numpy_shape(AdaptVal(I0).detach().cpu().numpy())
        simil = OTSimilarityGradient(spacing,shape,sinkhorn_iterations = nr_iterations_sinkhorn[0],std_dev = std_sink[0])
        result,other = simil.compute_similarity(I0,I1)
        multiplier0.copy_(AdaptVal(simil.multiplier0.data))
        multiplier1.copy_(AdaptVal(simil.multiplier1.data))
        ctx.save_for_backward(phi,I0,I1,multiplier0,multiplier1,spacing,nr_iterations_sinkhorn,std_sink)
        return result.data

    @staticmethod
    def backward(ctx, grad_output):
        phi,I0,I1,multiplier0,multiplier1,spacing,nr_iterations_sinkhorn,std_sink = ctx.saved_variables
        shape = numpy_shape(AdaptVal(I0).detach().cpu().numpy())
        simil = OTSimilarityGradient(spacing.data,shape,sinkhorn_iterations = nr_iterations_sinkhorn.data[0],std_dev = std_sink.data[0])
        grad_input = simil.compute_gradient(I0,I1,multiplier0,multiplier1)
        fm = FM.RHSLibrary(spacing.detach().cpu().numpy())
        result_gradient = fm.rhs_advect_map_multiNC(torch.unsqueeze(phi,0),torch.unsqueeze(grad_input,0))
        #result_gradient = -grad_input
        return -2*result_gradient,None,None,None,None,None,None,None



class OTSimilarityGradient(object):
    """Computes a regularized optimal transport distance between two densities.

    Formally:
    :math:`sim = W^2/(\\sigma^2)`
    """

    def __init__(self, spacing, shape, sinkhorn_iterations=300, std_dev=0.07):
        self.spacing = spacing
        self.shape = shape
        self.gibbs = []
        self.std_dev = std_dev
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.small_mass = 0.00001
        self.multiplier0 = None
        self.multiplier1 = None
        self.dim = len(self.spacing)

        for i in range(self.dim):
            self.gibbs.append(self.build_kernel_matrix(self.shape[i], self.std_dev))

        for i in range(self.dim):
            self.gibbs.append(self.build_kernel_matrix_gradient(self.shape[i], self.std_dev))

    def my_dot(self, a, b):
        """
               Dot product in pytorch
               :param a: tensor
               :param b: tensor
               :return: <a,b>
               """
        result = torch.mul(a, b)
        for i in range(a.dim()):
            result = torch.sum(result, 0)
        return result

    def my_sum(self, a):
        """
               Dot product in pytorch
               :param a: tensor
               :return: sum(a)
               """
        result = torch.sum(a, 0)
        for i in range(a.dim() - 1):
            result = torch.sum(result, 0)
        return result

    def build_kernel_matrix(self, length, std):
        """Computation of the gaussian kernel.

        :param length: length of the vector
        :param std: standard deviation of the gaussian kernel
        :return: :math:`\\exp(-|x_i - x_j|^2/\\sigma^2)`
        """
        x = torch.linspace(0, 1, length)
        x_col = x.unsqueeze(1)
        y_lin = x.unsqueeze(0)
        c = torch.abs(x_col - y_lin) ** 2
        return torch.exp(-torch.div(c, std ** 2))

    def build_kernel_matrix_gradient(self, length, std):
        """Computation of the gaussian first derivative kernel multiplied by :math:`1/2\\sigma^2`

        :param length: length of the vector
        :param std: standard deviation of the gaussian kernel
        :return: :math:`(x_i - x_j) \\exp(-|x_i - x_j|^2/\\sigma^2)`
        """
        x = torch.linspace(0, 1, length)
        x_col = x.unsqueeze(1)
        y_lin = x.unsqueeze(0)
        c = torch.abs(x_col - y_lin) ** 2
        return torch.mul((y_lin - x_col), torch.exp(-torch.div(c, std ** 2)))

    def kernel_multiplication(self, multiplier):
        """
               Computes the multiplication of a d-dimensional vector (d = 1,2 or 3) with the gaussian kernel K
               :param multiplier: the vector
               :param choice_kernel: the choice function that outputs the index in the kernel list.
               :return: K*multiplier
               """
        temp = None
        if self.dim == 1:
            temp = torch.matmul(self.gibbs[0], multiplier)

        elif self.dim == 2:
            temp = torch.matmul(self.gibbs[0], multiplier)
            temp = temp.permute(1, 0)
            temp = torch.matmul(self.gibbs[1], temp)
            temp = temp.permute(1, 0)

        elif self.dim == 3:
            ### multiplication along the first axis
            temp = torch.matmul(AdaptVal(self.gibbs[0]), AdaptVal(multiplier.permute(2, 0, 1)))
            temp = temp.permute(1, 2, 0)

            ### multiplication along the second axis
            temp = torch.matmul(AdaptVal(self.gibbs[1]), temp)

            ### multiplication along the third axis
            temp = temp.permute(0, 2, 1)  # z,x,y
            temp = torch.matmul(AdaptVal(self.gibbs[2]), temp)

            temp = temp.permute(0, 2, 1)

        return temp

    def kernel_multiplication_gradient_helper(self, multiplier, choice_kernel):
        """Computes the multiplication of a d-dimensional vector (d = 1,2 or 3) with the
        (derivative along a given axis) gaussian kernel and given by the choice_kernel function (give the axis).

       :param multiplier: the vector
       :param choice_kernel: the choice function that outputs the index in the kernel list.
       :return: K*multiplier
       """

        if self.dim == 1:
            temp = torch.matmul(self.gibbs[choice_kernel(0)], multiplier)

        elif self.dim == 2:
            temp = torch.matmul(self.gibbs[choice_kernel(0)], multiplier)
            temp = temp.permute(1, 0)
            temp = torch.matmul(self.gibbs[choice_kernel(1)], temp)
            temp = temp.permute(1, 0)

        elif self.dim == 3:
            ### multiplication along the first axis
            temp = torch.matmul(self.gibbs[choice_kernel(0)], multiplier.permute(2, 0, 1))
            temp = temp.permute(1, 2, 0)

            ### multiplication along the second axis
            temp = torch.matmul(self.gibbs[choice_kernel(1)], temp)

            ### multiplication along the third axis
            temp = temp.permute(0, 2, 1)  # z,x,y
            temp = torch.matmul(self.gibbs[choice_kernel(2)], temp)
            temp = temp.permute(0, 2, 1)
        return temp

    def set_choice_kernel_gibbs(self, i, offset):
        """Set the choice of the kernels for the computation of the gradient.

       :param i: the (python) index of the dimension
       :param offset: the dimension
       :return: the function for choosing the kernel
       """
        if i == -1:
            return lambda k: k
        else:
            return lambda k: k + (i == k) * offset
        return None

    def compute_similarity(self, I0, I1):
        """
       Computes the OT-based similarity measure between two densities.

       :param I0: first density
       :param I1: second density
       :return: W^2/sigma^2
       """
        ### pretreat densities by adding a small amount of mass to have non-zero coefficients (TODO:has to be fixed whether in log domain or directly)
        temp = torch.add(I0, self.small_mass)
        I0rescaled = torch.div(temp, self.my_sum(temp))
        temp2 = torch.add(I1, self.small_mass)
        I1rescaled = torch.div(temp2, self.my_sum(temp2))

        ### definition of the lagrange multiplier
        multiplier0 = torch.ones(I0.size())
        multiplier1 = torch.ones(I1.size())
        multiplier0.requires_grad= True
        multiplier1.requires_grad = True
        convergence = []
        ### iteration of sinkhorn loop
        for i in range(self.sinkhorn_iterations):
            multiplier0 = torch.div(I0rescaled, self.kernel_multiplication(multiplier1))
            multiplier1 = torch.div(I1rescaled, self.kernel_multiplication(multiplier0))
            convergence.append(log(
                AdaptVal(self.my_sum(torch.abs(I0rescaled - multiplier0 * self.kernel_multiplication(multiplier1))).data)
                    .item()))
        temp = self.my_dot(torch.log(multiplier0), I0rescaled) + self.my_dot(torch.log(multiplier1),
                                                                             I1rescaled) - self.my_dot(
            multiplier0, self.kernel_multiplication(multiplier1))

        self.multiplier0 = multiplier0
        self.multiplier1 = multiplier1
        return (self.std_dev ** 2) * temp, convergence

    def compute_gradient(self, I0, I1, multiplier0, multiplier1):
        """
               Compute the gradient of the similarity with respect to the grid points

               :param I0: first density
               :param I1: second density
               :param multiplier0: Lagrange multiplier for the first marginal
               :param multiplier1: Lagrange multiplier for the second marginal
               :return: Gradient wrt the grid
               """
        gradient = torch.zeros((self.dim,) + I0.size())
        for i in range(self.dim):
            choice_kernel = self.set_choice_kernel_gibbs(i, self.dim)
            gradient[i] = 2 * torch.mul(multiplier0,
                                        self.kernel_multiplication_gradient_helper(multiplier1, choice_kernel)) * I0.size()[i]
        return gradient


