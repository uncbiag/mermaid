import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from torch.nn.modules.module import Module
from torch.autograd import Function

class SplineInterpolation_ND_BCXYZ(Module):
    """
    Spline transform code for nD spatial transforms. Uses the BCXYZ image format.
    """
    def __init__(self, spacing, spline_order):
        super(SplineInterpolation_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial dimension"""
        self.spline_order = spline_order
        """spline order"""

        self.n = spline_order
        self.Ns = None

        if self.n not in [2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError('Unknown spline order')

        self.poles = dict()
        self.poles[2] = torch.from_numpy(np.array([np.sqrt(8.) - 3.]))
        self.poles[3] = torch.from_numpy(np.array([np.sqrt(3.) - 2.]))
        self.poles[4] = torch.from_numpy(np.array([np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
                                                   np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0]))
        self.poles[5] = torch.from_numpy(
            np.array([np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0,
                      np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0]))
        self.poles[6] = torch.from_numpy(np.array([-0.48829458930304475513011803888378906211227916123938,
                                                   -0.081679271076237512597937765737059080653379610398148,
                                                   -0.0014141518083258177510872439765585925278641690553467]))
        self.poles[7] = torch.from_numpy(np.array([-0.53528043079643816554240378168164607183392315234269,
                                                   -0.12255461519232669051527226435935734360548654942730,
                                                   -0.0091486948096082769285930216516478534156925639545994]))
        self.poles[8] = torch.from_numpy(np.array([-0.57468690924876543053013930412874542429066157804125,
                                                   -0.16303526929728093524055189686073705223476814550830,
                                                   -0.023632294694844850023403919296361320612665920854629,
                                                   -0.00015382131064169091173935253018402160762964054070043]))
        self.poles[9] = torch.from_numpy(np.array([-0.60799738916862577900772082395428976943963471853991,
                                                   -0.20175052019315323879606468505597043468089886575747,
                                                   -0.043222608540481752133321142979429688265852380231497,
                                                   -0.0021213069031808184203048965578486234220548560988624]))

    def _scale_map_to_ijk(self, phi, spacing, sz_image):
        """
        Scales the map to the [0,i-1]x[0,j-1]x[0,k-1] format

        :param map: map in BxCxXxYxZ format
        :param spacing: spacing in XxYxZ format
        :param ijk-size of image that needs to be interpolated
        :return: returns the scaled map
        """
        sz = phi.size()

        scaling = np.array(list(sz_image[2:])).astype('float64')/np.array(list(sz[2:])).astype('float64') # to account for different number of pixels/voxels ijk coordinates (only physical coordinates are consistent)

        phi_scaled = torch.zeros_like(phi)
        ndim = len(spacing)

        for d in range(ndim):
            phi_scaled[:, d, ...] = phi[:, d, ...]*(scaling[d]/spacing[d])

        return phi_scaled

    def _slice_dim(self,val,idx,dim):

        if dim==1:
            return val[:,:,idx,...]
        elif dim==2:
            return val[:,:,:,idx,...]
        elif dim==3:
            return val[:,:,:,:,idx,...]
        else:
            raise ValueError('Dimension needs to be 1, 2, or 3')

    def _initial_causal_coefficient(self,c,z,tol,dim=1):
        """

        :param c: coefficient (numpy array)
        :param z: pole
        :param tol: tolerance
        :return:
        """

        if self.Ns is None:
            raise ValueError('Unknown data length')

        if dim not in [1,2,3]:
            raise ValueError('Dimension needs to be 1, 2, or 3')

        horizon = self.Ns[dim-1]
        if tol > 0:
            horizon = int(np.ceil(np.log(tol)/np.log(np.abs(z))))

        if horizon<self.Ns[dim-1]:
            # accelerated loop
            zn = z
            Sum = self._slice_dim(c,0,dim=dim)
            for n in range(1,horizon):
                Sum += zn*self._slice_dim(c,n,dim=dim)
                zn *= z

            return Sum
        else:
            # full loop
            zn = z
            iz = 1./z
            z2n = z**(self.Ns[dim-1]-1.)
            Sum = self._slice_dim(c,0,dim=dim) + z2n*self._slice_dim(c,-1,dim=dim)
            z2n *= z2n * iz
            for n in range(1,self.Ns[dim-1]-1):
                Sum += (zn + z2n )*self._slice_dim(c,n,dim=dim)
                zn *= z
                z2n *= iz

            return Sum/(1.-zn*zn)

    def _initial_anti_causal_coefficient(self,c,z,dim=1):
        """

        :param c: coefficients
        :param z: pole
        :return:
        """

        if self.Ns is None:
            raise ValueError('Unknown data length')

        return (z/(z*z-1.))*(z*self._slice_dim(c,-2,dim=dim) + self._slice_dim(c,-1,dim=dim))

    # todo: there is some code replication here (to compute the interpolation coefficients for the different dimensions)
    # todo: not clear (to me) how to avoid this without in-place operations which are not permitted by pyTorch

    def _convert_to_interpolation_cofficients_in_dim_1(self,c,z,tol):

        dim = 1

        nb_poles = len(z)

        lam = 1.
        # compute the overall gain
        for k in range(0, nb_poles):
            lam *= (1. - z[k]) * (1. - 1. / z[k])

        # apply the gain
        c *= lam

        # loop over all the poles
        for k in range(0, nb_poles):
            # causal initialization
            c[:, :, 0,...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            # causal recursion
            for n in range(1, self.Ns[dim-1]):
                c[:, :, n,...] = c[:, :, n,...] + z[k] * c[:, :, n - 1,...]
            # anti-causal initialization
            c[:, :, -1,...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            # anti-causal recursion
            for n in range(self.Ns[dim-1] - 2, -1, -1):
                c[:, :, n,...] = z[k] * (c[:, :, n + 1,...] - c[:, :, n,...])

        return c

    def _convert_to_interpolation_cofficients_in_dim_2(self,c,z,tol):

        dim = 2

        nb_poles = len(z)

        lam = 1.
        # compute the overall gain
        for k in range(0, nb_poles):
            lam *= (1. - z[k]) * (1. - 1. / z[k])

        # apply the gain
        c *= lam

        # loop over all the poles
        for k in range(0, nb_poles):
            # causal initialization
            c[:, :, :, 0,...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            # causal recursion
            for n in range(1, self.Ns[dim-1]):
                c[:, :, :, n,...] = c[:, :, :, n,...] + z[k] * c[:, :, :, n - 1,...]
            # anti-causal initialization
            c[:, :, :, -1,...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            # anti-causal recursion
            for n in range(self.Ns[dim-1] - 2, -1, -1):
                c[:, :, :, n,...] = z[k] * (c[:, :, :, n + 1,...] - c[:, :, :, n,...])

        return c

    def _convert_to_interpolation_cofficients_in_dim_3(self,c,z,tol):

        dim = 3

        nb_poles = len(z)

        lam = 1.
        # compute the overall gain
        for k in range(0, nb_poles):
            lam *= (1. - z[k]) * (1. - 1. / z[k])

        # apply the gain
        c *= lam

        # loop over all the poles
        for k in range(0, nb_poles):
            # causal initialization
            c[:, :, :, :, 0,...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            # causal recursion
            for n in range(1, self.Ns[dim-1]):
                c[:, :, :, :, n,...] = c[:, :, :, :, n,...] + z[k] * c[:, :, :, :, n - 1,...]
            # anti-causal initialization
            c[:, :, :, :, -1,...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            # anti-causal recursion
            for n in range(self.Ns[dim-1] - 2, -1, -1):
                c[:, :, :, :, n,...] = z[k] * (c[:, :, :, :, n + 1,...] - c[:, :, :, :, n,...])

        return c

    def _convert_to_interpolation_cofficients_in_dim(self,c,z,tol,dim=1):

        if dim==1:
            cr = self._convert_to_interpolation_cofficients_in_dim_1(c,z,tol)
        elif dim==2:
            cr = self._convert_to_interpolation_cofficients_in_dim_2(c,z,tol)
        elif dim==3:
            cr = self._convert_to_interpolation_cofficients_in_dim_3(c,z,tol)
        else:
            raise ValueError('not yet implemented')

        return cr


    def _convert_to_interpolation_coefficients(self,s,z,tol):
        """

        :param s: input signal
        :param z: poles
        :param tol: tolerance
        :return:
        """

        sz = s.size()
        dim = len(sz)-2
        if dim not in [1,2,3]:
            raise ValueError('Signal needs to be of dimensions 1, 2, or 3 and in format B x C x X x Y x Z')

        c = torch.zeros_like(s)
        c[:] = s

        self.Ns = list(s.size()[2:])
        if np.any(np.array(self.Ns)<=1):
            raise ValueError('Expected at least two values, but at least one of the dimensions has less')

        # do this dimension by dimension (as the filter is separable)
        for d in range(dim):
            c = self._convert_to_interpolation_cofficients_in_dim(c,z,tol,dim=d+1)

        return c

    def get_interpolation_coefficients(self,s,tol=0):

        return self._convert_to_interpolation_coefficients(s,self.poles[self.n],tol)

    def _compute_interpolation_weights(self,x):

        sz = x.size()
        dim = sz[1]

        index = torch.LongTensor(*([self.n+1]+list(x.size())))
        weight = Variable(torch.zeros(*([self.n+1]+list(x.size()))), requires_grad=False)

        # compute the interpolation indexes
        # todo: can likely be simplified (without loop over dimension)
        if self.n%2==0: # even
            for d in range(dim):
                i = (torch.floor(x[:,d,...].data + 0.5) - self.n/2)
                for k in range(0,self.n+1):
                    index[k,:,d,...] = i+k
        else:
            for d in range(dim):
                i = (torch.floor(x[:,d,...].data)-self.n/2)
                for k in range(0,self.n+1):
                    index[k,:,d,...] = i+k

        # compute the weights
        if self.n==2:
            w = x - Variable(index[1,...].float(),requires_grad=False)
            weight[1,...] = 3.0 / 4.0 - w * w
            weight[2,...] = (1.0 / 2.0) * (w - weight[1,...] + 1.0)
            weight[0,...] = 1.0 - weight[1,...] - weight[2,...]
        elif self.n==3:
            w = x - Variable(index[1,...].float(),requires_grad=False)
            weight[3,...] = (1.0 / 6.0) * w * w * w
            weight[0,...] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - weight[3,...]
            weight[2,...] = w + weight[0,...] - 2.0 * weight[3,...]
            weight[1,...] = 1.0 - weight[0,...] - weight[2,...] - weight[3,...]
        elif self.n==4:
            w = x - Variable(index[2].float(),requires_grad=False)
            w2 = w * w
            t = (1.0 / 6.0) * w2
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0]
            weight[0] *= (1.0 / 24.0) * weight[0]
            t0 = w * (t - 11.0 / 24.0)
            t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t)
            weight[1] = t1 + t0
            weight[3] = t1 - t0
            weight[4] = weight[0] + t0 + (1.0 / 2.0) * w
            weight[2] = 1.0 - weight[0] - weight[1] - weight[3] - weight[4]
        elif self.n==5:
            w = x - Variable(index[2].float(),requires_grad=False)
            w2 = w * w
            weight[5] = (1.0 / 120.0) * w * w2 * w2
            w2 -= w
            w4 = w2 * w2
            w -= 1.0 / 2.0
            t = w2 * (w2 - 3.0)
            weight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - weight[5]
            t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0)
            t1 = (-1.0 / 12.0) * w * (t + 4.0)
            weight[2] = t0 + t1
            weight[3] = t0 - t1
            t0 = (1.0 / 16.0) * (9.0 / 5.0 - t)
            t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0)
            weight[1] = t0 + t1
            weight[4] = t0 - t1
        elif self.n==6:
            w = x - Variable(index[3].float(),requires_grad=False)
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0] * weight[0]
            weight[0] *= weight[0] / 720.0
            weight[1] = (361.0 / 192.0 - w * (59.0 / 8.0 + w
                                                * (-185.0 / 16.0 + w * (25.0 / 3.0 + w * (-5.0 / 2.0 + w)
                                                                        * (1.0 / 2.0 + w))))) / 120.0
            weight[2] = (10543.0 / 960.0 + w * (-289.0 / 16.0 + w
                                                 * (79.0 / 16.0 + w * (43.0 / 6.0 + w * (-17.0 / 4.0 + w
                                                                                         * (-1.0 + w)))))) / 48.0
            w2 = w * w
            weight[3] = (5887.0 / 320.0 - w2 * (231.0 / 16.0 - w2
                                                 * (21.0 / 4.0 - w2))) / 36.0
            weight[4] = (10543.0 / 960.0 + w * (289.0 / 16.0 + w
                                                 * (79.0 / 16.0 + w * (-43.0 / 6.0 + w * (-17.0 / 4.0 + w
                                                                                          * (1.0 + w)))))) / 48.0
            weight[6] = 1.0 / 2.0 + w
            weight[6] *= weight[6] * weight[6]
            weight[6] *= weight[6] / 720.0
            weight[5] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[6]
        elif self.n==7:
            w = x - Variable(index[3].float(),requires_grad=False)
            weight[0] = 1.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0] * weight[0]
            weight[0] *= (1.0 - w) / 5040.0
            w2 = w * w
            weight[1] = (120.0 / 7.0 + w * (-56.0 + w * (72.0 + w
                                                          * (-40.0 + w2 * (12.0 + w * (-6.0 + w)))))) / 720.0
            weight[2] = (397.0 / 7.0 - w * (245.0 / 3.0 + w * (-15.0 + w
                                                                * (-95.0 / 3.0 + w * (15.0 + w * (5.0 + w
                                                                                                  * (-5.0 + w))))))) / 240.0
            weight[3] = (2416.0 / 35.0 + w2 * (-48.0 + w2 * (16.0 + w2
                                                              * (-4.0 + w)))) / 144.0
            weight[4] = (1191.0 / 35.0 - w * (-49.0 + w * (-9.0 + w
                                                            * (19.0 + w * (-3.0 + w) * (-3.0 + w2))))) / 144.0
            weight[5] = (40.0 / 7.0 + w * (56.0 / 3.0 + w * (24.0 + w
                                                              * (40.0 / 3.0 + w2 * (-4.0 + w * (-2.0 + w)))))) / 240.0
            weight[7] = w2
            weight[7] *= weight[7] * weight[7]
            weight[7] *= w / 5040.0
            weight[6] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[5] - weight[7]
        elif self.n==8:
            w = x - Variable(index[4].float(),requires_grad=False)
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0]
            weight[0] *= weight[0] / 40320.0
            w2 = w * w
            weight[1] = (39.0 / 16.0 - w * (6.0 + w * (-9.0 / 2.0 + w2)))\
                          *(21.0 / 16.0 + w * (-15.0 / 4.0 + w * (9.0 / 2.0 + w
                                                                  * (-3.0 + w)))) / 5040.0;
            weight[2] = (82903.0 / 1792.0 + w * (-4177.0 / 32.0 + w
                                                  * (2275.0 / 16.0 + w * (-487.0 / 8.0 + w * (-85.0 / 8.0 + w
                                                                                              * (41.0 / 2.0 + w * (
                                        -5.0 + w * (-2.0 + w)))))))) / 1440.0
            weight[3] = (310661.0 / 1792.0 - w * (14219.0 / 64.0 + w
                                                   * (-199.0 / 8.0 + w * (-1327.0 / 16.0 + w * (245.0 / 8.0 + w
                                                                                                * (53.0 / 4.0 + w * (
                                        -8.0 + w * (-1.0 + w)))))))) / 720.0
            weight[4] = (2337507.0 / 8960.0 + w2 * (-2601.0 / 16.0 + w2
                                                     * (387.0 / 8.0 + w2 * (-9.0 + w2)))) / 576.0
            weight[5] = (310661.0 / 1792.0 - w * (-14219.0 / 64.0 + w
                                                   * (-199.0 / 8.0 + w * (1327.0 / 16.0 + w * (245.0 / 8.0 + w
                                                                                               * (-53.0 / 4.0 + w * (
                                        -8.0 + w * (1.0 + w)))))))) / 720.0
            weight[7] = (39.0 / 16.0 - w * (-6.0 + w * (-9.0 / 2.0 + w2)))*(21.0 / 16.0 + w * (15.0 / 4.0 + w * (9.0 / 2.0 + w
                                                   * (3.0 + w)))) / 5040.0
            weight[8] = 1.0 / 2.0 + w
            weight[8] *= weight[8]
            weight[8] *= weight[8]
            weight[8] *= weight[8] / 40320.0
            weight[6] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[5] - weight[7] - weight[8]
        elif self.n==9:
            w = x - Variable(index[4].float(),requires_grad=False)
            weight[0] = 1.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0]
            weight[0] *= weight[0] * (1.0 - w) / 362880.0
            weight[1] = (502.0 / 9.0 + w * (-246.0 + w * (472.0 + w
                                                           * (-504.0 + w * (308.0 + w * (-84.0 + w * (-56.0 / 3.0 + w
                                                                                                      * (24.0 + w * (
                                        -8.0 + w))))))))) / 40320.0
            weight[2] = (3652.0 / 9.0 - w * (2023.0 / 2.0 + w * (-952.0 + w
                                                                  * (938.0 / 3.0 + w * (112.0 + w * (-119.0 + w * (56.0 / 3.0 + w
                                                                                                                   * (14.0 + w * (
                                        -7.0 + w))))))))) / 10080.0
            weight[3] = (44117.0 / 42.0 + w * (-2427.0 / 2.0 + w * (66.0 + w
                                                                      * (434.0 + w * (-129.0 + w * (-69.0 + w * (34.0 + w * (6.0 + w
                                                                                                                             * (-6.0 + w))))))))) / 4320.0
            w2 = w * w
            weight[4] = (78095.0 / 63.0 - w2 * (700.0 + w2 * (-190.0 + w2
                                                               * (100.0 / 3.0 + w2 * (-5.0 + w))))) / 2880.0
            weight[5] = (44117.0 / 63.0 + w * (809.0 + w * (44.0 + w
                                                             * (-868.0 / 3.0 + w * (-86.0 + w * (46.0 + w * (68.0 / 3.0 + w
                                                                                                             * (-4.0 + w * (
                                        -4.0 + w))))))))) / 2880.0
            weight[6] = (3652.0 / 21.0 - w * (-867.0 / 2.0 + w * (-408.0 + w
                                                                   * (-134.0 + w * (48.0 + w * (51.0 + w * (-4.0 + w) * (-1.0 + w)
                                                                                                * (2.0 + w))))))) / 4320.0
            weight[7] = (251.0 / 18.0 + w * (123.0 / 2.0 + w * (118.0 + w
                                                                 * (126.0 + w * (77.0 + w * (21.0 + w * (-14.0 / 3.0 + w
                                                                                                         * (-6.0 + w * (
                                        -2.0 + w))))))))) / 10080.0
            weight[9] = w2 * w2
            weight[9] *= weight[9] * w / 362880.0
            weight[8] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3]- weight[4] - weight[5] - weight[6] - weight[7] - weight[9]
        else:
            raise ValueError('Unsupported spline order')

        return index,weight

    def _interpolate_values(self,c,x):

        sz = c.size()
        dim = x.size()[1]

        if dim not in [1,2,3]:
            raise ValueError('Only dimensions 1, 2, and 3 are currently supported')

        index,weight = self._compute_interpolation_weights(x)

        # apply the mirror boundary conditions
        for d in range(dim):

            width = sz[2+d]
            width2 = 2 * width - 2

            lt_z = (index[:,:,d,...]<0)
            ge_z = (index[:,:,d,...]>=0)

            index[:,:,d,...][lt_z] = (-index[:,:,d,...][lt_z] - width2 * ((-index[:,:,d,...][lt_z]) / width2))
            index[:,:,d,...][ge_z] = (index[:,:,d,...][ge_z] - width2 * (index[:,:,d,...][ge_z] / width2))

            ge_w = (index[:,:,d,...]>=width)
            index[:,:,d,...][ge_w] = width2 - index[:,:,d,...][ge_w]

        # perform interpolation

        sz_weight = weight.size()
        batch_size = c.size()[0]
        nr_of_channels = c.size()[1]
        w = Variable(torch.zeros([batch_size,nr_of_channels] + list(sz_weight[3:])), requires_grad=False)

        if dim==1:
            for b in range(0,batch_size):
                for ch in range(0,nr_of_channels):
                    for k1 in range(0,self.n+1):
                        w[b,ch,...] = w[b,ch,...] + weight[k1,b,0,...] * c[b,ch,...][(index[k1,b,0,...])]
        elif dim==2:
            for b in range(0, batch_size):
                for ch in range(0, nr_of_channels):
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n+1):
                            w[b, ch, ...] = w[b, ch, ...] \
                                            + weight[k1, b, 0, ...] * weight[k2,b,1,...] \
                                            * c[b, ch, ...][(index[k1, b, 0, ...]),(index[k2,b,1,...])]
        elif dim ==3:
            for b in range(0, batch_size):
                for ch in range(0, nr_of_channels):
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n + 1):
                            for k3 in range(0, self.n+1):
                                w[b, ch, ...] = w[b, ch, ...] \
                                                + weight[k1, b, 0, ...] * weight[k2, b, 1, ...] * weight[k3,b,2,...] \
                                                * c[b, ch, ...][(index[k1, b, 0, ...]), (index[k2, b, 1, ...]), (index[k3, b, 2, ...])]
        else:
            raise ValueError('Dimension needs to be 1, 2, or 3.')

        interpolated = w

        return interpolated

    def interpolate(self,c,xs):

        r = self._interpolate_values(c,xs)

        return r

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        print('Computing spline interpolation')

        # compute interpolation coefficients
        c = self.get_interpolation_coefficients(input1)
        interpolated_values = self.interpolate(c, self._scale_map_to_ijk(input2,self.spacing,input1.size()))

        return interpolated_values

# for testing

def test_me(test_dim=1):

    import utils

    testDim = test_dim

    if testDim==1:
        #s = np.array([20,-15,10,-5,5,-12,12,-20,20,-30,30,-7,7,-3,3,-20,20,-1,1,-5,5,3,2,1])
        s = np.array([1.,1.,1.,1.,1.,1.])
        spacing = np.array([1.])
        #s = np.tile(s,2)
        x = np.arange(0,len(s))
        #
        xi = np.arange(0,len(s),0.1)

        s_torch_orig = Variable(torch.from_numpy(s.astype('float32')), requires_grad=True)
        xi_torch_orig = Variable(torch.from_numpy(xi.astype('float32')), requires_grad=True)

        s_torch = s_torch_orig.view(torch.Size([1, 1] + list(s_torch_orig.size())))
        # s_torch = torch.cat((s_torch,s_torch),0)
        # s_torch = torch.cat((s_torch,0.5*s_torch),1)

        xi_torch = xi_torch_orig.view(torch.Size([1, 1] + list(xi_torch_orig.size())))
        # xi_torch = torch.cat((xi_torch,xi_torch),0)
        # xi_torch = xi_torch_orig

    elif testDim==2:
        s = np.random.rand(10,10)

        #s = np.random.rand(1, 10)
        #s = np.tile(s,(10,1))

        #s = np.random.rand(10, 1)
        #s = np.tile(s, (1,10))

        #s = np.ones([10,1])
        #s = np.tile(s, (1, 10))

        x = utils.identity_map_multiN([1,1,10,10],[1,1])
        xi = utils.identity_map_multiN([1,1,20,20],[0.5,0.5])
        #xi = xi*10

        spacing = np.array([1.,1.])

        s_torch_orig = Variable(torch.from_numpy(s.astype('float32')), requires_grad=True)
        s_torch = s_torch_orig.view(torch.Size([1, 1] + list(s_torch_orig.size())))
        xi_torch = Variable(torch.from_numpy(xi.astype('float32')), requires_grad=True)

    else:
        raise ValueError('unsupported test dimension')


    # do the interpolation

    si = SplineInterpolation_ND_BCXYZ(spacing,spline_order=3)

    ctst = si.get_interpolation_coefficients(s_torch)
    si_tst = si.interpolate(ctst,xi_torch)

    val = (si_tst*si_tst).sum()
    val.backward()

    # do the plotting

    if testDim==1:
        plt.plot(x,s)
        plt.plot(xi,si_tst[0,0,...].data.numpy())

        plt.show()
    elif testDim==2:
        plt.subplot(121)
        plt.imshow(s)
        plt.clim(0,1.5)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(si_tst[0, 0, ...].data.numpy())
        plt.clim(0,1.5)
        plt.colorbar()
        plt.show()
    else:
        raise ValueError('Unsupported dimension')


#test_me(2)