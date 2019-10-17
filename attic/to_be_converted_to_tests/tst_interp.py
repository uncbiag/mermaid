import torch
import matplotlib.pyplot as plt

r = torch.load('tst_interp.pt')

input1 = r['input1']
input2 = r['input2']
p_output = r['output']

zero_boundary = 'zeros'
mode = 'bilinear'

# todo double check, it seems no transpose is need for 2d, already in height width design
input2_ordered = torch.zeros_like(input2)
input2_ordered[:, 0, ...] = input2[:, 1, ...]
input2_ordered[:, 1, ...] = input2[:, 0, ...]
output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 1]), mode=mode,
                                          padding_mode=zero_boundary)


# now test this for 1D
tst_phi = input2[:,:,:,0]
tst_phi_rs = tst_phi.reshape(list(tst_phi.size()) + [1])

tst_input1 = input1[:,:,:,0]
tst_input1_rs = tst_input1.reshape(list(tst_input1.size())+[1])

tst_phi_rs_ordered = torch.zeros_like(tst_phi_rs)
#tst_phi_rs_ordered[:, 0, ...] = tst_phi_rs[:, 1, ...] # keep this at zero
tst_phi_rs_ordered[:, 1, ...] = tst_phi_rs[:, 0, ...]

tst_output = torch.nn.functional.grid_sample(tst_input1_rs, tst_phi_rs_ordered.permute([0,2,3,1]), mode=mode, padding_mode=zero_boundary)

tst_output_orig_sz = tst_output[:,:,:,0]

plt.plot(tst_phi[0,0,...].detach().cpu().numpy())
plt.title('phi')
plt.show()


plt.plot(tst_input1[0,0,...].detach().cpu().numpy())
plt.title('input1')
plt.show()

plt.plot(tst_output_orig_sz[0,0,...].detach().cpu().numpy())
plt.title('output')
plt.show()

print('Hello')