# This is a simple atlas builder
# To be used (for now) to create training data for the learned smoother

import set_pyreg_paths

# first do the torch imports
import torch
import pyreg.fileio as FIO
import pyreg.simple_interface as SI

import matplotlib.pyplot as plt

def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f

def compute_average_image(images):
    im_io = FIO.ImageIO()
    Iavg = None
    for nr,im_name in enumerate(images):
        Ic,hdrc,spacing,_ = im_io.read_to_nc_format(filename=im_name)
        if nr==0:
            Iavg = torch.from_numpy(Ic)
        else:
            Iavg += torch.from_numpy(Ic)
    Iavg = Iavg/len(images)
    return Iavg,spacing


# first get a few files to build the atlas from

images = get_image_range(0,100)
Iavg,spacing = compute_average_image(images)

plt.imshow(Iavg[0,0,...].numpy(),cmap='gray')
plt.title('Initial average based on ' + str(len(images)) + ' images')
plt.colorbar()
plt.show()

# now register all these image to the average image and while doing so compute a new average image
si = SI.RegisterImagePair()
im_io = FIO.ImageIO()

nr_of_cycles = 3

for c in range(nr_of_cycles):
    print('Starting cycle ' + str(c) + '/' + str(nr_of_cycles))
    for i,im_name in enumerate(images):
        print('Registering image ' + str(i) + '/' + str(len(images)))
        Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)
        si.register_images(Ic, Iavg.numpy(), spacing,
                           model_name='svf_scalar_momentum_map',
                           map_low_res_factor=0.5,
                           nr_of_iterations=50,
                           visualize_step=None)
        wi = si.get_warped_image()

        if c==nr_of_cycles-1: # last time this is run, so let's save the image
            current_filename = './out_data/reg_oasis2d_' + str(i).zfill(4) + '.nrrd'
            im_io.write(current_filename,wi,hdrc)

        if i==0:
            newAvg = wi.data
        else:
            newAvg += wi.data

    Iavg = newAvg/len(images)

    plt.imshow(Iavg[0,0,...].numpy(),cmap='gray')
    plt.title( 'Average ' + str(c+1) + '/' + str(nr_of_cycles) )
    plt.colorbar()
    plt.show()


