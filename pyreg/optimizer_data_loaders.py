from torch.utils.data import Dataset, DataLoader
import torch
import os

from . import fileio as FIO

class PairwiseRegistrationDataset(Dataset):
    """keeps track of pairwise image as well as checkpoints for their parameters"""

    def __init__(self, output_directory, source_image_filenames, target_image_filenames, params):

        self.params = params

        self.output_directory = output_directory
        self.source_image_filenames = source_image_filenames
        self.target_image_filenames = target_image_filenames

        assert( len( source_image_filenames ) == len( target_image_filenames ))

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.params[('data_loader', {}, 'data loader settings')]
        cparams = self.params['data_loader']
        self.intensity_normalize = cparams[('intensity_normalize',True,'intensity normalize images when reading')]
        self.squeeze_image = cparams[('squeeze_image',False,'squeezes image first (e.g, from 1x128x128 to 128x128)')]
        self.normalize_spacing = cparams[('normalize_spacing',True,'normalizes the image spacing')]

    def __len__(self):
        return len(self.source_image_filenames)

    def _get_source_target_image_filenames(self,idx):
        return (self.source_image_filenames[idx],self.target_image_filenames[idx])

    def _get_parameter_filename(self,idx):
        parameter_filename = os.path.join(self.output_directory,'individual_parameter_pair_{:05d}.pt'.format(idx))
        return parameter_filename

    def __getitem__(self,idx):

        # load the actual images
        current_source_filename, current_target_filename = self._get_source_target_image_filenames(idx)

        im_io = FIO.ImageIO()

        ISource,_,_,_ = im_io.read_batch_to_nc_format([current_source_filename],
                                                intensity_normalize=self.intensity_normalize,
                                                squeeze_image=self.squeeze_image,
                                                normalize_spacing=self.normalize_spacing,
                                                silent_mode=True)
        ITarget,_,_,_ = im_io.read_batch_to_nc_format([current_target_filename],
                                                intensity_normalize=self.intensity_normalize,
                                                squeeze_image=self.squeeze_image,
                                                normalize_spacing=self.normalize_spacing,
                                                silent_mode=True)

        # load the parameter file if it already exists
        current_parameter_filename = self._get_parameter_filename(idx)
        # check if there is already a saved file
        if os.path.isfile(current_parameter_filename):
            individual_parameter = torch.load(current_parameter_filename)
        else:
            individual_parameter = None

        sample = dict()
        if individual_parameter is not None:
            sample['individual_parameter'] = individual_parameter
        sample['idx'] = idx
        sample['individual_parameter_filename'] = current_parameter_filename
        sample['ISource'] = ISource[0,...] # as we only loaded a batch-of-one we remove the first dimension
        sample['ITarget'] = ITarget[0,...] # as we only loaded a batch-of-one we remove the first dimension

        return sample