from __future__ import print_function
from builtins import object
#import progressbar as pb

from torch.utils.data import Dataset

from .data_utils import *


class BaseDataSet(object):

    def __init__(self, name, dataset_type, file_type_list, sched=None):
        """

        :param name: name of data set
        :param dataset_type: ''mixed' like oasis including inter and  intra person  or 'custom' like LPBA40, only includes inter person
        :param file_type_list: the file types to be filtered, like [*1_a.bmp, *2_a.bmp]
        :param data_path: path of the dataset
        """
        self.name = name
        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.pair_name_list = []
        self.pair_path_list = []
        self.file_type_list = file_type_list
        self.save_format = 'h5py'
        """currently only support h5py"""
        self.sched = sched
        """inter or intra, for inter-personal or intra-personal registration"""
        self.dataset_type = dataset_type
        """custom or mixed"""
        self.normalize_sched = 'tp'
        """ settings for normalization, currently not used"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def generate_pair_list(self):
        pass


    def set_data_path(self, path):
        self.data_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)

    def set_normalize_sched(self,sched):
        self.normalize_sched = sched

    def set_divided_ratio(self,ratio):
        self.divided_ratio = ratio

    def get_file_num(self):
        return len(self.pair_path_list)

    def get_pair_name_list(self):
        return self.pair_name_list

    def read_file(self, file_path, is_label=False):
        """
        currently, default using file_io, reading medical format
        :param file_path:
        :param  is_label: the file_path is label_file
        :return:
        """
        # img, info = read_itk_img(file_path)
        img, info = file_io_read_img(file_path, is_label=is_label)
        return img, info

    def extract_pair_info(self, info1, info2):
        return info1

    def save_shared_info(self,info):
        save_sz_sp_to_json(info, self.output_path)

    def save_pair_to_file(self):
        pass


    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        print("starting preapare data..........")
        print("the output file path is: {}".format(self.output_path))
        self.pair_path_list = self.generate_pair_list()
        print("the total num of pair is {}".format(self.get_file_num()))
        self.save_pair_to_file()
        print("data preprocessing finished")



class UnlabeledDataSet(BaseDataSet):
    """
    unlabeled dataset
    """
    def __init__(self, name, dataset_type, file_type_list, sched=None):
        BaseDataSet.__init__(self,name, dataset_type, file_type_list,sched)

    def save_pair_to_file(self):
        """
        save the file into h5py
        :param pair_path_list: N*2  [[full_path_img1, full_path_img2],[full_path_img2, full_path_img3]
        :param pair_name_list: N*1 for 'mix': [folderName1_sliceName1_folderName2_sliceName2, .....]  for custom: [sliceName1_sliceName2, .....]
        :param ratio:  divide dataset into training val and test, based on ratio, e.g [0.7, 0.1, 0.2]
        :param saving_path_list: N*1 list of path for output files e.g [ouput_path/train/sliceName1_sliceName2.h5py,.........]
        :param info: dic including pair name information
        :param normalized_sched: normalized the image
        """
        random.shuffle(self.pair_path_list)
        self.pair_name_list = generate_pair_name(self.pair_path_list, sched=self.dataset_type)
        saving_path_list = divide_data_set(self.output_path, self.pair_name_list, self.divided_ratio)
        img_size = ()
        info = None
        #pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(self.pair_path_list)).start()
        for i, pair in enumerate(self.pair_path_list):
            img1, info1 = self.read_file(pair[0])
            img2, info2 = self.read_file(pair[1])
            if i == 0:
                img_size = img1.shape
                check_same_size(img2, img_size)
            else:
                check_same_size(img1, img_size)
                check_same_size(img2, img_size)
                # Normalized has been done in fileio, though additonal normalization can be done here
                # normalize_img(img1, self.normalize_sched)
                # normalize_img(img2, self.normalize_sched)
            img_pair = np.asarray([(img1, img2)])
            info = self.extract_pair_info(info1, info2)
            save_to_h5py(saving_path_list[i], img_pair, info, [self.pair_name_list[i]], verbose=False)
            pbar.update(i+1)
        pbar.finish()
        self.save_shared_info(info)




class LabeledDataSet(BaseDataSet):
    """
    labeled dataset
    """
    def __init__(self, name, dataset_type, file_type_list, sched=None):
        BaseDataSet.__init__(self, name, dataset_type, file_type_list,sched)
        self.label_path = None
        self.pair_label_path_list=[]


    def set_label_path(self, path):
        self.label_path = path



    def save_pair_to_file(self):
        """
        save the file into h5py
        :param pair_path_list: N*2  [[full_path_img1, full_path_img2],[full_path_img2, full_path_img3]
        :param pair_name_list: N*1 for 'mix': [folderName1_sliceName1_folderName2_sliceName2, .....]  for custom: [sliceName1_sliceName2, .....]
        :param ratio:  divide dataset into training val and test, based on ratio, e.g [0.7, 0.1, 0.2]
        :param saving_path_list: N*1 list of path for output files e.g [ouput_path/train/sliceName1_sliceName2.h5py,.........]
        :param info: dic including pair information
        :param normalized_sched: normalized the image
        """
        random.shuffle(self.pair_path_list)
        self.pair_label_path_list = find_corr_map(self.pair_path_list, self.label_path)
        self.pair_name_list = generate_pair_name(self.pair_path_list, sched=self.dataset_type)
        saving_path_list = divide_data_set(self.output_path, self.pair_name_list, self.divided_ratio)
        img_size = ()
        info = None
        #pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(self.pair_path_list)).start()
        for i, pair in enumerate(self.pair_path_list):
            img1, info1 = self.read_file(pair[0])
            img2, info2 = self.read_file(pair[1])
            label1, linfo1 = self.read_file(self.pair_label_path_list[i][0], is_label=True)
            label2, linfo2 = self.read_file(self.pair_label_path_list[i][1], is_label=True)
            if i == 0:
                img_size = img1.shape
                check_same_size(img2, img_size)
                check_same_size(label1, img_size)
                check_same_size(label2, img_size)
            else:
                check_same_size(img1, img_size)
                check_same_size(img2, img_size)
                check_same_size(label1, img_size)
                check_same_size(label2, img_size)
                # Normalized has been done in fileio, though additonal normalization can be done here
                # normalize_img(img1, self.normalize_sched)
                # normalize_img(img2, self.normalize_sched)
            img_pair = np.asarray([(img1, img2)])
            label_pair = np.asarray([(label1,label2)])
            info = self.extract_pair_info(info1, info2)
            save_to_h5py(saving_path_list[i], img_pair, info, [self.pair_name_list[i]], label_pair, verbose=False)
            pbar.update(i + 1)
        pbar.finish()
        self.save_shared_info(info)





class CustomDataSet(BaseDataSet):
    """
    dataset format that orgnized as data_path/slice1, slic2, slice3 .......
    """
    def __init__(self,name, dataset_type, file_type_list,full_comb=False):
        BaseDataSet.__init__(self, name, dataset_type, file_type_list)
        self.full_comb = full_comb


    def generate_pair_list(self, sched=None):
        """
        :param sched:
        :return:
        """
        pair_path_list = inter_pair(self.data_path, self.file_type_list, self.full_comb, mirrored=True)
        return pair_path_list


class VolumetricDataSet(BaseDataSet):
    """
    3D dataset
    """
    def __init__(self,name, dataset_type, file_type_list):
        BaseDataSet.__init__(self, name, dataset_type, file_type_list)
        self.slicing = -1
        self.axis = -1

    def set_slicing(self, slicing, axis):
        if slicing >0 and axis>0:
            print("slcing is set on , the slice of {} th dimension would be sliced ".format(slicing))
        self.slicing = slicing
        self.axis = axis


    def read_file(self, file_path, is_label=False, verbose=False):
        """

        :param file_path:the path of the file
        :param is_label:the file is label file
        :param verbose:
        :return:
        """
        if self.slicing != -1:
            if verbose:
                print("slicing file: {}".format(file_path))
            img, info = file_io_read_img_slice(file_path, self.slicing, self.axis, is_label=is_label)
        else:
            img, info= BaseDataSet.read_file(self,file_path,is_label=is_label)
        return img, info




class MixedDataSet(BaseDataSet):
    """
     include inter-personal and intra-personal data, which is orgnized as oasis2d, root/patient1_folder/slice1,slice2...
    """
    def __init__(self, name, dataset_type, file_type_list, sched, full_comb=False):
        BaseDataSet.__init__(self, name,dataset_type, file_type_list, sched=sched)
        self.full_comb = full_comb


    def generate_pair_list(self):
        """
         return the list of  paths of the paired image  [N,2]
        :param file_type_list: filter and get the image of certain type
        :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
        :param sched: sched can be inter personal or intra personal
        :return:
        """
        if self.sched == 'intra':
            dic_list = list_dic(self.data_path)
            pair_path_list = intra_pair(self.data_path, dic_list, self.file_type_list, self.full_comb, mirrored=True)
        elif self.sched == 'inter':
            pair_path_list = inter_pair(self.data_path, self.file_type_list, self.full_comb, mirrored=True)
        else:
            raise ValueError("schedule should be 'inter' or 'intra'")

        return pair_path_list



class Oasis2DDataSet(UnlabeledDataSet, MixedDataSet):
    """"
    sched:  'inter': inter_personal,  'intra': intra_personal
    """
    def __init__(self, name, sched, full_comb=False):
        file_type_list = ['*a.mhd'] if sched == 'intra' else ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
        UnlabeledDataSet.__init__(self, name, 'mixed', file_type_list, sched)
        MixedDataSet.__init__(self, name, 'mixed', file_type_list, sched, full_comb)




class LPBADataSet(VolumetricDataSet, LabeledDataSet, CustomDataSet):
    def __init__(self, name, full_comb=True):
        VolumetricDataSet.__init__(self, name, 'custom', ['*.nii'])
        LabeledDataSet.__init__(self, name, 'custom', ['*.nii'])
        CustomDataSet.__init__(self,name, 'custom', ['*.nii'], full_comb)




class IBSRDataSet(VolumetricDataSet, LabeledDataSet, CustomDataSet):
    def __init__(self, name, full_comb=True):
        VolumetricDataSet.__init__(self, name, 'custom', ['*.nii'])
        LabeledDataSet.__init__(self, name, 'custom', ['*.nii'])
        CustomDataSet.__init__(self, name, 'custom', ['*.nii'], full_comb)



class CUMCDataSet(VolumetricDataSet, LabeledDataSet, CustomDataSet):
    def __init__(self, name, full_comb=True):
        VolumetricDataSet.__init__(self, name, 'custom', ['*.nii'])
        LabeledDataSet.__init__(self, name, 'custom', ['*.nii'])
        CustomDataSet.__init__(self, name, 'custom', ['*.nii'], full_comb)




if __name__ == "__main__":
    pass

    # #########################       OASIS TESTING           ###################################3
    #
    # path = '/playpen/zyshen/data/oasis'
    # name = 'oasis'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #oasis  intra testing
    # full_comb = True
    # sched= 'intra'
    #
    # output_path = '/playpen/zyshen/data/'+ name+'_pre_'+ sched
    # oasis = Oasis2DDataSet(name='oasis',sched=sched, full_comb=True)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()


    # ###################################################
    # # oasis inter testing
    # sched='inter'
    # full_comb = False
    # output_path = '/playpen/zyshen/data/' + name + '_pre_' + sched
    # oasis = Oasis2DDataSet(name='oasis', sched=sched, full_comb=full_comb)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()




    # ###########################       LPBA TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # file_type_list = ['*.nii']
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    # sched= 'intra'
    #
    # lpba = LPBADataSet(name=name, full_comb=full_comb)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()


    # ###########################       LPBA Slicing TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre_slicing'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    #
    # lpba = LPBADataSet(name=name, full_comb=full_comb)
    # lpba.set_slicing(90,1)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()





