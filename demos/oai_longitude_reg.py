import numpy as np
import os
from glob import glob
from functools import reduce
import pyreg.module_parameters as pars
from pyreg.data_utils import make_dir
import random

#######################################################
"""
######################################### Section 1. Raw Data Organization  ###############################################

data root:  /playpen/zhenlinx/Data/OAI_segmentation/

The images were saved as nifti format at ./Nifti_6sets_rescaled
The list file are images_6sets_left.txt and images_6sets_right.txt. The images file name are ordered by patient IDs but not ordered by time for each patient.
For a file name like 9000099_20050712_SAG_3D_DESS_LEFT_10424405_image.nii.gz
        9000099 is the patient ID, 20050712 is the scan date,
        SAG_3D_DESS is the image modality, 
        LEFT means left knee, 
        and 10424405 is the image id.

Segmentations for images_6sets_right  predicted by UNetx2 were saved at  
/playpen/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038



#######################################  Section 2. Processed Data Organization  #############################################

data root:  /playpen/zyshen/summer/oai_registration/data


The patient id will be saved in patient_id.txt
The modality list will be saved in modality.txt  ( where each line will be organized as following  MRI  mod1  #newline  CT  mod2 ...........)
the patient slices list will be save in folder  "patient_slice"

./patient_slice/  :                     each patient_id is a separate folder
./patient_slice/idxxxxxxx/:             each modality is a separate folder
./patient_slice/idxxxxxxx/mod1/:        each specificity is a separate folder ie. left,  right
./patient_slice/idxxxxxxx/mod1/spec1/   paths of slices will be recorded in "slice.txt", each line links to a slice and is ordered by time
./patient_slice/idxxxxxxx/mod1/spec1/   paths of slice labels will be recorded in "slice_label.txt", each line links to a slice_label and one to one related with "slice_txt"

########################################   Section 3. Code Organization  ####################################################










class Patient:  
                class Patient are initialized from each patient_id folder, so it need  the path of patient_id folder as input
                
                object varaible included: 
                    basic_information:
                        patient_id, modality (set), specificity(set), patient_slices_path_dic ([modality][specificity]: slice_list)  (dict),
                        patient_slices_num_dic (dict)
                        
                    annotation_information:
                        has_label, label_is_complete, patient_slices_label_path_dic (dict), 
                
                function called outside:
                    check_if_taken(modality,specificity,has_label)
                    get_slice_list(modality,specificity)
                    get_label_path_list(modality,specificity)
                    get_slice_num(modality,specificity)
                    
                
                function called inside:
                    __get_slice_num()
                    __init__()
                    
                    
                    
                
                
                                

class Patients: 
                class Patients are initialized from patient_slice folder, so it need the path of patient_slice folder as input
                the  class Patient will be created during the initialization

                object varaible included:
                    patients_id_list (list), modality(set), specificity(set), patients_slices_path_dic, patient_slices_num_dic,
                    patients_modality_dic. patients_specificity_dic,
                    patients_slices_label_path_dic, patients_has_label_list 
                    
                    
                function called outside:
                    get_patients_id_list(modality, specificity, has_label, num_of_patients= -1, len_time_range=(-1,-1), use_random=True)
                    get_patients_slices_path(patients_id_list)
                    get_patients_statistic_distribution(is_modality=False, is_specificity= False, has_label=False)
                    
                function call inside:
                    __filter_patients_id(modality, specificity, has_label, num_of_time_series=-1)
                    __print_patients_id_list(patients_id_list)
                    



class DataPrepare:
                class Dataprepare are specificed to oai_dataset, which will transfer Raw Data Organization into Processed Data Organization
                
                object variable included:
                    raw_data_path, output_data_path, 
                    
                function_call_outside:
                    prepare_data()
                    
                function_call_inside:
                    __factor_file(file_name)
                    __factor_file_list()
                    __build_and_write_in()
                    
                

"""
class Patients(object):
    def __init__(self,full_init=False):
        self.full_init = full_init
        self.root_path = "/playpen/zyshen/summer/oai_registration/data"
        self.patients_id_txt_name = 'patient_id.txt'
        self.patients_info_folder = 'patient_slice'
        self.patients_id_list= []
        self.modality = None
        self.specificity = None
        self.patients = []
        if not full_init:
            self.__init_basic_info()
        else:
            self.__init_full_info()


    def __init_basic_info(self):
        self.__get_patients_id_list_from_txt()
        self.patients_num = len(self.patients_id_list)


    def __init_full_info(self):
        self.__get_patients_id_list_from_txt()
        self.patients_num = len(self.patients_id_list)
        for patient_id in self.patients_id_list:
            patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
            self.patients.append(Patient(patient_info_path))

    def get_that_patient_info(self,patient_id):
        assert patient_id in self.patients_id_list
        patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
        patient = Patient(patient_info_path)
        return patient



    def filter_patients_id_list(self,modality=None, specificity=None, has_label=None, num_of_patients= -1, len_time_range=None, use_random=False):
        index = list(range(self.patients_num))
        num_of_patients = num_of_patients if num_of_patients>0 else self.patients_num
        filtered_patients_list =[]
        if use_random:
            random.shuffle(index)
        count = 0
        for i in index:
            if not self.full_init:
                patient_id = self.patients_id_list[i]
                patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
                patient = Patient(patient_info_path)
            else:
                patient = self.patients[i]
            modality = patient.modality[0] if modality is None else modality
            specificity = patient.specificity[0] if specificity is None else specificity
            if_taken = patient.check_if_taken(modality=modality,specificity=specificity,has_label=has_label,len_time_range=len_time_range)
            if if_taken:
                filtered_patients_list.append(patient)
                count+=1
                if count>= num_of_patients:
                    break
        if len(filtered_patients_list)< num_of_patients:
            print("not enough patients meet the filter requirement. We want {} but got {} patients".format(num_of_patients, len(filtered_patients_list)))
        return filtered_patients_list





    def __get_patients_id_list_from_txt(self):
        """
        get the patient id from the txt i.e patient_id.txt
        :param file_name:
        :return: type list, list of patient id
        """

        txt_path = os.path.join(self.root_path, self.patients_id_txt_name)
        with open(txt_path, 'r') as f:
            content = f.read().splitlines()
            if len(content) > 0:
                infos = [line.split('\t') for line in content]
            self.patients_id_list = [info[0] for info in infos]
            self.patients_has_label_list = [info[1]=='annotation_complete' for info in infos]

    def get_that_patient_slice(patient_id,mod,specificity):
        """
        get the slices of specific patient, the slices will be filtered by 'mod' and 'specificity'
        :param patient_id:  the id of the patient
        :param mod: the modality of the slice
        :param specificity:
        :return: type list, list of the filtered slice
        """




class Patient():
    def __init__(self, path):
        # patient_id, modality(set), specificity(set), patient_slices_path_dic([modality][specificity]: slice_list)
        self.patient_root_path = path
        self.patient_id = -1
        self.modality = None
        self.specificity = None
        self.patient_slices_path_dic = {}
        self.patient_slices_num_dic = {}
        self.has_label = False
        self.label_is_complete = True
        self.patient_slices_label_path_dic = {}
        self.patient_has_label_dic= {}
        self.txt_file_name = 'slice_label.txt'
        self.__init_patient_info()


    def __init_patient_info(self):
        self.patient_id = os.path.split(self.patient_root_path)[1]
        modality_list = os.listdir(self.patient_root_path)
        specificity_list = os.listdir(os.path.join(self.patient_root_path, modality_list[0]))
        self.modality = tuple(modality_list)
        self.specificity = tuple(specificity_list)
        for mod in self.modality:
            for spec in self.specificity:
                if mod not in self.patient_slices_path_dic:
                    self.patient_slices_path_dic[mod]={}
                    self.patient_slices_label_path_dic[mod]={}
                    self.patient_has_label_dic[mod]= {}
                    self.patient_slices_num_dic[mod]={}
                self.patient_slices_path_dic[mod][spec], self.patient_slices_label_path_dic[mod][spec] \
                    = self.__init_path_info(mod, spec)
                self.patient_slices_num_dic[mod][spec] = len(self.patient_slices_path_dic[mod][spec])
                has_complete_spec_label = True
                for label_path in self.patient_slices_label_path_dic[mod][spec]:
                    if label_path !='None':
                        self.has_label = True
                    else:
                        self.label_is_complete= False
                        has_complete_spec_label= False
                self.patient_has_label_dic[mod][spec] = has_complete_spec_label



    def __init_path_info(self,modality, specificity):
        txt_path = os.path.join(self.patient_root_path,modality, specificity,self.txt_file_name)
        paths = []
        with open(txt_path, 'r') as f:
            content = f.read().splitlines()
            if len(content) > 0:
                paths = [line.split('\t') for line in content]
            slices_path_list = [path[0] for path in paths]
            slices_label_path_list = [path[1] for path in paths]
        return slices_path_list,slices_label_path_list

    def check_if_taken(self, modality=None, specificity=None, len_time_range=None, has_label=None):
        modality_met =True if modality is None else modality in self.modality
        specificity_met = True if specificity is None else  specificity in self.specificity
        has_label_met = True if has_label is None else self.label_is_complete == has_label
        if modality not in self.modality or specificity not in self.specificity:
            return False

        len_time_met =True
        if len_time_range is not None:
            cur_len_time = self.patient_slices_num_dic[modality][specificity]
            len_time_met = len_time_range[0]<= cur_len_time and len_time_range[1]>= cur_len_time
        if_taken = modality_met and specificity_met and len_time_met and has_label_met
        return if_taken

    def get_slice_list(self, modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_path_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_path_dic[modality][specificity]
        else:
            raise ValueError("patient{} doesn't has slice in format {} and {}".format(self.patient_id, modality, specificity))


    def get_label_path_list(self,modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_label_path_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_label_path_dic[modality][specificity]
        else:
            raise ValueError("patient{} doesn't has label in format {} and {}".format(self.patient_id, modality, specificity))

    def get_slice_num(self,modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_num_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_num_dic[modality][specificity]
        else:
            raise ValueError("patient{} doesn't has slice in format {} and {}".format(self.patient_id, modality, specificity))






class OAIDataPrepare():
    def __init__(self):
        self.raw_data_path = "/playpen/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled"
        self.raw_label_path = "/playpen/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038"
        self.output_root_path = "/playpen/zyshen/summer/oai_registration/data"
        self.output_data_path = "/playpen/zyshen/summer/oai_registration/data/patient_slice"
        self.raw_file_path_list = []
        self.raw_file_label_path_list= []
        self.patient_info_dic= {}
        self.file_end = '*.nii.gz'

    def prepare_data(self):
        self.get_file_list()
        self.__factor_file_list()
        self.__build_and_write_in()


    def __filter_file(self, path, file_end):
        f_filter =[]
        import fnmatch
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, file_end):
                f_filter.append(os.path.join(root, filename))
        return f_filter


    def get_file_list(self):

        self.raw_file_path_list = self.__filter_file(self.raw_data_path,self.file_end)
        self.raw_file_label_path_list = self.__filter_file(self.raw_label_path,self.file_end)


    def __factor_file(self, f_path):
        """
        For a file name like 9000099_20050712_SAG_3D_DESS_LEFT_10424405_image.nii.gz
        9000099 is the patient ID, 20050712 is the scan date,
        SAG_3D_DESS is the image modality,
        LEFT means left knee,
        and 10424405 is the image id.
        :return:
        """
        file_name = os.path.split(f_path)[-1]
        factor_list = file_name.split('_')
        patient_id = factor_list[0]
        scan_date = int(factor_list[1])
        modality = factor_list[2] + '_' + factor_list[3] + '_'+factor_list[4]
        specificity = factor_list[5]
        f = lambda x,y : x+'_'+y
        file_name = reduce(f,factor_list[:7])
        return {'file_path': f_path,'slice_name': file_name,'patient_id':patient_id, 'scan_date':scan_date, 'modality':modality, 'specificity':specificity,'label_path':'None'}



    def __factor_file_list(self):

        for f_path in self.raw_file_path_list:
            fd = self.__factor_file(f_path)
            if fd['patient_id'] not in self.patient_info_dic:
                self.patient_info_dic[fd['patient_id']] = {}
            if fd['modality'] not in self.patient_info_dic[fd['patient_id']]:
                self.patient_info_dic[fd['patient_id']][fd['modality']] = {}
            if fd['specificity'] not in self.patient_info_dic[fd['patient_id']][fd['modality']]:
                self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']] = {}
            cur_dict = self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']][fd['slice_name']]={}
            cur_dict['file_path'] =fd['file_path']
            cur_dict['slice_name'] =fd['slice_name']
            cur_dict['scan_date'] =fd['scan_date']
            cur_dict['label_path'] = fd['label_path']



        for f_path in self.raw_file_label_path_list:
            fd = self.__factor_file(f_path)
            self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']][fd['slice_name']]['label_path'] = f_path




    def __build_and_write_in(self):
        with open(os.path.join(self.output_root_path,'patient_id.txt'),'w') as fr:
            has_label = True
            for pat_id in self.patient_info_dic:
                for mod in self.patient_info_dic[pat_id]:
                    for spec in self.patient_info_dic[pat_id][mod]:
                        folder_path = os.path.join(self.output_data_path,pat_id,mod,spec)
                        make_dir(folder_path)
                        slices_info_dict = self.patient_info_dic[pat_id][mod][spec]
                        sorted_slice_name_list = self.__sort_by_scan_date(slices_info_dict)
                        with open(os.path.join(folder_path,'slice_label.txt'), 'w') as f:
                            for name in sorted_slice_name_list:
                                f.write(slices_info_dict[name]['file_path'])
                                f.write("\t")
                                f.write(slices_info_dict[name]['label_path'])
                                f.write("\n")
                                has_label = has_label if slices_info_dict[name]['label_path'] !='None' else False
                label_complete_str = 'annotation_complete' if has_label else 'annotation_not_complete'
                fr.write(pat_id +'\t' + label_complete_str +'\n')





    def __sort_by_scan_date(self, info_dict):
        slices_name_list=[]
        slices_date_list= []
        for slice in info_dict:
            slices_name_list.append(info_dict[slice]['slice_name'])
            slices_date_list.append(info_dict[slice]['scan_date'])
        slices_name_np = np.array(slices_name_list)
        slices_date_np = np.array(slices_date_list)
        sorted_index = np.argsort(slices_date_np)
        slices_name_np = slices_name_np[sorted_index]
        return list(slices_name_np)



#
# test = OAIDataPrepare()
# test.prepare_data()

patients = Patients()
patients.filter_patients_id_list(specificity='RIGHT',num_of_patients=20, len_time_range=[2,7], use_random=True)