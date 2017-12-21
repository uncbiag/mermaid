from data_utils import *
from os.path import join
import torch
from data_loader import  *
from data_pool import *



class DataManager(object):
    def __init__(self, task_name, dataset_name, sched=''):
        self.task_name = task_name
        self.full_task_name = None
        self.dataset_name = dataset_name
        self.sched = sched
        self.data_path = None
        self.output_path= None
        self.task_root_path = None
        self.label_path = None
        self.full_comb = False     #e.g [1,2,3,4] True:[1,2],[2,3],[3,4],[1,3],[1,4],[2,4]  False: [1,2],[2,3],[3,4]
        self.divided_ratio = [0.7,0.1,0.2]
        self.slicing = -1
        self.axis = -1
        self.dataset = None
        self.task_path =None


    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_output_path(self,output_path):
        self.output_path = output_path

    def set_label_path(self, label_path):
        self.label_path = label_path

    def set_slicing(self, slicing, axis):
        self.slicing = slicing
        self.axis = axis

    def set_full_comb(self, full_comb):
        self.full_comb = full_comb

    def set_divided_ratio(self,divided_ratio):
        self.divided_ratio = divided_ratio

    def set_full_task_name(self, full_task_name):
        self.full_task_name = full_task_name

    def get_full_task_name(self):
        return os.path.split(self.task_root_path)[1]



    def generate_saving_path(self):
        slicing_info = '_slicing_{}_axis_{}'.format(self.slicing, self.axis) if self.slicing>0 else ''
        comb_info = '_full_comb' if self.full_comb else ''
        full_task_name = self.task_name+'_'+self.sched+slicing_info+comb_info
        self.set_full_task_name(full_task_name)
        self.task_root_path = os.path.join(self.output_path,full_task_name)

    def generate_task_path(self):
        self.task_path = {x:os.path.join(self.task_root_path,x) for x in ['train','val', 'test']}
        return self.task_path

    def get_task_root_path(self):
        return self.task_root_path

    def manual_set_task_path(self, task_path):
        self.task_path = {x:os.path.join(task_path,x) for x in ['train','val', 'test']}
        return self.task_path



    def init_dataset(self):
        if self.dataset_name =='oasis':
            self.dataset = OasisDataSet(name=self.task_name, sched=self.sched, full_comb= self.full_comb)
        elif self.dataset_name == 'lpba':
            self.dataset =  LPBADataSet(name=self.task_name, full_comb=self.full_comb)
            self.dataset.set_slicing(self.slicing, self.axis)
            self.dataset.set_label_path(self.label_path)
        elif self.dataset_name == 'ibsr':
            self.dataset = IBSRDataSet(name=self.task_name, full_comb= self.full_comb)
            self.dataset.set_slicing(self.slicing, self.axis)
            self.dataset.set_label_path(self.label_path)
        elif self.dataset_name =='cmuc':
            self.dataset = CUMCDataSet(name=self.task_name,full_comb= self.full_comb)
            self.dataset.set_slicing(self.slicing, self.axis)
            self.dataset.set_label_path(self.label_path)
        self.dataset.set_data_path(self.data_path)
        self.dataset.set_output_path(self.task_root_path)
        self.dataset.set_divided_ratio(self.divided_ratio)



    def prepare_data(self):
        self.dataset.prepare_data()


    def data_loaders(self, batch_size=20):
        composed = transforms.Compose([ToTensor()])
        sess_sel = self.task_path
        transformed_dataset = {x: RegistrationDataset(data_path=sess_sel[x], transform=composed) for x in sess_sel}
        dataloaders = {x: torch.utils.data.DataLoader(transformed_dataset[x], batch_size=batch_size,
                                                shuffle=False, num_workers=4) for x in sess_sel}
        dataloaders['data_size'] = {x: len(dataloaders[x]) for x in ['train', 'val','test']}
        dataloaders['info'] = {x: transformed_dataset[x].pair_name_list for x in ['train', 'val','test']}
        print('dataloader is ready')


        return dataloaders




if __name__ == "__main__":

    prepare_data = False

    dataset_name = 'lpba'
    task_name = 'lpba'
    task_path = '/playpen/zyshen/data/lpba__slicing90'

    if prepare_data:
        data_path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
        label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
        dataset_name = 'lpba'
        task_name = 'lpba'
        full_comb = False
        output_path = '/playpen/zyshen/data/'
        divided_ratio = (0.6, 0.2, 0.2)
        slicing = 90
        axis = 1

        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name)
        data_manager.set_data_path(data_path)
        data_manager.set_output_path(output_path)
        data_manager.set_label_path(label_path)
        data_manager.set_full_comb(full_comb)
        data_manager.set_slicing(slicing, axis)
        data_manager.set_divided_ratio(divided_ratio)
        data_manager.generate_saving_path()
        data_manager.generate_task_path()

        data_manager.init_dataset()
        data_manager.prepare_data()

    else:
        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name)
        data_manager.manual_set_task_path(task_path)


    dataloaders = data_manager.data_loaders(batch_size=20)



