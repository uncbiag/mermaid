from data_utils import *
from os.path import join
import torch
from dataset import  *

def prepare_data(save_path, img_type, path='./data',skip=True, sched='intra'):
    '''
    default:
    path: './data'
    img_type: '*a.mhd'
    skip: True

     '''
    pair_list = list_pairwise(path, img_type, skip,sched)
    img_pair_list, info, img_pair_path_list = load_as_data(pair_list)
    save_to_h5py(save_path, img_pair_list, info, img_pair_path_list)



class DataManager(object):
    def __init__(self, sched='intra'):
        self.train_data_path = '/playpen/zyshen/data/mermaid/data/train'
        self.val_data_path = '/playpen/zyshen/data/mermaid/data/val'
        self.test_data_path = '/playpen/zyshen/data/mermaid/data/test'
        self.raw_data_path=[self.train_data_path,self.val_data_path,self.test_data_path]
        self.sched = sched


        self.skip = True  # e.g [1,2,3,4] True:[1,2],[2,3],[3,4],[1,3],[1,4],[2,4]  False: [1,2],[2,3],[3,4]
        if sched == 'intra': # filter the file
            self.raw_img_type= ['*a.mhd']
        elif sched == 'inter':  # filter the file
            self.raw_img_type = ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
        else:
            raise ValueError('the sampling schedule should be intra or inter')
        self.train_h5_path = '../data/train_data_' + sched + '.h5py'
        self.val_h5_path = '../data/val_data_' + sched + '.h5py'
        self.test_h5_path = '../data/test_data_' + sched + '.h5py'
        self.save_h5_path = [self.train_h5_path,self.val_h5_path,self.test_h5_path]


    def prepare_data(self):
        for idx, path in enumerate(self.raw_data_path):
            prepare_data(self.save_h5_path[idx], self.raw_img_type, self.raw_data_path[idx], sched=self.sched)


    def dataloaders(self, batch_size=20):
        train_data_path = self.train_h5_path
        val_data_path = self.val_h5_path
        test_data_path= self.test_h5_path
        composed = transforms.Compose([ToTensor()])
        sess_sel = {'train': train_data_path, 'val': val_data_path, 'test': test_data_path}
        transformed_dataset = {x: RegistrationDataset(data_dir=sess_sel[x], transform=composed) for x in sess_sel}
        dataloaders = {x: torch.utils.data.DataLoader(transformed_dataset[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4) for x in sess_sel}
        dataloaders['data_size'] = {x: len(dataloaders[x]) for x in ['train', 'val']}
        dataloaders['info'] = {x: transformed_dataset[x].info for x in ['train', 'val']}

        return dataloaders




if __name__ == "__main__":

    path = '/home/hbg/cs_courses/2d_data_code/data/train'
    img_type = ['*a.mhd']
    skip = True
    sched = 'intra'
    save_path = '../data/data_'+ sched +'.h5py'


    prepare_data(save_path, img_type, path, skip, sched)
    dic= read_file(save_path)
    #print(dic['info'])
    print(dic['data'].shape)
    print('finished')

    img_type = ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
    sched='inter'
    skip = False
    save_path = '../data/data_' + sched + '.h5py'
    prepare_data(save_path, img_type, path, skip, sched)
    dic = read_file(save_path)
    #print(dic['info'])
    print(dic['data'].shape)
    print('finished')

