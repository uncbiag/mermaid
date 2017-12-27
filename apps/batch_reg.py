"""
This implements registration as a command line tool.
Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

# Note: all images have to be in the format BxCxXxYxZ (BxCxX in 1D and BxCxXxY in 2D)
# I.e., in 1D, 2D, 3D we are dealing with 3D, 4D, 5D tensors. B is the batchsize, and C are the channels
# (for example to support color-images or general multi-modal registration scenarios)

from __future__ import print_function
import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
from time import time
import os
import numpy as np

import pyreg.module_parameters as pars
import pyreg.utils as utils
import pyreg.fileio as fileio
from pyreg import data_utils
from pyreg.data_manager import DataManager
from pyreg.metrics import get_multi_metric




def do_registration():

    from pyreg.data_wrapper import AdaptVal
    import pyreg.smoother_factory as SF
    import pyreg.multiscale_optimizer as MO
    from pyreg.config_parser import nr_of_threads



    ############################################     data  setting  #############################################
    par_dataset = pars.ParameterDict()
    par_dataset.load_JSON('../settings/datapro_settings.json')

    dataset_name = par_dataset['datapro']['dataset']['name']
    task_name = par_dataset['datapro']['dataset']['task_name']

    # switch to exist task
    switch_to_exist_task = par_dataset['datapro']['switch']['switch_to_exist_task']
    task_root_path = par_dataset['datapro']['switch']['task_root_path']

    # work on current task
    prepare_data = par_dataset['datapro']['mode']['prepare_data']

    data_path = par_dataset['datapro']['dataset']['data_path']
    label_path = par_dataset['datapro']['dataset']['label_path']
    data_path = data_path if data_path.ext else None
    label_path = label_path if label_path.ext else None


    sched = par_dataset['datapro']['mode']['sched']
    full_comb = par_dataset['datapro']['mode']['all_comb']
    output_path = par_dataset['datapro']['dataset']['output_path']
    divided_ratio = par_dataset['datapro']['mode']['divided_ratio']
    slicing = par_dataset['datapro']['mode']['slicing']
    axis = par_dataset['datapro']['mode']['axis']

    if switch_to_exist_task:
        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name)
        data_manager.manual_set_task_root_path(task_root_path)
    else:

        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name, sched=sched)
        data_manager.set_data_path(data_path)
        data_manager.set_output_path(output_path)
        data_manager.set_label_path(label_path)
        data_manager.set_full_comb(full_comb)
        data_manager.set_slicing(slicing, axis)
        data_manager.set_divided_ratio(divided_ratio)
        data_manager.generate_saving_path()
        data_manager.generate_task_path()
        if prepare_data:
            data_manager.init_dataset()
            data_manager.prepare_data()
        task_root_path = data_manager.get_task_root_path()

    dataloaders = data_manager.data_loaders(batch_size=2)
    data_info = pars.ParameterDict()
    data_info.load_JSON(os.path.join(task_root_path,'info.json'))
    task_full_name = data_manager.get_full_task_name()
    spacing = np.asarray(data_info['info']['spacing'])
    sz = data_info['info']['img_sz']






    ################################  task  setting  ###################################

    par_algconf = pars.ParameterDict()
    par_algconf.load_JSON('../settings/algconf_settings.json')



    sess = ['train']
    params = pars.ParameterDict()

    par_image_smoothing = par_algconf['algconf']['image_smoothing']
    par_model = par_algconf['algconf']['model']
    par_optimizer = par_algconf['algconf']['optimizer']

    use_map = par_model['deformation']['use_map']
    map_low_res_factor = par_model['deformation']['map_low_res_factor']
    model_name = par_model['deformation']['name']

    if use_map:
        model_name = model_name + '_map'
    else:
        model_name = model_name + '_image'

    # general parameters
    params['model']['registration_model'] = par_algconf['algconf']['model']['registration_model']

    torch.set_num_threads(nr_of_threads)
    print('Number of pytorch threads set to: ' + str(torch.get_num_threads()))
    smooth_images = par_image_smoothing['smooth_images']

    par_respro = pars.ParameterDict()
    par_respro.load_JSON('../settings/respro_settings.json')
    expr_name = par_respro['respro']['expr_name']
    visualize = par_respro['respro']['visualize']
    visualize_step = par_respro['respro']['visualize_step']
    save_fig = par_respro['respro']['save_fig']
    save_fig_path = par_respro['respro']['save_fig_path']
    save_excel = par_respro['respro']['save_excel']

    use_multi_scale = par_algconf['algconf']['optimizer']['multi_scale']['use_multiscale']



    if not use_multi_scale:
        # create multi-scale settings for single-scale solution
        multi_scale_scale_factors = [1.0]
        multi_scale_iterations_per_scale = [par_optimizer['single_scale']['nr_of_iterations']]
    else:
        multi_scale_scale_factors = par_optimizer['multi_scale']['scale_factors']
        multi_scale_iterations_per_scale = par_optimizer['multi_scale']['scale_iterations']


    mo = MO.MultiScaleRegistrationOptimizer([1,1]+sz, spacing, use_map, map_low_res_factor, params)

    optimizer_name = par_optimizer['name']

    mo.set_optimizer_by_name(optimizer_name)
    mo.set_light_analysis_on(False)
    mo.set_visualization(visualize)
    mo.set_visualize_step(visualize_step)
    mo.set_expr_name(expr_name)
    mo.set_save_fig(save_fig)
    mo.set_save_fig_path(save_fig_path)
    mo.set_save_fig_num(10)
    mo.set_save_excel(save_excel)
    mo.set_model(model_name)
    mo.set_scale_factors(multi_scale_scale_factors)
    mo.set_number_of_iterations_per_scale(multi_scale_iterations_per_scale)
    mo.set_limit_max_batch(2)
    recorder = mo.init_recorder(expr_name)

    #########################    batch iteration setting   ############################################
    LSource, LTarget = None, None
    sessions = ['train']
    batch_id = 0
    for sess in sessions:
        pair_path_list = dataloaders['info'][sess]
        for data in dataloaders[sess]:
            ISource = AdaptVal(Variable(data['image'][:,:1]))
            ITarget = AdaptVal(Variable(data['image'][:,1:2]))
            pair_path_idx = data['pair_path'].numpy().tolist()
            pair_path = [pair_path_list[idx] for idx in pair_path_idx]
            LSource, LTarget = None, None
            if 'label' in data:
                LSource = AdaptVal(Variable(data['label'][:, :1], volatile=True))
                LTarget = AdaptVal(Variable(data['label'][:, 1:2], volatile=True))

            if smooth_images:
                # smooth both a little bit
                params['image_smoothing'] = par_algconf['algconf']['image_smoothing']
                cparams = params['image_smoothing']
                s = SF.SmootherFactory(sz, spacing).create_smoother(cparams)
                ISource = s.smooth_scalar_field(ISource)
                ITarget = s.smooth_scalar_field(ITarget)

            if params['model']['registration_model']['forward_model']['smoother']['type'] == 'adaptiveNet':
                params['model']['registration_model']['forward_model']['smoother']['input'] = [ISource, ITarget]
                params['model']['registration_model']['forward_model']['smoother']['use_adp'] = True
            else:
                params['model']['registration_model']['forward_model']['smoother']['use_adp'] = False


            mo.set_pair_path(pair_path)
            mo.set_source_image(ISource)
            mo.set_target_image(ITarget)
            mo.set_batch_id(batch_id)
            mo.set_source_label(LSource)
            mo.set_target_label(LTarget)
            mo.set_saving_env()


            # and now do the optimization
            mo.optimize()


            batch_id += 1
            if batch_id > mo.get_limit_max_batch():
                break


        if LSource is not None and save_excel:
            recorder.set_summary_based_env()
            recorder.saving_results(sched='summary')





if __name__ == "__main__":

    do_registration()




