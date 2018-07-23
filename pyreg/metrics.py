from __future__ import print_function

import numpy as np


def get_multi_metric(pred, gt, eval_label_list=None, rm_bg=False, verbose=True):
    """
    implemented iou, dice, recall, precision metrics for each label of each instance in batch

    :param pred:  predicted(warpped) label map Bx....
    :param gt: ground truth label map  Bx....
    :param eval_label_list: manual selected label need to be evaluate
    :param rm_bg: remove the background label, assume the background label is the first label of label_list when using auto detection
    :return: dictonary, has four items:  multi_metric_res, label_avg_res, batch_avg_res, label_list
    multi_metric_res:{iou: Bx #label , dice: Bx#label...} ,
    label_avg_res:{iou: Bx1 , dice: Bx1...} ,
    batch_avg_res{iou: 1x#label , dice: 1x#label...} ,
    label_list: the labels contained by batch
    """

    if not isinstance(pred, (np.ndarray, np.generic)):
        pred = pred.detach().cpu().numpy()
    if not isinstance(gt, (np.ndarray, np.generic)):
        gt = gt.detach().cpu().numpy()
    label_list = np.unique(gt).tolist()
    pred_list = np.unique(pred).tolist()
    union_set = set(label_list).union(set(pred_list))
    if verbose:
        if len(union_set)> len(set(label_list)):
            print("Warning, label {} is in prediction map but not in the ground truth map".format(set(pred_list)-set(label_list)))
    label_list = list(union_set)

    if rm_bg:
        label_list = label_list[1:]
    if eval_label_list is not None:
        for label in eval_label_list:
            assert label in label_list, "label {} is not in label_list".format(label)
        label_list = eval_label_list
    num_label = len(label_list)
    num_batch = pred.shape[0]
    metrics = ['iou', 'dice', 'recall', 'precision']
    multi_metric_res = {metric: np.zeros([num_batch, num_label]) for metric in metrics}
    label_avg_res = {metric: np.zeros([num_batch, 1]) for metric in metrics}
    batch_avg_res = {metric: np.zeros([1, num_label]) for metric in metrics}
    batch_label_avg_res ={metric: np.zeros(1) for metric in metrics}
    label_batch_avg_res ={metric: np.zeros(1) for metric in metrics}
    if num_label==0:
        print("Warning, there is no label in current img")
        return {'multi_metric_res': multi_metric_res, 'label_avg_res': label_avg_res, 'batch_avg_res': batch_avg_res,
            'label_list': label_list, 'batch_label_avg_res':batch_label_avg_res,'label_batch_avg_res':label_batch_avg_res}

    for l in range(num_label):
        label_pred = (pred == label_list[l]).astype(np.int32)
        label_gt = (gt == label_list[l]).astype(np.int32)
        for b in range(num_batch):
            metric_res = cal_metric(label_pred[b].reshape(-1), label_gt[b].reshape(-1))
            for metric in metrics:
                multi_metric_res[metric][b][l] = metric_res[metric]

    for metric in multi_metric_res:
        for s in range(num_batch):
            no_n_index = np.where(multi_metric_res[metric][s] != -1)
            label_avg_res[metric][s] = float(np.mean(multi_metric_res[metric][s][no_n_index]))
        batch_label_avg_res[metric] = float(np.mean(label_avg_res[metric]))

        for l in range(num_label):
            no_n_index = np.where(multi_metric_res[metric][:, l] != -1)
            batch_avg_res[metric][:, l] = float(np.mean(multi_metric_res[metric][:, l][no_n_index]))
        label_batch_avg_res[metric] = float(np.mean(batch_avg_res[metric]))

    return {'multi_metric_res': multi_metric_res, 'label_avg_res': label_avg_res, 'batch_avg_res': batch_avg_res,
            'label_list': label_list, 'batch_label_avg_res':batch_label_avg_res,'label_batch_avg_res':label_batch_avg_res}


def cal_metric(label_pred, label_gt):
    eps = 1e-11
    iou = -1
    recall = -1
    precision = -1
    dice = -1
    gt_loc = set(np.where(label_gt == 1)[0])
    pred_loc = set(np.where(label_pred == 1)[0])
    total_len = len(label_gt)
    # iou
    intersection = set.intersection(gt_loc, pred_loc)
    union = set.union(gt_loc, pred_loc)
    # recall
    len_intersection = len(intersection)
    tp = float(len_intersection)
    tn = float(total_len - len(union))
    fn = float(len(gt_loc) - len_intersection)
    fp = float(len(pred_loc) - len_intersection)

    if len(gt_loc) != 0:
        iou = tp / (float(len(union)) + eps)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        dice = 2 * tp / (2 * tp + fn + fp + eps)
    else:
        if len(pred_loc)>0:
            iou = 0.
            recall = 0.
            precision = 0.
            dice = 0.
        else:
            iou = 1.
            recall = 1.
            precision = 1.
            dice = 1.

    res = {'iou': iou, 'dice': dice, 'recall': recall, 'precision': precision}

    return res

