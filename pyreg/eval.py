from __future__ import print_function

import numpy as np


def get_multi_metric(pred, gt, metric, eval_label_list = None, rm_bg=False):
    label_list = np.unique(gt).tolist()
    if rm_bg:
        label_list = label_list[1:]
    if eval_label_list is not None:
        for label in eval_label_list:
            assert label in label_list, "label {} is not in label_list".format(label)
        label_list = eval_label_list
    num_label = len(label_list)
    num_batch, num_channel = pred.shape[0], pred.shape[1]
    dice_multi = np.zeros([num_batch,num_channel, num_label])
    iou_multi = np.zeros([num_batch, num_channel, num_label])
    precision_multi = np.zeros([num_batch, num_channel, num_label])
    recall_multi = np.zeros([num_batch, num_channel, num_label])
    for l in range(num_label):
        label_pred = (pred == label_list[l]).astype(np.int32)
        label_gt = (gt == label_list[l]).astype(np.int32)
        for b in range(num_batch):
            for c in range(num_channel):
                metric_res = cal_metric(label_pred[b][c].reshape(-1),  label_gt[b][c].reshape(-1))
                dice_multi[b][c][l] =metric_res['dice']
                iou_multi[b][c][l] = metric_res['dice']
                precision_multi[b][c][l] = metric_res['dice']
                recall_multi[b][c][l] = metric_res['dice']




def cal_metric(label_pred, label_gt):
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
        iou = tp / float(len(union))
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        dice = (tn+fp)/(2*tp+fp+tn)

    res={'iou': iou, 'dice': dice, 'recall': recall, 'precision': precision}

    return res

