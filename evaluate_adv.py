import scipy.io
import torch
import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Evaluating')
parser.add_argument('--loss_type', default='soft', type=str)
parser.add_argument('--name', default='resnet_50', type=str, help='output model name')
parser.add_argument('--attack', default='I-FGSM', type=str, choices=['FGSM','I-FGSM','MI-FGSM'])
parser.add_argument('--epsilon', default=5, type=int)

opt = parser.parse_args()

assert opt.loss_type in ['soft', 'triplet_loss']
assert opt.name in ['resnet_50', 'resnext_50', 'densenet_121']

def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

####################### original ##############################################
FEA = scipy.io.loadmat(os.path.join('Model/pytorch_fea_adv_(%s)_(epsilon%s).mat' % (opt.attack, opt.epsilon)))
results = []

query_feature = FEA['query_f']
query_cam = FEA['query_cam'][0]
query_label = FEA['query_label'][0]
gallery_cam = FEA['gallery_cam'][0]
gallery_label = FEA['gallery_label'][0]
for ii in range(2):
    if ii == 0:
        tag = 'clean'
        gallery_feature = FEA['gallery_f']
    elif ii == 1:
        tag = 'adv'
        gallery_feature = FEA['gallery_f_adv']

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
      # print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    mAP = ap / len(query_label)
    print('clean: top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))

    results.append({
        'tag': tag,
        'rank-1': CMC[0].item(),
        'rank-5': CMC[4].item(),
        'rank-10': CMC[9].item(),
        'mAP': mAP})

with open('Model/results_(%s)_(epsilon%s).json' % (opt.attack, opt.epsilon), 'w') as fp:
     json.dump(results, fp, indent=1)
