# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import scipy.io
from model import ft_net, ft_net_dense
from resnext import resnext50
import numpy as np
import json

######################################################################
# Options
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--name', default='resnet_50', type=str, help='the model used to extract feature')
parser.add_argument('--loss_type', default='soft', type=str)
parser.add_argument('--gpu_ids', default='1', type=str)
parser.add_argument('--attack', default='I-FGSM', type=str, choices=['FGSM','I-FGSM','MI-FGSM'])
parser.add_argument('--epsilon', default=5, type=int)

parser.add_argument('--test_dir', default='./Market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--adv', action='store_true')

opt = parser.parse_args()

torch.cuda.set_device( int(opt.gpu_ids) )

assert opt.loss_type in ['soft', 'triplet_loss']

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

if opt.adv:
    gallery_str = 'gallery_adv_(%s)_(%s)_(%s)_(epsilon%s)'%(opt.name,opt.loss_type,opt.attack,opt.epsilon)
else:
    gallery_str = 'gallery'

image_datasets = {x: datasets.ImageFolder( os.path.join(opt.test_dir, x), data_transforms) for x in [gallery_str,'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=4) for x in [gallery_str,'query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################

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

def load_network(network):
    save_path = 'Model/%s_%s.pth' % (opt.loss_type, opt.name)
    network.load_state_dict(torch.load(save_path))
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders, flip=False):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data

        img = torch.nn.functional.interpolate(img, size=(256, 128), mode='bilinear', align_corners=False)
        img -= torch.FloatTensor([[[0.485]], [[0.456]], [[0.406]]])
        img /= torch.FloatTensor([[[0.229]], [[0.224]], [[0.225]]])

        img = Variable(img.cuda())
        f1 = model(img, False)
        if flip:
            flip_img = fliplr(img)
            f2 = model(flip_img, False)
            ff = f1 + f2
        else:
            ff = f1
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff / fnorm
        ff = ff.data.cpu()
        features = torch.cat((features,ff), 0)
        del ff
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets[gallery_str].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

print('-------test-----------')
if opt.loss_type == 'soft':
    if opt.name == 'resnet_50':
        model_structure = ft_net(751)
    elif opt.name == 'resnext_50':
        model_structure = resnext50(num_classes=751)
    elif opt.name == 'densenet_121':
        model_structure = ft_net_dense(751)
model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature_2048= extract_feature(model, dataloaders[gallery_str])
query_feature_2048 = extract_feature(model, dataloaders['query'])

FEA = {'gallery_f': gallery_feature_2048.numpy(), 'gallery_label': gallery_label,
            'gallery_cam': gallery_cam, 'query_f': query_feature_2048.numpy(), 'query_label': query_label,
            'query_cam': query_cam}

# Save to Matlab for check
scipy.io.savemat('/tmp/pytorch_fea_from_image.mat', FEA)
FEA = scipy.io.loadmat('/tmp/pytorch_fea_from_image.mat')
os.remove('/tmp/pytorch_fea_from_image.mat')

query_feature = FEA['query_f']
query_cam = FEA['query_cam'][0]
query_label = FEA['query_label'][0]
gallery_feature = FEA['gallery_f']
gallery_cam = FEA['gallery_cam'][0]
gallery_label = FEA['gallery_label'][0]

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
print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))

results = {
     'rank-1': CMC[0].item(),
     'rank-5': CMC[4].item(),
     'rank-10': CMC[9].item(),
     'mAP': mAP}
