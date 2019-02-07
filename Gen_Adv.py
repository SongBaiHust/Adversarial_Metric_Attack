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
from collections import OrderedDict

# set_trace()

######################################################################
# Options
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--loss_type', default='soft', type=str)
parser.add_argument('--name', default='resnet_50', type=str, help='output model name')
parser.add_argument('--gpu_ids', default='1', type=str)
parser.add_argument('--attack', default='I-FGSM', type=str, choices=['FGSM','I-FGSM','MI-FGSM'])
parser.add_argument('--epsilon', default=5, type=int)
parser.add_argument('--save_img', action='store_true')
parser.add_argument('--save_fea', action='store_true')
parser.add_argument('--test_dir', default='./Market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
opt = parser.parse_args()

torch.cuda.set_device(int(opt.gpu_ids))
assert opt.loss_type in ['soft', 'triplet_loss']
assert opt.name in ['resnet_50', 'resnext_50', 'densenet_121']

def load_network(network):
    save_path = 'Model/%s_%s.pth' % (opt.loss_type, opt.name)
    network.load_state_dict(torch.load(save_path))
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    inv_idx = Variable(inv_idx.cuda(async=True))
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature_img(model, data, flip=False):
    img = data
    # Resize and  Normalize
    img = torch.nn.functional.interpolate(img, size=(256, 128), mode='bilinear', align_corners=False)
    img -= torch.cuda.FloatTensor([[[0.485]], [[0.456]], [[0.406]]])
    img /= torch.cuda.FloatTensor([[[0.229]], [[0.224]], [[0.225]]])

    f1 = model(img, False)
    if flip:
        flip_img = fliplr(img)
        f2 = model(flip_img, False)
        ff = f1 + f2
    else:
        ff = f1
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff / fnorm
    return ff

def criterion(f1s, f2s):
    ret = 0
    loss = torch.nn.MSELoss()
    for f1 in f1s:
        for f2 in f2s:
            ret += loss(f1, f2)
    return ret

class LoadFromFloder(torch.utils.data.Dataset):
    def __init__(self, name_list, transform):
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.name_list[idx])
        image = self.transform(image)
        sample = image

        return sample

def extract_features_gid(model, gid):
    gallery_datasets = LoadFromFloder(gallery_dict[gid], data_transforms)
    gallery_dataloader = torch.utils.data.DataLoader(gallery_datasets, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    g_features = torch.FloatTensor()
    g_images = torch.FloatTensor()

    for g_data in gallery_dataloader:
        g_img = g_data
        g_img = Variable(g_img.data.cuda(async=True))
        # g_img = Variable(g_img.cuda())
        g_feature = extract_feature_img(model, g_img)

        g_feature = g_feature.data.cpu()
        g_features = torch.cat((g_features, g_feature), 0)
        g_images = torch.cat((g_images, g_img.cpu()), 0)

    return g_features, g_images

def FGSM(model, gid, epsilon = 10):
    query_datasets = LoadFromFloder(query_dict[gid], data_transforms)
    query_dataloader = torch.utils.data.DataLoader(query_datasets, batch_size=len(query_dict[gid]), shuffle=False, num_workers=4)

    gallery_datasets = LoadFromFloder(gallery_dict[gid], data_transforms)
    gallery_dataloader = torch.utils.data.DataLoader(gallery_datasets, batch_size=len(gallery_dict[gid]), shuffle=False, num_workers=4)

    for q_data in query_dataloader:
        q_img = q_data
        q_img = Variable(q_img.data.cuda(async=True))
        for g_data in gallery_dataloader:
            g_img = g_data
            x_adv = Variable(g_img.data.cuda(async=True))
            x_adv.requires_grad = True
            # assert q_label[0] == g_label[0]
            q_feature = extract_feature_img(model, q_img)
            g_feature = extract_feature_img(model, x_adv)
            loss = criterion(q_feature, g_feature)
            loss.backward()
            x_grad = torch.sign(x_adv.grad.data)
            x_adv = x_adv + epsilon / 255 * x_grad
            x_adv[x_adv < 0.0] = 0.0
            x_adv[x_adv > 1.0] = 1.0

            g_adv_feature = extract_feature_img(model, x_adv)
            # loss2 = criterion(q_feature, g_adv_feature)
            # print(loss, loss2)

            q_feature = q_feature.data.cpu()
            g_feature = g_feature.data.cpu()
            g_adv_feature = g_adv_feature.data.cpu()

            assert g_feature.shape[0] == len(gallery_dict[str(gid)])
            return q_feature, g_feature, g_adv_feature, x_adv

def MI_FGSM(model, gid, epsilon=10.0, alpha=1.0, momentum=0.0):
    max_iter = int( min(epsilon+4, 1.25*epsilon) )

    query_datasets = LoadFromFloder(query_dict[gid], data_transforms)
    query_dataloader = torch.utils.data.DataLoader(query_datasets, batch_size=len(query_dict[gid]), shuffle=False, num_workers=4)

    gallery_datasets = LoadFromFloder(gallery_dict[gid], data_transforms)
    gallery_dataloader = torch.utils.data.DataLoader(gallery_datasets, batch_size=len(gallery_dict[gid]), shuffle=False, num_workers=4)

    for q_data in query_dataloader:
        q_img = q_data
        q_img = Variable(q_img.data.cuda(async=True))
        q_feature = extract_feature_img(model, q_img)
        q_feature.detach_()
        for g_data in gallery_dataloader:
            g_img = g_data
            x_adv = Variable(g_img.data.cuda(async=True))
            lower_bound = g_img.data.cuda() - epsilon / 255.0
            lower_bound[lower_bound < 0.0] = 0.0
            upper_bound = g_img.data.cuda() + epsilon / 255.0
            upper_bound[upper_bound > 1.0] = 1.0

            x_adv.requires_grad = True
            grad = None
            for _ in range(max_iter):
                g_feature = extract_feature_img(model, x_adv)
                loss = criterion(q_feature, g_feature)
                loss.backward()

                # get normed x_grad
                x_grad = x_adv.grad.data
                norm = torch.mean(torch.abs(x_grad).view((x_grad.shape[0], -1)), dim=1).view((-1, 1, 1, 1))
                norm[norm < 1e-12] = 1e-12
                x_grad /= norm

                grad = x_grad if grad is None else momentum * grad + x_grad
                x_adv = x_adv.data + alpha / 255.0 * torch.sign(grad)

                x_adv = torch.max(x_adv, lower_bound)
                x_adv = torch.min(x_adv, upper_bound)
                x_adv.requires_grad = True

            g_adv_feature = extract_feature_img(model, x_adv)
            g_feature = extract_feature_img(model, g_img.data.cuda())

            q_feature = q_feature.data.cpu()
            g_feature = g_feature.data.cpu()
            g_adv_feature = g_adv_feature.data.cpu()

            assert g_feature.shape[0] == len(gallery_dict[str(gid)])
            return q_feature, g_feature, g_adv_feature, x_adv

def get_id(img_path):
    camera_id = []
    labels = []
    data_list = OrderedDict()
    for path, v in img_path:
        label, filename = path.split('/')[-2:]
        camera = filename.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
        if label in data_list:
            data_list[label].append(path)
        else:
            data_list[label] = [path]
    return camera_id, labels, data_list

#################################################
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

image_datasets = {x: datasets.ImageFolder(os.path.join(opt.test_dir, x), data_transforms) for x in ['gallery', 'query']}

use_gpu = torch.cuda.is_available()
qids = image_datasets['query'].classes
gids = image_datasets['gallery'].classes

query_path = image_datasets['query'].imgs
gallery_path = image_datasets['gallery'].imgs

query_cam, query_label, query_dict = get_id(query_path)
gallery_cam, gallery_label, gallery_dict = get_id(gallery_path)

"""
for key, value in query_dict.items():
    print(key)

for key, value in gallery_dict.items():
    print(key)
"""

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

for p in model.parameters():
    p.requires_grad = False
    # print(p.requires_grad)

gallery_feature_2048_adv = torch.FloatTensor()
gallery_feature_2048 = torch.FloatTensor()
query_feature_2048 = torch.FloatTensor()
toImage = transforms.ToPILImage()

for gid in gids:
    print(gid)
    if gid in ['-1', '0000']:
        # continue

        gf, xs = extract_features_gid(model, gid)
        gallery_feature_2048_adv = torch.cat((gallery_feature_2048_adv, gf), 0)
        gallery_feature_2048 = torch.cat((gallery_feature_2048, gf), 0)

        # save ori image as adv'
        if opt.save_img:
            cur_path = os.path.join(opt.test_dir, 'gallery_adv_(%s)_(%s)_(%s)_(epsilon%s)'%(opt.name,opt.loss_type,opt.attack,opt.epsilon), gid)
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            for x, filename in zip(xs, gallery_dict[gid]):
                image = toImage(x)
                image.save(os.path.join(cur_path, os.path.basename(filename)[:-4]+'.png'))
            del gf
    else:
        assert gid in qids
        if opt.attack == 'FGSM':
            qf, gf, g_adv_f, x_adv = FGSM(model, gid, epsilon = opt.epsilon)
        elif opt.attack == 'I-FGSM':
            qf, gf, g_adv_f, x_adv = MI_FGSM(model, gid, epsilon=opt.epsilon, alpha=1)
        elif opt.attack == 'MI-FGSM':
            qf, gf, g_adv_f, x_adv = MI_FGSM(model, gid, epsilon=opt.epsilon, alpha=1, momentum=1)

        gallery_feature_2048_adv = torch.cat((gallery_feature_2048_adv, g_adv_f), 0)
        gallery_feature_2048 = torch.cat((gallery_feature_2048, gf), 0)
        query_feature_2048 = torch.cat((query_feature_2048, qf), 0)

        # save adv image
        if opt.save_img:
            cur_path = os.path.join(opt.test_dir, 'gallery_adv_(%s)_(%s)_(%s)_(epsilon%s)'%(opt.name,opt.loss_type,opt.attack,opt.epsilon), gid)
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            for x, filename in zip(x_adv, gallery_dict[gid]):
                image = toImage(x.cpu())
                image.save(os.path.join(cur_path, os.path.basename(filename)[:-4] + '.png'))

            del qf, gf, g_adv_f, x_adv

if opt.save_fea:
    FEA_2048 = {'gallery_f': gallery_feature_2048.numpy(), 'gallery_f_adv': gallery_feature_2048_adv.numpy(), 'gallery_label': gallery_label,
            'gallery_cam': gallery_cam, 'query_f': query_feature_2048.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat(os.path.join('Model/pytorch_fea_adv_(%s)_(epsilon%s).mat' % (opt.attack, opt.epsilon)), FEA_2048)
