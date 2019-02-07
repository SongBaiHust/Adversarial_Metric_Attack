# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import os
from PIL import Image

######################################################################
# Options
parser = argparse.ArgumentParser(description='Concat')
parser.add_argument('--clean_dir', default='./Market1501/pytorch/gallery', type=str)
parser.add_argument('--output_dir', default='./Market1501/pytorch/gallery_compare', type=str)
parser.add_argument('--name', default='resnet_50', type=str, help='the model used to extract feature')
parser.add_argument('--loss_type', default='soft', type=str)
parser.add_argument('--attack', default='I-FGSM', type=str, choices=['FGSM','I-FGSM','MI-FGSM'])
parser.add_argument('--epsilon', default=5, type=int)
opt = parser.parse_args()

opt.adv_dir = './Market1501/pytorch/gallery_adv_(%s)_(%s)_(%s)_(epsilon%s)'%(opt.name,opt.loss_type,opt.attack,opt.epsilon)

clean_ids = [x[0].split('/')[-1] for x in os.walk(opt.clean_dir)][1:]
clean_list = {}
for cid in clean_ids:
    clean_list[cid] = [x[2] for x in os.walk(os.path.join(opt.clean_dir, cid))][0]

adv_ids = [x[0].split('/')[-1] for x in os.walk(opt.adv_dir)][1:]
adv_list = {}
for aid in adv_ids:
    adv_list[aid] = [x[2] for x in os.walk(os.path.join(opt.adv_dir, aid))][0]

for cid in clean_ids:
    aid = cid
    assert aid in adv_ids
    for clean_filename in clean_list[cid]:
        adv_filename = os.path.basename(clean_filename)[:-4]+'.png'
        assert adv_filename in adv_list[aid]
        clean_image_c = os.path.join(os.path.join(opt.clean_dir, cid, clean_filename))
        adv_image_c = os.path.join(os.path.join(opt.adv_dir, aid, adv_filename))
        images = map(Image.open, [clean_image_c, adv_image_c])
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        save_dir = os.path.join(opt.output_dir, cid)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        new_im.save(os.path.join(save_dir, adv_filename))