#!/bin/sh
loss_type=soft
name=resnet_50
attack=I-FGSM
epsilon=5

echo ${loss_type}_${name}

# generate adversarial images and save at pytorch/gallery_adv. Directly test the performance using features "pytorch_fea_adv.mat" (both original and adv image features)
if [ 1 -eq 1 ]; then
    python Gan_Adv.py \
      --loss_type=$loss_type \
      --name=$name \
      --attack=$attack \
      --epsilon=$epsilon \
      --save_img \
      --save_fea

    python evaluate_adv.py \
      --loss_type=$loss_type \
      --name=$name \
      --attack=$attack \
      --epsilon=$epsilon
fi

# test adv images in pytorch/gallery_adv. It generates "pytorch_fea_from_image.mat"
if [ 1 -eq 1 ]; then
    python test.py \
      --loss_type=$loss_type \
      --name=$name \
      --attack=$attack \
      --epsilon=$epsilon \
      --adv
fi

# test clean images pytorch/gallery. It generates "pytorch_fea_from_image.mat"
if [ 1 -eq 1 ]; then
    python test.py \
      --loss_type=$loss_type \
      --name=$name
fi

# concat clean and image for visualization
if [ 1 -eq 1 ]; then
    python concatImage.py \
      --loss_type=$loss_type \
      --name=$name \
      --attack=$attack \
      --epsilon=$epsilon
fi
