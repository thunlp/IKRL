# IKRL

Image-embodied Knowledge Representation Learning (IJCAI-2017)

New: Add dataset


# INTRODUCTION

Image-embodied Knowledge Representation Learning (IKRL)

Image-embodied Knowledge Representation Learning (IJCAI-2017)

Written by Ruobing Xie


# COMPILE 

Just type make in the folder ./


# DATA

We use a new dataset WN9-IMG, with triples extracted from WN18 and images extracted from ImageNet.

There are additional files needed in training, pre-training is optional:

1. image2vec_fc7.txt: image feature vector, pre-trained by AlexNet (fc7 layer)
2. (optional) entity2vec.unif / relation2vec.unif: entity & relation vector, pre-trained by TransE
3. (optional) image_mat.unif: image projection matrix, pre-trained by IKRL (AVG)

For image2vec_fc7.txt, it is a file that contains the image feature vectors of all images learned by AlexNet (or other models). Due to its large size, we do not release our features. However, it is very easy to use other visual features learned by updated visual encoders and images.

Step1. Collect (multiple) images for all entities.

Step2. Learn image feature vectors for all images by AlexNet (or other models). We use the last hidden embedding (fc7 layer) as IKRL's image features.

Please refer to the original paper for more details:

Ruobing Xie, Zhiyuan Liu, Huanbo Luan, Maosong Sun. Image-embodied Knowledge Representation Learning. The 26th International Joint Conference on Artificial Intelligence (IJCAI'17).


# RUN

train: time ./Train_transI -size 50 -margin 4 -method 0

test: ./Test unif


# CITE

If the codes or datasets help you, please cite the following paper:

Ruobing Xie, Zhiyuan Liu, Huanbo Luan, Maosong Sun. Image-embodied Knowledge Representation Learning. The 26th International Joint Conference on Artificial Intelligence (IJCAI'17).
