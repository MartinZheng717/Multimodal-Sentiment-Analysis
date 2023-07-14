import random
import sys
import torch
import os
import numpy as np

# 将工作目录切换到当前文件所在的目录
# os.chdir(os.path.dirname(__file__))
# 获得工作区路径
working_path = os.getcwd()

# 获得当前文件的路径
# current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# 获得当前文件的上上层目录
# root_path = os.path.abspath(os.path.join(current_path,"../.."))

root_path = working_path
# image_path = root_path + '\data\raw\img'
# text_path = root_path + '\data\raw\text'
# training_set_path = root_path + '\data\train.txt'
# test_set_path = root_path + '\data\test_without_label.txt'
# pretrained_image_path = root_path + '\code\model\pretrained\resnet50.pth'

image_path = os.path.join(root_path, 'data\\raw\img')
text_path = os.path.join(root_path, 'data\\raw\\text')
training_set_path = os.path.join(root_path, 'data\\train.txt')
testing_set_path = os.path.join(root_path, 'data\\test_without_label.txt')
test_set_path = os.path.join(root_path, 'data\\test_without_label.txt')
pretrained_image_model_path = os.path.join(root_path, 'code\model\pretrained\\resnet\\resnet50.pth')
pretrained_text_model_path = os.path.join(root_path, 'code\model\pretrained\\roberta')
best_weights_path = os.path.join(root_path, 'bestweights\\best_model.pth')

seed = 717
batch_size = 64
epoch = 6
lr = 1e-3
weight_decay = 1e-4
fine_tune = False
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_seed():
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # all gpu
    np.random.seed(seed)
    random.seed(seed)
