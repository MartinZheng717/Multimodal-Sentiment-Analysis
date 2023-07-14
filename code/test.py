import os
import sys
import time
import torch
from torch import nn
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import logging as transformer_logging
transformer_logging.set_verbosity_error()
from PIL import Image
import numpy as np
import pandas as pd
from tool.dataloader import MultimodalDatasetWithGuid
from tool.dataloader import MultimodalDataloader
from sklearn.metrics import accuracy_score
from model.model import MultimodalModel
from tqdm import tqdm
from tool.utils import *
import wandb
import logging
import argparse
from config import *

# set args
parser = argparse.ArgumentParser("Multimodal-Sentiment-Analysis")
parser.add_argument('--testing_set_path', type=str, default=testing_set_path, help='path of input training data')
parser.add_argument('--image_path', type=str, default=image_path, help='path of input training image data folder')
parser.add_argument('--text_path', type=str, default=text_path, help='path of input training text data folder')
parser.add_argument('--text_model', type=str, default="roberta", help='model used for texts')
parser.add_argument('--image_model', type=str, default="resnet50", help='model used for images')
#parser.add_argument('--pretrained_text_model_path', type=str, default=pretrained_text_model_path, help='path of pretrained text model')
parser.add_argument('--pretrained_text_model_path', type=str, default="code/model/pretrained/sentiment-roberta", help='path of pretrained text model')
parser.add_argument('--pretrained_image_model_path', type=str, default=pretrained_image_model_path, help='path of pretrained image model')
parser.add_argument('--fusion_method', type=str, default="concat", help='the fusion method')
parser.add_argument('--best_weights', type=str, default=best_weights_path, help='training checkpoints')
parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
parser.add_argument('--max_len', type=int, default=64, help='the maximun length of the sequence')
parser.add_argument('--cuda', type=bool, default=False, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default="0", help='gpu device id')
parser.add_argument('--seed', type=int, default=seed, help='random seed')
parser.add_argument('--epoch', type=int, default=epoch, help='num of epochs')
parser.add_argument('--result', type=str, default='result', help='path of the output results')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 用于加载测试数据并返回测试数据的DataLoader对象
def test_data_loader(test_data):
    loaded_data = []
    with open(test_data, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            parts = line.strip().split(',')
            guid = parts[0]
            tag = parts[1]
            text_name = os.path.join(args.text_path, guid + '.txt')
            img_name = os.path.join(args.image_path, guid + '.jpg')
            
            with open(text_name, 'rb') as text_file:
                text = text_file.read().strip().decode("utf-8","ignore")

            with Image.open(img_name) as img:
                img = transform(img)

            loaded_data.append((guid, img, text, tag))

    test_dataset = MultimodalDatasetWithGuid(loaded_data, args.pretrained_text_model_path, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_dataloader

# 将模型设为评估模式，并使用`test_dataloader`逐批获取图像、文本和标签。
# 将文本和图像数据传入模型进行前向传播，得到预测输出。通过`torch.max`函数找到预测概率最高的类别，并将标识符和预测结果保存到Guids和predictions列表中
def predict(model, test_dataloader):
    print("Testing")
    model.eval()
    Guids=[]
    predictions=[]
    with torch.no_grad():
        for batch, (guids, images, texts, _) in enumerate(test_dataloader):
            # images = images.cuda()
            # texts = {name: tensor.squeeze(1).cuda() for name, tensor in texts.items()}
            texts = {name: tensor.squeeze(1) for name, tensor in texts.items()}
            outputs = model(texts, images)
            _, predicted = torch.max(outputs, dim=1)
            Guids.extend(list(guids))
            predictions.extend(predicted.cpu().numpy().tolist())
    return Guids,predictions

# 首先检查是否有可用的GPU设备，然后加载多模态模型并将最优权重加载到模型中，接着加载测试数据并进行预测，得到预测结果。最后调用test_output函数，将预测结果保存到指定路径的输出文件中。
def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        # sys.exit(1)

    print("Loading multi modal model")
    # model = MultimodalModel(pretrained_TextModel=args.pretrained_text,
    #                         pretrained_ImageModel=args.pretrained_image,
    #                         fusion_method=args.fusion_method).cuda()
    model = MultimodalModel(pretrained_TextModel=args.pretrained_text_model_path,
                            pretrained_ImageModel=args.pretrained_image_model_path,
                            fusion_method=args.fusion_method)
    model.load_state_dict(torch.load(args.best_weights))

    print("Loading data")
    test_dataloader=test_data_loader(args.testing_set_path)

    guids, predictions=predict(model,test_dataloader)
    test_output(args.result, guids, predictions)


if __name__ == '__main__':
    main()