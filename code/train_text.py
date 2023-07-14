import os
import sys
import time
import torch
from torch import nn
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import logging as transformer_logging
transformer_logging.set_verbosity_error()
import numpy as np
import pandas as pd
from tool.dataloader import TextDataloader
from sklearn.metrics import accuracy_score
from model.model import TextModel
from tqdm import tqdm
from tool.utils import *
import wandb
import logging
import argparse
from config import *

# set args
parser = argparse.ArgumentParser("Text-Sentiment-Analysis")
parser.add_argument('--training_set_path', type=str, default=training_set_path, help='path of input training data')
parser.add_argument('--text_path', type=str, default=text_path, help='path of input training text data folder')
parser.add_argument('--text_model', type=str, default="roberta", help='model used for texts')
#parser.add_argument('--pretrained_text_model_path', type=str, default=pretrained_text_model_path, help='path of pretrained text model')
parser.add_argument('--pretrained_text_model_path', type=str, default="code/model/pretrained/sentiment-roberta", help='path of pretrained text model')
parser.add_argument('--validation_size', type=float, default=0.2, help='the size of the validation set')
parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
parser.add_argument('--max_len', type=int, default=64, help='the maximun length of the sequence')
parser.add_argument('--cuda', type=bool, default=False, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default="0", help='gpu device id')
parser.add_argument('--seed', type=int, default=seed, help='random seed')
parser.add_argument('--epoch', type=int, default=epoch, help='num of epochs')
parser.add_argument('--lr', type=float, default=lr, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=weight_decay, help='weight decay')
parser.add_argument('--log', type=str, default="log", help='path of the experiment log')
parser.add_argument('--best_weights_path', type=str, default="bestweights", help='path of the best model weights')
parser.add_argument('--wandb_id', type=str, default="", help='weight & bias id')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if len(args.wandb_id) != 0:
    wandb.init(
        project="Text Sentiment Analysis using " + args.text_model,
        config=args,
        entity=args.wandb_id
    )

# 将模型设置为评估模式，遍历验证集的数据加载器，获取模型的预测结果并计算模型在验证集上的准确率
def evaluate(model, valid_dataloader):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for batch, (texts, labels) in enumerate(valid_dataloader):
            # texts = {name: tensor.squeeze(1).cuda() for name, tensor in texts.items()}
            texts = {name: tensor.squeeze(1) for name, tensor in texts.items()}
            labels = labels.cuda()
            outputs = model(texts)
            _, predicted = torch.max(outputs, dim=1)
            val_pred.extend(predicted.cpu().numpy().tolist())
            val_true.extend(labels.cpu().numpy().tolist())
    accuracy = accuracy_score(val_true, val_pred)
    return accuracy

# 首先设置模型为训练模式，然后迭代训练集的数据加载器进行训练。计算损失并进行反向传播和参数更新。每训练完成一个批次，输出当前的训练损失。
# 计算并输出当前的平均损失和训练时间。接下来设置模型为评估模式，调用evaluate函数评估模型在验证集上的准确率。如果当前准确率优于历史最佳准确率，保存模型的权重到best_weights_path路径。
def train_and_eval(model, train_dataloader, valid_dataloader, optimizer,scheduler):
    best_acc=0
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(args.epoch):
        start = time.time()
        model.train()
        print("Training epoch {} ".format(i + 1))
        total_loss = 0.0
        for batch, (texts, labels) in enumerate(train_dataloader):
            # texts = {name: tensor.squeeze(1).cuda() for name, tensor in texts.items()}
            texts = {name: tensor.squeeze(1) for name, tensor in texts.items()}
            # labels = labels.cuda()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (batch + 1) % (len(train_dataloader) // 10 if len(train_dataloader) >= 10 else 1) == 0:
                logging.info("epoch:{:03d}  step:{:03d}/{:03d}  loss:{:.3f}".format(i + 1, batch + 1, len(train_dataloader), loss))
            # if (batch + 1) % (len(train_dataloader) // 10 if len(train_dataloader) >= 10 else 1) == 0:
            #     print("epoch:{:03d}  step:{:03d}/{:03d}  loss:{:.3f}".format(i + 1, batch + 1, len(train_dataloader), loss))
        logging.info("epoch:{:03d}  average_loss:{:.3f}  time costed: {:.3f}".format(i + 1, total_loss / (batch + 1), time.time() - start))
        # print("epoch:{:03d}  average_loss:{:.3f}  time costed: {:.3f}".format(i + 1, total_loss / (batch + 1), time.time() - start))
        
        # evaluation and saving the best model
        model.eval()
        acc = evaluate(model, valid_dataloader)
        
        create_bestweight_dir(args.best_weights_path)
        if acc > best_acc:
            best_acc = acc
            # best_weights_path = os.path.join(args.best_weights_path, "best_model_weight.pth")
            # torch.save(model.state_dict(), best_weights_path)
            
        logging.info("current acc is {:.3f}, best acc is {:.3f}".format(acc, best_acc))
        logging.info("time costed:{:.3f}s \n".format(round(time.time() - start, 5)))
        # print("current acc is {:.3f}, best acc is {:.3f}".format(acc, best_acc))
        # print("time costed:{:.3f}s \n".format(round(time.time() - start, 5)))



# 用于组织整个训练过程。创建日志目录和日志文件，设置日志的格式和输出，检查是否有可用的GPU设备，设置随机种子和启用CuDNN加速
# 打印命令行参数，加载数据集，创建数据加载器，加载多模态模型，并创建优化器和学习率调度器，计算模型参数数量，最后调用train_and_eval函数进行训练和评估过程。
def main():
    # create logging directory
    args.log = os.path.join(args.log, 'Train-Text-{}'.format(time.strftime("%H-%M-%S")))
    create_log_dir(args.log)

    # set logging configuration
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # check the cuda devices
    if not torch.cuda.is_available():
        print('no gpu device available')
        # sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %s' % args.gpu)
    # logging.info("args = %s", args)
    print("args = %s", args)

    # load data
    print("Loading data")
    data_loader=TextDataloader(args.training_set_path,
                               args.text_path,
                               args.pretrained_text_model_path,
                               max_len=args.max_len, 
                               batch_size=args.batch_size, 
                               validation_size=args.validation_size)
    
    train_dataloader,valid_dataloader = data_loader()

    # load model
    print(f"Loading text model:{args.text_model}")
    # model = TextModel(pretrained_TextModel=args.pretrained_text).cuda()
    model = TextModel(pretrained_TextModel=args.pretrained_text_model_path)
    # 优化器
    optimizer = AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # 余弦退火学习率调度器
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader),
                                                num_training_steps=args.epoch * len(train_dataloader))

    # count model parameters
    num_of_parameters = count_parameters(model)
    # logging.info("model parameters:{}".format(num_of_parameters))
    print("model parameters:{}".format(num_of_parameters))

    # start training
    # print("start training")
    train_and_eval(model, train_dataloader, valid_dataloader, optimizer, scheduler)

if __name__ == '__main__':
    main()
