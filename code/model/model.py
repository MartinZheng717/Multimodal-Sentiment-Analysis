'''
融合图像模型和文本模型，定义多模态模型
'''
import torch
from torch import nn
import numpy as np
from torchvision.models import resnet50,resnet34,resnet18
from transformers import RobertaModel, RobertaConfig
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TextModel(nn.Module):
    def __init__(self,pretrained_TextModel,num_classes=3):
        super(TextModel, self).__init__()
        self.pretrained_TextModel = pretrained_TextModel
        self.config = RobertaConfig.from_pretrained(self.pretrained_TextModel)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_TextModel)
        # 定义一个线性层作为分类器，将RoBERTa模型的输出特征进行分类
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
    
    # 传入文本数据，通过RoBERTa模型获取文本的特征表示，然后通过分类器进行分类，并返回分类结果。
    def forward(self,text):
        text_outputs = self.text_model(**text)
        pooled_output = text_outputs.pooler_output
        outputs = self.classifier(pooled_output)
        return outputs

class ImageModel(nn.Module):
    # 加载预训练的ResNet-50模型，并根据预训练模型的输出特征数定义一个线性层作为分类器
    def __init__(self,pretrained_ImageModel,num_classes=3):
        super(ImageModel, self).__init__()
        self.pretrained_ImageModel = pretrained_ImageModel
        self.image_model = resnet50()
        self.image_model.load_state_dict(torch.load(self.pretrained_ImageModel))
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, num_classes)
        
    # 传入图像数据，通过ResNet-50模型获取图像的特征表示，然后通过分类器进行分类，并返回分类结果
    def forward(self,image):
        outputs = self.image_model(image)
        return outputs

class SelfAttention(nn.Module):
    # 使用`nn.Linear`定义了三个线性层，用于计算查询（query）、键（key）和值（value）
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    # 传入输入数据，通过线性层计算查询、键和值，然后计算注意力分数，应用softmax函数得到注意力权重，最后通过加权求和得到加权输出。
    def forward(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        attention_scores = torch.matmul(Q, K.transpose(1, 0))
        attention_weights = self.softmax(attention_scores)
        weighted_output = torch.matmul(attention_weights, V)
        return weighted_output


class MultimodalModel(nn.Module):
    # 首先创建文本模型和图像模型。接着加载预训练的图像模型的权重，并将其全连接层替换为一个具有与RoBERTa模型隐藏层大小相同的线性层。根据融合方法选择对应的操作：
    # 如果是"add"，则将文本和图像特征相加；如果是"concat"，则将文本和图像特征在特征维度上拼接；如果是"attention"，则将文本和图像特征在特征维度上拼接，并通过自注意力模块进行融合。最后使用线性层进行分类。
    def __init__(self, pretrained_TextModel,pretrained_ImageModel,fusion_method,num_classes=3):
        super(MultimodalModel, self).__init__()
        self.pretrained_TextModel = pretrained_TextModel
        self.pretrained_ImageModel = pretrained_ImageModel
        self.config = RobertaConfig.from_pretrained(self.pretrained_TextModel)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_TextModel)
        self.image_model_name=pretrained_ImageModel[-12:-4]
        if self.image_model_name=='resnet50':
            self.image_model = resnet50()
        elif self.image_model_name=='resnet34':
            self.image_model = resnet34()
        else:
            self.image_model = resnet18()
        self.image_model.load_state_dict(torch.load(self.pretrained_ImageModel))
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, self.config.hidden_size)
        self.fusion_method=fusion_method
        if self.fusion_method=='attention':
            self.self_attention = SelfAttention(self.config.hidden_size*2)
        if self.fusion_method=='add':
            self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        elif self.fusion_method=='concat':
            self.classifier = nn.Linear(self.config.hidden_size*2, num_classes)

    # 传入文本和图像数据，分别通过文本模型和图像模型获取对应的特征表示。根据融合方法进行融合操作，然后通过分类器进行分类，并返回分类结果
    def forward(self, text, image):
        text_outputs = self.text_model(**text)
        image_outputs = self.image_model(image)
        pooled_output = text_outputs.pooler_output
        if self.fusion_method=='add':
            outputs = pooled_output + image_outputs
        elif self.fusion_method=='concat':
            outputs = torch.cat([pooled_output, image_outputs], dim=1)
        elif self.fusion_method=='attention':
            outputs = torch.cat([pooled_output, image_outputs], dim=1)
            outputs = self.self_attention(outputs)
        outputs = self.classifier(outputs)
        return outputs
