'''
用于加载文本和图像数据，划分数据集，并生成相应的数据加载器用于训练和验证模型
'''
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data, pretrained_TextModel, max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len = max_len
        # 使用AutoTokenizer根据预训练的文本模型创建分词器（tokenizer） 
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    # 根据索引idx获取数据集中的文本和标签。使用分词器对文本进行分词，并进行填充（padding）和截断（truncation）等处理，生成模型所需的输入。将标签转换为torch.tensor，并返回输入和标签。
    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)
        return inputs, label


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)
        return img, label


class MultimodalDataset(Dataset):
    def __init__(self, data, pretrained_TextModel, max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len = max_len
        # 使用AutoTokenizer根据预训练的文本模型创建分词器（tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    # 根据索引idx获取数据集中的图像、文本和标签。使用分词器对文本进行分词，并进行填充（padding）和截断（truncation）等处理，生成模型所需的输入。将标签转换为torch.tensor，并返回图像、输入和标签。
    def __getitem__(self, idx):
        img, text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)
        return img, inputs, label


class MultimodalDatasetWithGuid(Dataset):
    def __init__(self, data, pretrained_TextModel, max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len = max_len
        # 使用AutoTokenizer根据预训练的文本模型创建分词器（tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, img, text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)
        return guid, img, inputs, label


class TextDataloader:
    def __init__(self, data, data_folder, pretrained_TextModel, max_len, batch_size, validation_size):
        self.data = data
        self.data_folder = data_folder
        self.pretrained_TextModel = pretrained_TextModel
        self.batch_size = batch_size
        self.max_len = max_len
        self.validation_size = validation_size

    # 读取数据文件，遍历每一行数据，根据文件夹和GUID加载对应的文本文件，读取文本内容，并将文本和标签保存到loaded_data列表中。使用train_test_split将loaded_data划分为训练集和验证集
    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                text_name = os.path.join(self.data_folder, guid + '.txt')

                with open(text_name, 'rb') as text_file:
                    text = text_file.read().strip().decode("utf-8", "ignore")

                loaded_data.append((text, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    # 调用load_data方法加载数据，并创建训练集和验证集的数据集对象（TextDataset），使用`DataLoader`创建训练集和验证集的数据加载器
    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = TextDataset(train_data, self.pretrained_TextModel, self.max_len)
        valid_dataset = TextDataset(valid_data, self.pretrained_TextModel, self.max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader


class ImageDataloader:
    def __init__(self, data, data_folder, transform, batch_size, validation_size):
        self.data = data
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.transform = transform
        self.validation_size = validation_size

    # 读取数据文件，遍历每一行数据，根据文件夹和GUID加载对应的图像文件，对图像进行预处理，并将预处理后的图像和标签保存到loaded_data列表中。使用train_test_split将loaded_data划分为训练集和验证集
    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                img_name = os.path.join(self.data_folder, guid + '.jpg')

                with Image.open(img_name) as img:
                    img = self.transform(img)

                loaded_data.append((img, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    # 调用load_data方法加载数据，并创建训练集和验证集的数据集对象（ImageDataset），使用DataLoader创建训练集和验证集的数据加载器
    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = ImageDataset(train_data)
        valid_dataset = ImageDataset(valid_data)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader


class MultimodalDataloader:
    def __init__(self, data, img_data_folder, text_data_folder, pretrained_TextModel, transform, max_len, batch_size, validation_size):
        self.data = data
        self.img_data_folder = img_data_folder
        self.text_data_folder = text_data_folder
        self.pretrained_TextModel = pretrained_TextModel
        self.batch_size = batch_size
        self.transform = transform
        self.max_len = max_len
        self.validation_size = validation_size

    # 读取数据文件，遍历每一行数据，根据文件夹和GUID加载对应的图像文件和文本文件，读取文本内容，并将图像、文本和标签保存到loaded_data列表中。使用train_test_split将loaded_data划分为训练集和验证集
    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                text_name = os.path.join(self.text_data_folder, guid + '.txt')
                img_name = os.path.join(self.img_data_folder, guid + '.jpg')

                with open(text_name, 'rb') as text_file:
                    text = text_file.read().strip().decode("utf-8", "ignore")

                with Image.open(img_name) as img:
                    img = self.transform(img)

                loaded_data.append((img, text, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    # 调用load_data方法加载数据，并创建训练集和验证集的数据集对象（MultimodalDataset），使用DataLoader创建训练集和验证集的数据加载器
    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = MultimodalDataset(train_data, self.pretrained_TextModel, self.max_len)
        valid_dataset = MultimodalDataset(valid_data, self.pretrained_TextModel, self.max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader
