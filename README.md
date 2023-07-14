## 依赖环境

### 实验环境

- windows 10
- AMD Ryzen 7 4800U with Radeon Graphics 1.80 GHz
- Python 3.7.12
- Visual Studio Code 1.80.1
- Anaconda 23.3.1

### 依赖库

- numpy == 1.21.6
- pandas == 1.2.3
- pillow == 9.4.0
- scikit-learn == 0.22.1
- torchaudio == 0.13.1
- torchvision == 0.14.1
- tqdm == 4.65.0
- transformers == 4.30.2

### 安装

1.创建新的conda环境

```python
conda create -n multimodal python=3.7
conda activate multimodal
```

2.安装依赖库

```python
pip install -r requirements.txt
```



## 代码文件结构

```python
|-- code
	|-- model
		|-- pretrained
			|-- resnet/				# ResNet模型的预训练模型
			|-- sentiment-roberta/	# RoBERTa模型的预训练模型
		|-- model.py				# 融合图像模型和文本模型，定义多模态模型
	|-- tool
		|-- datadivider.py			# 用于将文本数据和图像数据分到两个文件夹
		|-- dataloader.py			# 用于加载文本和图像数据，划分数据集，并生成相应的数据加载器用于训练和验证模型
		|-- utils.py				# 辅助函数和操作
	|-- config.py					# 参数设置
	|-- train_image.py				# 用于消融实验结果，检查多模态模型对于只输入图像数据的表现
	|-- train_text.py				# 用于消融实验结果，检查多模态模型对于只输入文本数据的表现
	|-- train_multi.py				# 训练多模态模型，给出表现评估，并保存最优的模型权重
	|-- test.py						# 加载多模态模型和最优权重，对测试数据进行预测
	
|-- data
    |-- raw
		|-- img/					# 图像数据
		|-- text/					# 文本数据
    train.txt						# 数据的guid和对应的情感标签
    test_without_label.txt			# 数据的guid和空的情感标签

|-- log/							# 实验过程的日志文件
|-- result/							# 实验结果预测文件
|-- bestweights						# 模型最优权重
```



## 代码执行流程

1. 训练多模态模型

   ```python
   python train_multi.py
   ```

2. 对测试数据进行预测

   ```shell
   python test.py
   ```

3. 消融实验结果

   ```shell
   python	train_image.py
   python	train_text.py
   ```

   

## 代码参考

- NAN

## 引用

If you find this code working for you, please cite:

```shell
@project{Multimodal Sentiment Analysis,
 author={Zheng Zhiwei},
 year={2023}
}
```

