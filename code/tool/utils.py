'''
辅助函数和操作
'''
import os
from torchvision import transforms

transform=transforms.Compose([  transforms.RandomResizedCrop(224),  # 随机裁剪图像到指定的尺寸(224x224)
                                transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                                transforms.ToTensor(),  # 将图像转换为PyTorch张量
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 对图像进行标准化，将每个通道的数值减去均值(0.485, 0.456, 0.406)并除以标准差(0.229, 0.224, 0.225)。

# 创建实验目录
def create_log_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(f'Log dir : {path}')

def create_bestweight_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(f'Best Weight dir : {path}')

# 模型中可学习参数的总数量
def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    return total_num

# 标签转换
def num_to_tag(num):
    return 'negative' if num==0 else 'neutral' if num==1 else 'positive'

# 预测结果保存
def test_output(save_path, guids, predictions):
    output_file=os.path.join(save_path,"result.txt")
    with open(output_file, 'w') as f:
        f.writelines("guid,tag")
        f.write('\r\n')
        for id, (guid, pred) in enumerate(zip(guids,predictions)):
            f.writelines(str(guid) + "," + num_to_tag(pred))
            f.write('\r\n')
    f.close()