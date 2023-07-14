'''
用于将文本数据和图像数据分到两个文件夹
'''
import shutil
import os

os.chdir(os.path.dirname(__file__))

folder_path = "../../data/raw"
text_folder = os.path.join(folder_path, 'text')
img_folder = os.path.join(folder_path, 'img')

if not os.path.exists(text_folder):
    os.makedirs(text_folder)

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# 遍历文件夹中的所有文件
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.txt':
            # 移动.txt文件到text文件夹
            shutil.move(file_path, os.path.join(text_folder, file))
        elif file_extension == '.jpg':
            # 移动.jpg文件到img文件夹
            shutil.move(file_path, os.path.join(img_folder, file))