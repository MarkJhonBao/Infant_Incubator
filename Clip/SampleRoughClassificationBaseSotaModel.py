import os.path
import shutil
import torch

import numpy as np
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
# 用于检测高斯分布数据集中异常值的对象。
from sklearn.covariance import EllipticEnvelope
# 用于异常检测类
from sklearn.svm import OneClassSVM
# 数据集导入
from sklearn.datasets import load_wine

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
# weight_path = r"/home/markjhon/Common/InfantMulti/2023-7-1-C3D-NewDataset-0004.pt"
# model = torch.load(weight_path)
model.eval()

def model_load(path_video):
    vid, _, _ = read_video(path_video, output_format="TCHW")
    vid = vid[:32]  # 可选缩短持续时间
    # Step 1: 初始化模型权重
    # weights = R3D_18_Weights.DEFAULT
    # model = r3d_18(weights=weights)
    # model.eval()
    # Step 2: 初始化模型推理变换
    preprocess = weights.transforms()
    # Step 3: 应用推理预处理转换
    batch = preprocess(vid).unsqueeze(0)
    # Step 4: 使用模型并打印预测的类别
    prediction = model(batch).squeeze(0).softmax(0)
    print(prediction)
    return prediction
    label = prediction.argmax().item()
    score = prediction[label].item()
    category_name = weights.meta["categories"][label]
    print(f"{category_name}: {100 * score}%:{path_video}")
    # return prediction.cpu().detach().numpy()
    return str(category_name)
# model_load(path_video=r'C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\Infant\Both_hands\23_10.10-10.12.mp4')

def FileRead_FeatureReturn(root_dir=r'C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\Infant\Both_hands'):
    feature_total = []
    for item in os.listdir(root_dir):
        feature = model_load(os.path.join(root_dir, item))
        feature_total.append(feature)
    return feature_total

# feature_total = FileRead_FeatureReturn(r'path')
path = r'/home/markjhon/Common/20230701-08/20230708_fixLength/20230708'
des_path = r'/home/markjhon/Common/20230701-08/20230702_fixLength/Class_Dir'


for video_item in os.listdir(path):
    category = model_load(os.path.join(path, video_item))
    os.makedirs(os.path.join(des_path, category), exist_ok=True)
    # try:
    #     shutil.move(os.path.join(path, video_item), os.path.join(des_path, category))
    # except:
    #     os.remove(os.path.join(path, video_item))
print("Process Done!!")
# X1 = np.asarray(feature_total).transpose(1, 0)




