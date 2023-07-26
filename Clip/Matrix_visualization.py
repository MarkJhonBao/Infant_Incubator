# coding: utf-8
import os

import torchvision.transforms.functional
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2, csv,codecs
import matplotlib.pyplot as plt

def draw_CAM(model, img_path, save_path,
             transform=None, visual_heatmap=False):
    '''
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图，经过验证111正合适，尽量不要修改
    # 遍历所有样本
    # plt.scatter(data[:, 0], data[:, 1], c=label[:], label="t-SNE")
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], #label[i],
                 '.',
                 color=plt.cm.Set1(label[i] / 20),
                 # color= 'winter_r',
                 fontdict={'weight': 'bold', 'size': 17})
    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    return fig, plt

def main2(data, label):
    from sklearn.manifold import TSNE
    # data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='random', random_state=0)
    reslut = ts.fit_transform(data)
    # 调用函数，绘制图像
    fig, plt = plot_embedding(reslut, label, ' ')
    # 显示图像
    plt.show()
    # plt.savefig(str(time.time())+"test1.png")


if __name__ == "__main__":
    from sklearn import preprocessing
    import time
    from tqdm import tqdm
    import torch
    from C3D_model2_3 import C3D
    from torch.utils.data import DataLoader
    from dataloaders.dataset_1 import VideoDataset
    from network.Module.opts import args_parse
    args = args_parse()

    weight_path = r'C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\pretrained\2023-06-10_08_54_04.886887.pth'
    model = torch.load(weight_path).cuda()
    model_c3d2 = C3D(num_classes=6, pretrained=False).cuda()  # .load_state_dict(model)
    p_dict = model.state_dict()
    s_dict = model_c3d2.state_dict()

    for name in s_dict:
        if 'backbone.' + name not in p_dict.keys():
            continue
        s_dict[name] = p_dict['backbone.' + name]
    model_c3d2.load_state_dict(s_dict)

    def eval(imagePath_dir=r'C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\Scripts\images\test_img'):

        val_dataloader = DataLoader(VideoDataset(args=args,
                                                 dataset=args.dataset,
                                                 split='train',
                                                 clip_len=args.clip_len),
                                    batch_size=args.batch_size, shuffle=True,
                                    # num_workers=args.num_workers,
                                    pin_memory=True)

        trainval_loaders = {'train': val_dataloader}

        cluster_array =[]
        label_array = []
        for inputs, labels in tqdm(trainval_loaders['train']):
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels = labels.detach().cpu().numpy()
            # labels = np.argmax(labels, axis=1)

            labels = np.matmul(labels, np.array([1, 2, 4, 8, 16, 32]))

            infernece_featureMap = model_c3d2(inputs)
            # infernece_featureMap = torch.reshape()
            infernece_featureMap = infernece_featureMap.detach().cpu().numpy()
            cluster_array.append(infernece_featureMap)
            label_array.append(labels)

        cluster_array = np.concatenate(cluster_array, axis=0)
        # print(cluster_array.shape)
        # exit(0)

        # cluster_array = np.reshape(cluster_array, [-1, 1360, 32, 32])
        # cluster_array = np.reshape(cluster_array, [-1, 1360, 1024])
        cluster_array = np.reshape(cluster_array, [-1, 512, 6])
        cluster_array = cluster_array.transpose(0, 2, 1)
        print(cluster_array.shape)
        np.savetxt("./0623-exapleOps3.txt", cluster_array[0])
        exit(0)


        infernece_featureMap = cluster_array
        Adj_matirc = True
        reshape_x = False
        if reshape_x:
            # infernece_featureMap = infernece_featureMap.detach().cpu().numpy()
            infernece_featureMap = (infernece_featureMap - np.min(infernece_featureMap)) / (
                    np.max(infernece_featureMap) - np.min(infernece_featureMap)) * 255
            print(infernece_featureMap.shape)
            # infernece_featureMap = infernece_featureMap.transpose(0, 2, 3, 1)
            for i in range(infernece_featureMap.shape[-1]):
                plt.imshow(infernece_featureMap[0, :, :], cmap="gray")
                plt.show()
                plt.savefig("feature-GCN-1-{}.png".format(time.time()))
            exit(0)
        if Adj_matirc:
            # infernece_featureMap = infernece_featureMap.detach().cpu().numpy()
            infernece_featureMap = (infernece_featureMap - np.min(infernece_featureMap)) / (
                    np.max(infernece_featureMap) - np.min(infernece_featureMap)) * 255
            infernece_featureMap = np.mean(infernece_featureMap, axis=0).astype(np.uint8)
            infernece_featureMap = infernece_featureMap.transpose(1, 2, 0)
            for i in range(infernece_featureMap.shape[-1]//3):
                # plt.imshow(infernece_featureMap[0, :, :])
                # plt.show()
                # plt.savefig("GraphMatrix/{}.png".format(time.time()))
                cv2.imshow("image", infernece_featureMap[:, :, (3*i):(3*i+3)])
                cv2.imwrite("GraphMatrixOps3/"+str(time.time())+".png", infernece_featureMap[:, :, (3*i):(3*i+3)])
                cv2.waitKey(0)
        if not(Adj_matirc) and not(reshape_x):
            print("Done")
            infernece_featureMap = infernece_featureMap.detach().cpu().numpy()
            infernece_featureMap = (infernece_featureMap - np.min(infernece_featureMap)) / (np.max(infernece_featureMap) - np.min(infernece_featureMap)) * 255
            infernece_featureMap = infernece_featureMap.astype(np.uint8).transpose(0, 2, 3, 4, 1)
            infernece_featureMap = np.mean(infernece_featureMap, axis=1)

            for i in range(infernece_featureMap.shape[-1]//3):
                img = infernece_featureMap[0, :, :, (3*i):(3*i+3)]
                cv2.imshow("image", img)
                cv2.imwrite("{}.png".format(time.time()), img)
                cv2.waitKey(0)


        print(infernece_featureMap.shape)
        exit(0)

        # enumerate the infernce_featureMap and resize to (512, 512) each
        new_inference_feature = []
        for i, item_feature in enumerate(infernece_featureMap):

            item_feature = item_feature[:1, :50, :, :, :]
            item_feature = torch.mean(item_feature, dim=2)
            item_feature = torch.mean(item_feature, dim=1)
            item_feature = torch.unsqueeze(item_feature, dim=0)
            # item_feature = torch.mean(item_feature, dim=1)
            # item_feature = torch.mean(item_feature, dim=1)
            item_feature = F.interpolate(item_feature, size=[512, 512], mode='nearest')

            min_a = torch.min(item_feature)
            max_a = torch.max(item_feature)
            item_feature = (item_feature - min_a) / (max_a - min_a)

            new_inference_feature.append(item_feature.cpu())
        return new_inference_feature

    # returned percentage feature map and visualization with opencv
    res = eval()


    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import time
    import numpy as np
    import cv2


    img_path = r"C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\Scripts\images\ground_truth"
    for i,item_img in enumerate(os.listdir(img_path)):

        img = cv2.imread(os.path.join(img_path, item_img))
        img = cv2.resize(img, (512, 512))
        # adaptive gaussian filter
        # adaptive = h5py.File(r'C:\Users\MSI-NB\Desktop\ComputerEngine\InfantMulti\Scripts\images\ground_truth\26_17.13.mp4_20230611_214018331.jpg'.replace('jpg', 'h5'), 'r')
        # adaptive = np.asarray(adaptive['density'])
        adaptive = res[3].detach().numpy()

        adaptive = adaptive.squeeze(axis=0)
        adaptive = np.concatenate((adaptive, adaptive, adaptive), axis=0).transpose((1, 2, 0))

        heatmap = adaptive / np.max(adaptive)
        heatmap = np.uint8(255 * heatmap)[:, :, :1]

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img1 = heatmap * 0.9 + img

        fixed = res[4].detach().numpy()
        fixed = fixed.squeeze(axis=0)
        fixed = np.concatenate((fixed, fixed, fixed), axis=0).transpose((1, 2, 0))
        heatmap = fixed / np.max(fixed)
        heatmap = np.uint8(255 * heatmap)[:, :, :1]

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img2 = heatmap * 0.9 + img
        imgs = np.hstack([img, superimposed_img1, superimposed_img2])

        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        cv2.imwrite(now+'SingleChannelorTemporal{}.jpg'.format(i), imgs)

        time.sleep(2)

    exit(0)
    import torch
    from torch import nn
    from PIL import Image
    from torchvision import transforms,datasets
    import cv2

    img_path = r'C:\Users\MSI-NB\Desktop\Action_Detection\image_example\AI_Images\1(1).jpg'
    # （1）此处为使用PIL进行测试的代码
    transform_valid = transforms.Compose([
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor()])
    img = Image.open(img_path)                  # 3 channels
    img_ = transform_valid(img).unsqueeze(0)    # 拓展维度
    img_ = img_[:, :1, :, :].expand(1, 128, 256, 256)

    conv = torch.nn.Conv2d()
    res = conv(img_).cpu().detach()
    res = res[0, 7, :, :].numpy()

    data_write_csv("channel-0-7.csv", res)
    # print(res[0, :, :, :].shape)


    def random_num(size, end):
        import random
        range_ls = [i for i in range(end)]
        num_ls = []
        for i in range(size):
            num = random.choice(range_ls)
            range_ls.remove(num)
            num_ls.append(num)
        return num_ls

    channel_num = random_num(25, res.shape[0])
    plt.figure(figsize=(100, 100))
    ax = plt.plot()
    plt.imshow(res[1, :, :])
    plt.savefig("feature-0.jpg")
    exit()
    plt.figure(figsize=(10, 10))
    for index, channel in enumerate(channel_num):
        ax = plt.subplot(5, 5, index + 1, )
        plt.imshow(res[channel, :, :])  # 灰度图参数cmap="gray"
    plt.savefig("feature.jpg", dpi=300)


