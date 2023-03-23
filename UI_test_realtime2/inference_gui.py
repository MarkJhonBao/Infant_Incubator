import torch
import numpy as np
from network import C3D_model, Resnet
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

torch.backends.cudnn.benchmark = True

def label_conversion(probs_01):
    prob = ','.join(str(i) for i in probs_01)
    prob = prob.replace(' ', ',')
    dic = {}
    keys = []
    fr = open('/\dataloaders\infant_multi_labels_2.txt', 'r')
    for line in fr.readlines():
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dic[k] = v
    fr.close()
    return dic[prob]


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def moderl_load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    # init model
    # model = C3D_model.C3D(num_classes=5)
    # checkpoint = torch.load(
    #     'E:\HDU\Project\InfantMulti\\run\C3D-infant_multi_labels_epoch-199.pth.tar',
    #     map_location=lambda storage, loc: storage)
    # model = C3D_model()
    # checkpoint = torch.load('E:\HDU\Project\InfantMulti\\run\\3DResnet_每50轮80%衰减50轮\models\\3DResnet-ucf101_epoch-1999.pth.tar', map_location='cpu')

    model = Resnet.generate_model(50)
    # checkpoint = torch.load(r'D:\hdu\InfantUI\resnet.pth.tar', map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def infantDetection(model, clip, cap):
    flag, frame = cap.read()
    class_result = ''
    prob_result = []
    while flag:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        retaining, frame = cap.read()
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        print('1',len(clip))
        if len(clip) == 16:
            print('2', len(clip))
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.sigmoid(outputs)
            prob_result = probs.cpu().detach().numpy()
            print(prob_result)
            class_result = np.int64(prob_result > 0.7)
            prob_result = prob_result.reshape(prob_result.shape[0] * prob_result.shape[1], )
            clip.clear()
        return frame, class_result, prob_result


class InfantDetection_Thread(QThread):
    finish_Single = pyqtSignal(list, float)

    def __init__(self, frame, parent=None):
        super(InfantDetection_Thread, self).__init__(parent)
        self.frame = frame

    def run(self):
        # class_result, prob_result = InfantDetection(self.frame)
        class_result, prob_result = 1, 0.5
        self.finish_Single.emit(list(class_result), float(prob_result))


if __name__ == '__main__':
    videoName = ""
    infantDetection(videoName)
