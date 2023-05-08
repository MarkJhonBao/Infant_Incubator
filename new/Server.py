import base64
import time
import zmq
import json
import torch
import numpy as np
from network import Resnet_3D
import cv2
import pickle
# from cv2 import VideoCapture


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def show_data(prob_result):
    probs = []
    res = str
    name_list = ['Head', 'LeftHand', 'RightHand', 'LeftLeg', 'RightLeg']
    for i in range(len(prob_result)):
        if prob_result[i] > 0.3:
            prob_result[i] = prob_result[i]
            probs.append(name_list[i])
    if len(probs):
        res = ('+'.join(probs))
    else:
        res = "Normal"
    return res


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    model = Resnet_3D.generate_model(50)
    # checkpoint = torch.load('G:\Project\InfantUI\resnet.pth.tar', map_location='cpu')
    checkpoint = torch.load(r'C:\Users\dev1se\Desktop\InfantGUI\run\3DResnet-ucf101_epoch-1999.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, device


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://10.2.125.73:12345')  # 客户端
    print('zmq send start....')
    model, device = load_model()

    context = zmq.Context()
    socket2 = context.socket(zmq.SUB)
    socket2.connect('tcp://10.2.125.73:1234')
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    print('zmq receive start....')

    # 要改
    # cap = cv2.VideoCapture(r"C:\Users\dev1se\Desktop\InfantGUI\UI2\test02.mp4")
    # assert cap.isOpened(), 'Cannot capture source'
    # retaining = True

    video_data = socket2.recv()
    video_clip = pickle.loads(video_data)

    #  要改
    clip = []
    i=0
    while True:
        # retaining, frame = cap.read()
        # if not retaining and frame is None:
        #     continue
        video_data = socket2.recv()
        video_clip = pickle.loads(video_data)
        for frame in video_clip:
            tmp_ = center_crop(cv2.resize(frame, (171, 128)))
            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
            clip.append(tmp)
            # print(len(clip))
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.sigmoid(outputs)
            prob_result = probs.cpu().detach().numpy()
            prob_result = prob_result.reshape(prob_result.shape[0] * prob_result.shape[1])
            json_prob = prob_result.tolist()
            json_prob = np.round(json_prob, 2)
            res = show_data(prob_result)
            # 全清空
            # clip.pop(0)
            clip = []
            msg = {
                'rtsp': 'rtsp://xxxx',
                'CSeq': 1,
                'result': res,
                'score':
                    {
                        'head': json_prob[0],
                        'left_hand': json_prob[1],
                        'right_hand': json_prob[2],
                        'left_leg': json_prob[3],
                        'right_leg': json_prob[4]
                    },
            }
            msg = json.dumps(msg)
            socket.send_json(msg)
            i += 1
            print("成功传输"+str(i))
