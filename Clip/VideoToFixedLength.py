import os.path
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import spacy
import logging
from collections import Counter
import csv
import pickle
import itertools
from sklearn.metrics import f1_score
import yaml
from moviepy.editor import *

def get_f1(y_pred, y_label):

    f1 = {"action_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[0]))),
                                list(itertools.chain.from_iterable(y_label[0])), average="weighted"),
          "object_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[1]))),
                                list(itertools.chain.from_iterable(y_label[1])), average="weighted"),
          "position_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[2]))),
                                  list(itertools.chain.from_iterable(y_label[2])), average="weighted")}

    return f1


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def numericalize(inputs, vocab=None, tokenize=False):
    # This should be 2 seperate functions
    # Create vocabs for train file
    if vocab == None:
        # check unique tokens
        counter = Counter()
        for i in inputs:
            if tokenize:
                counter.update(tokenizer(i))
            else:
                counter.update([i])

        # Create Vocab
        if tokenize:  # That is we are dealing with sentences.
            vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        else:
            vocab = {}

        vocab.update({j: i + len(vocab) for i, j in enumerate(counter)})

    # Convert tokens to numbers:
    numericalized_inputs = []
    for i in inputs:
        if tokenize:
            # Adding sos and eos tokens before and after tokenized string
            numericalized_inputs.append([vocab["<sos>"]]+[vocab[j] if j in vocab else vocab["<unk>"] for j in
                                         tokenizer(i)]+[vocab["<eos>"]])  # TODO: doing tokenization twice here
        else:
            numericalized_inputs.append(vocab[i])

    return numericalized_inputs, vocab


def collate_fn(batch,device, text_pad_value, audio_pad_value,audio_split_samples):
    """
    We use this function to pad the inputs so that they are of uniform length
    and convert them to tensor

    Note: This function would fail while using Iterable type dataset because
    while calculating max lengths the items in iterator will vanish.
    """
    max_audio_len = 0
    max_text_len = 0

    batch_size = len(batch)

    for audio_clip, transcript, _, _, _ in batch:
        if len(audio_clip) > max_audio_len:
            max_audio_len = len(audio_clip)
        if len(transcript) > max_text_len:
            max_text_len = len(transcript)

    # We have to pad the audio such that the audio length is divisible by audio_split_samples
    max_audio_len = (int(max_audio_len/audio_split_samples)+1)*audio_split_samples

    audio = torch.FloatTensor(batch_size, max_audio_len).fill_(audio_pad_value).to(device)
    text = torch.LongTensor(batch_size, max_text_len).fill_(text_pad_value).to(device)
    action = torch.LongTensor(batch_size).fill_(0).to(device)
    object_ = torch.LongTensor(batch_size).fill_(0).to(device)
    position = torch.LongTensor(batch_size).fill_(0).to(device)

    for i, (audio_clip, transcript, action_taken, object_chosen, position_chosen) in enumerate(batch):
        audio[i][:len(audio_clip)] = torch.tensor(audio_clip.tolist())
        text[i][:len(transcript)] = torch.tensor(transcript)
        action[i] = action_taken
        object_[i] = object_chosen
        position[i] = position_chosen

    return audio, text, action, object_, position


class Dataset:
    def __init__(self, audio, text, action, object_, position, wavs_location):
        self.audio = audio
        self.text = text
        self.action = action
        self.object = object_
        self.position = position
        self.wavs_location = wavs_location

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        _, wav = wavfile.read(os.path.join(self.wavs_location,self.audio[item]))
        return wav, self.text[item], self.action[item], self.object[item], self.position[item]


def load_csv(path, file_name):
    # Loads a csv and returns columns:
    with open(os.path.join(path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        au, t, ac, o, p = [], [], [], [], []
        for row in csv_reader:
            au.append(row[0])
            t.append(row[1])
            ac.append(row[2])
            o.append(row[3])
            p.append(row[4])
    return au, t, ac, o, p


def get_Dataset_and_vocabs(path, train_file_name, valid_file_name, wavs_location):
    train_data = load_csv(path, train_file_name)
    test_data = load_csv(path, valid_file_name)

    vocabs = []  # to store all the vocabs
    # audio location need not to be numercalized
    # audio files will be loaded in Dataset.__getitem__()
    numericalized_train_data = [train_data[0]]  # to store train data after converting string to ints
    numericalized_test_data = [test_data[0]]  # to store test data after converting strings to ints

    for i, (j, k) in enumerate(zip(train_data[1:], test_data[1:])):
        if i == 0:  # We have to only tokenize transcripts which come at 0th position
            a, vocab = numericalize(j, tokenize=True)
            b, _ = numericalize(k, vocab=vocab, tokenize=True)
        else:
            a, vocab = numericalize(j)
            b, _ = numericalize(k, vocab=vocab)
        numericalized_train_data.append(a)
        numericalized_test_data.append(b)
        vocabs.append(vocab)

    train_dataset = Dataset(*numericalized_train_data, wavs_location)
    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    Vocab = {'text_vocab': vocabs[0], 'action_vocab': vocabs[1], 'object_vocab': vocabs[2], 'position_vocab': vocabs[3]}

    logger.info(f"Transcript vocab size = {len(Vocab['text_vocab'])}")
    logger.info(f"Action vocab size = {len(Vocab['action_vocab'])}")
    logger.info(f"Object vocab size = {len(Vocab['object_vocab'])}")
    logger.info(f"Position vocab size = {len(Vocab['position_vocab'])}")


    # dumping vocab
    with open(os.path.join(path, "vocab"), "wb") as f:
        pickle.dump(Vocab, f)

    return train_dataset, valid_dataset, Vocab


def get_Dataset_and_vocabs_for_eval(path, valid_file_name, wavs_location):
    test_data = load_csv(path, valid_file_name)

    with open(os.path.join(path, "vocab"), "rb") as f:
        Vocab = pickle.load(f)

    numericalized_test_data = [test_data[0], numericalize(test_data[1], vocab=Vocab['text_vocab'], tokenize=True)[0],
                               numericalize(test_data[2], vocab=Vocab['action_vocab'])[0],
                               numericalize(test_data[3], vocab=Vocab['object_vocab'])[0],
                               numericalize(test_data[4], vocab=Vocab['position_vocab'])[0]]

    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    return valid_dataset, Vocab


def initialize_weights(m):
    # if hasattr(m, 'weight') and m.weight.dim() > 1:
    #     nn.init.xavier_uniform_(m.weight.data)
    for name, param in m.named_parameters():
        if not isinstance(m, nn.Embedding):
            nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, optim, clip):
    model.train()

    epoch_loss = 0

    # Tracking accuracies
    action_accuracy = []
    object_accuracy = []
    position_accuracy = []

    # for f1
    y_pred = []
    y_true = []

    for i, batch in enumerate(train_iterator):
        # running batch
        train_result = model(*batch)

        optim.zero_grad()

        loss = train_result["loss"]
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optim.step()

        # Statistics
        epoch_loss += loss.item()
        y_pred.append([train_result["predicted_action"].tolist(), train_result["predicted_object"].tolist(),
                       train_result["predicted_location"].tolist()])
        y_true.append([batch[2].tolist(), batch[3].tolist(), batch[4].tolist()])

        action_accuracy.append(sum(train_result["predicted_action"] == batch[2]) / len(batch[2]) * 100)
        object_accuracy.append(sum(train_result["predicted_object"] == batch[3]) / len(batch[2]) * 100)
        position_accuracy.append(sum(train_result["predicted_location"] == batch[4]) / len(batch[2]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_action_accuracy = sum(action_accuracy) / len(action_accuracy)
    epoch_object_accuracy = sum(object_accuracy) / len(object_accuracy)
    epoch_position_accuracy = sum(position_accuracy) / len(position_accuracy)

    return epoch_loss / len(
        train_iterator), (epoch_f1, epoch_action_accuracy, epoch_object_accuracy, epoch_position_accuracy)

def evaluate(model, valid_iterator):
    model.eval()

    epoch_loss = 0

    # Tracking accuracies
    action_accuracy = []
    object_accuracy = []
    position_accuracy = []

    # for f1
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            # running batch
            valid_result = model(*batch)

            loss = valid_result["loss"]

            # Statistics
            epoch_loss += loss.item()

            y_pred.append([valid_result["predicted_action"].tolist(), valid_result["predicted_object"].tolist(),
                           valid_result["predicted_location"].tolist()])
            y_true.append([batch[2].tolist(), batch[3].tolist(), batch[4].tolist()])

            action_accuracy.append(sum(valid_result["predicted_action"] == batch[2]) / len(batch[2]) * 100)
            object_accuracy.append(sum(valid_result["predicted_object"] == batch[3]) / len(batch[2]) * 100)
            position_accuracy.append(sum(valid_result["predicted_location"] == batch[4]) / len(batch[2]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_action_accuracy = sum(action_accuracy) / len(action_accuracy)
    epoch_object_accuracy = sum(object_accuracy) / len(object_accuracy)
    epoch_position_accuracy = sum(position_accuracy) / len(position_accuracy)

    return epoch_loss / len(
        valid_iterator), (epoch_f1, epoch_action_accuracy, epoch_object_accuracy, epoch_position_accuracy)

def add_to_writer(writer,epoch,train_loss,valid_loss,train_stats,valid_stats,config):
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Validation loss", valid_loss, epoch)
    writer.add_scalar("Train Action f1", train_stats[0]['action_f1'], epoch)
    writer.add_scalar("Train Object f1", train_stats[0]['object_f1'], epoch)
    writer.add_scalar("Train Position f1", train_stats[0]['position_f1'], epoch)
    writer.add_scalar("Train action accuracy", train_stats[1], epoch)
    writer.add_scalar("Train object accuracy", train_stats[2], epoch)
    writer.add_scalar("Train location accuracy", train_stats[3], epoch)
    writer.add_scalar("Valid Action f1", valid_stats[0]['action_f1'], epoch)
    writer.add_scalar("Valid Object f1", valid_stats[0]['object_f1'], epoch)
    writer.add_scalar("Valid Position f1", valid_stats[0]['position_f1'], epoch)
    writer.add_scalar("Valid action accuracy", valid_stats[1], epoch)
    writer.add_scalar("Valid object accuracy", valid_stats[2], epoch)
    writer.add_scalar("Valid location accuracy", valid_stats[3], epoch)


    writer.flush()

def wav_read(path, t_start=13, t_end=48):
    from scipy.io import wavfile
    like = wavfile.read(path)
    # 音频结果将返回一个tuple。第一维参数是采样频率，单位为秒；第二维数据是一个ndarray表示歌曲，
    # 如果第二维的ndarray只有一个数据表示单声道，两个数据表示立体声。所以，通过控制第二维数据就能对歌曲进行裁剪。
    # 对like这个元组第二维数据进行裁剪，所以是like[1];第二维数据中是对音乐数据切分。
    # start_s表示你想裁剪音频的起始时间；同理end_s表示你裁剪音频的结束时间。乘44100是因为每秒需要进行44100次采样
    # 这里表示对该音频的13-48秒进行截取
    wavfile.write('test2.wav',44100,like[1][t_start*44100:t_end*44100])

def extract_voice(data_path, des_path):
    # the des_path should inlude the audio file suffix,suah as .WMA
    import moviepy.editor as mp
    my_clip = mp.VideoFileClip(data_path)
    my_clip.audio.write_audiofile(des_path)
    return True

def clip_video(video_path, des_path):
    def isexist(name, path=None):
        '''
        :param name: 需要检测的文件或文件夹名
        :param path: 需要检测的文件或文件夹所在的路径，当path=None时默认使用当前路径检测
        :return: True/False 当检测的文件或文件夹所在的路径下有目标文件或文件夹时返回Ture,
                当检测的文件或文件夹所在的路径下没有有目标文件或文件夹时返回False
        '''
        if path is None:
            path = os.getcwd()
        if os.path.exists(path + '/' + name):
            print("Under the path: " + path + '\n' + name + " is exist")
            return True
        else:
            if (os.path.exists(path)):
                print("Under the path: " + path + '\n' + name + " is not exist")
            else:
                print("This path could not be found: " + path + '\n')
            return False
    if isexist(video_path, path=video_path):
        return False
    else:
        for item in os.listdir(video_path):
            subclips_from_dir(dir_path=os.path.join(os.path.basename, item),
                              new_path=des_path,
                              clip_duration=3)

def subclips_from_dir(dir_path, new_path, clip_duration=1):
    # ignore the error: IOError: [Errno 32] Broken pipe
    # from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
    # signal(SIGPIPE, SIG_IGN)
    # 恢复默认信号行为的方法
    # signal(SIGPIPE, SIG_DFL)
    import glob
    from shutil import copy

    for all_file_name in glob.glob(dir_path):

        file_name_array = all_file_name.split('/')
        print(file_name_array)
        sub_dir = file_name_array[-2]

        # get videoclip iterator
        video = VideoFileClip(all_file_name)

        file_name = file_name_array[-1]
        video_duration = int(video.duration)

        if not os.path.exists(os.path.join(new_path, sub_dir)):
            os.makedirs(os.path.join(new_path, sub_dir))
        if video_duration<3:
            copy(all_file_name, os.path.join(new_path, sub_dir, file_name))
        else:
            for i in range(0, video_duration, clip_duration):
                new_file_name = os.path.join(new_path, sub_dir, str(i)+'_'+file_name)
                sub_video = video.subclip(i, min(i + clip_duration, video_duration))
                sub_video.write_videofile(new_file_name, logger=None)

                if i % 10 == 0:
                    time.sleep(2)
        time.sleep(1)

# def extracr_audio(path, des_path):
#     from moviepy.editor import *
#     audioclip = AudioFileClip(path)
#     audioclip.write_audiofile(path, des_path)

if __name__ == "__main__":
    import torch
    import torchaudio.transforms as T

    import time
    import glob
    path = r'/home/markjhon/Common/20230701-08/20230708'
    new_path = r'/home/markjhon/Common/20230701-08/20230708_fixLength'

    # clip_video(path, new_path)
    # exit(0)

    for item in os.listdir(path):
        # print(glob.glob(os.path.join(path, '*.mp4')))
        subclips_from_dir(os.path.join(path, item), new_path)
        print(os.path.join(path, item))
