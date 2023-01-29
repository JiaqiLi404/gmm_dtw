# @Time : 2022/5/27 14:35 
# @Author : Li Jiaqi
# @Description :
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import random
import wave
import os
from scipy.fftpack import fft, fftshift


def generate_data():
    """
    生成随机分布的聚类数据
    """
    data = np.random.randn(500, 8)
    data[30:60, :] += 5
    data[60:90, :] += 10
    terget = np.zeros(500)
    terget[30:60] = 1
    terget[60:90] = 2
    terget = list(terget)
    return data, terget


def delay_data(data, times):
    """
    对数据进行时间延时
    :param data: 数据
    :param times: 插值次数
    """
    for i in range(times):
        index = random.randint(1, len(data) - 1)
        data.insert(index, (data[index - 1] + data[index]) / 2)
        # data.insert(index, data[index])
    return data


def load_voice_data():
    """
    读取音频聚类数据
    """
    files = os.listdir("datasets/voices")
    data_temp = []
    data_train = []
    min_data = -1
    for file in files:
        new_voice = load_voice("datasets/voices/" + file)
        if file in ["datasets/voices/1_01.wav", "datasets/voices/2_01.wav", "datasets/voices/3_01.wav"]:
            plt.plot(new_voice)
            plt.show()
        new_voice = new_voice[np.where(new_voice > 0)]

        new_d = []
        for i in range(len(new_voice) // 80):
            new_d.append(np.sum(new_voice[i * 80:i * 80 + 80]) / 80)

        if file in ["datasets/voices/1_01.wav", "datasets/voices/2_01.wav", "datasets/voices/3_01.wav"]:
            plt.plot(new_d)
            plt.show()
        if min_data == -1:
            min_data = len(new_voice)
        if min_data > len(new_voice):
            min_data = len(new_voice)
        data_temp.append(new_voice)
    target = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    for data in data_temp:
        data_train.append(data[:min_data])
    return np.array(data_train), target


def load_voice(file):
    """
    读取和整波音频文件
    :param file: 音频文件
    """
    f = wave.open(file, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    d = np.frombuffer(strData, dtype=np.int16)
    return d


def load_eye_data(time_delay=False):
    """
    读取由心电数据
    :param time_delay: 是否进行样本延时
    """
    data_ori = arff.loadarff('EEG Eye State.arff')[0]
    data = []
    target = []
    for i in data_ori[:500]:
        if time_delay:
            d = list([float(x) for x in i])
            d = delay_data(d, 10)
        else:
            d = list([float(x) for x in i])
        data.append(d[:-1])
        target.append(int(1 - d[-1]))
    return np.array(data), target


def load_real_data(class_num=3, time_delay=False):
    """
    读取由心电数据生成的真实聚类数据
    :param class_num: 预设类数
    :param time_delay: 是否进行样本延时
    """
    data = np.load("points.npy")
    div = [0]
    target = np.zeros(len(data), dtype=np.int8)
    # 建立多个类
    # for i in range(class_num - 1):
    #     num = -1
    #     while num == -1 or div.count(num) != 0:
    #         num = random.randint(div[-1] + 5, len(data) - (class_num-i-2) * 50)
    #     div.append(num)
    # div.append(len(data))
    div = [0, 130, 260, len(data)]
    # 为每个类加扰动
    last = div[1]
    for index, i in enumerate(div[2:]):
        target[last:i] = target[last - 1] + 1
        data[last:i, 2:] += random.randint((index + 1) * 50, (index + 1) * 50 + 80)
        last = i
    # 数据美观
    data[div[-3]:div[-2], 0] -= 50
    data[div[-2]:div[-1]] += 60
    data[div[-2]:div[-1], 0] -= 160

    if time_delay:
        data = list(data)
        for i in range(len(data)):
            data[i] = delay_data(list(data[i]), 5)
        data = np.array(data)
    return data, list(target)


def show_data_distribution(data, target):
    """
    展示数据的分布情况
    :param data: 样本集
    :param target: 归属类别
    """
    class_num = max(target) + 1
    result = []
    for i in range(int(class_num)):
        result.append([])
    for i in range(len(data)):
        result[int(target[i])].append(data[i])
    for obj_per_class in result:
        x = [i[1] for i in obj_per_class]
        y = [i[2] for i in obj_per_class]
        plt.scatter(x, y)
    plt.title("data original distribution")
    plt.show()


def load_hot_words_data(process=True):
    """
    读取谷歌热点词的词频数据集
    """
    path = 'datasets/hotWords'
    final_datas = []
    words = []
    target = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        with open(file_path, encoding='utf8') as f:
            datas = []
            csv_reader = csv.reader(f)
            next(csv_reader)
            next(csv_reader)
            row = next(csv_reader)
            # 提取热词
            for w in row[1:]:
                words.append(w.split(':')[0])
            # 提取矢量
            for row in csv_reader:
                data = row[1]
                if data == '<1':
                    data = 1
                else:
                    data = int(data)
                datas.append(data)
            final_datas.append(datas[-104:])
            if file.find('news') != -1:
                target.extend([1])
            else:
                target.extend([0])
    if not process:
        return final_datas,target,words
    # 进行均值滤波降维
    datas = final_datas.copy()
    final_datas = []
    for data in datas:
        d = []
        for i in range(len(data) // 4):
            d.append(np.sum(data[i * 4:i * 4 + 4]) / len(data[i * 4:i * 4 + 4]))
        final_datas.append(d)
    # 进行小波均衡化去均值
    datas = final_datas.copy()
    # for i, data in enumerate(datas):
    #     all = sum(data)
    #     datas[i] = [x / all for x in datas[i]]
    min_num = 0
    for i, data in enumerate(datas):
        mu = sum(data) / len(data)
        datas[i] = [x - mu for x in datas[i]]
        min_num_temp = min(datas[i])
        min_num = min(min_num_temp, min_num)
        datas[i].insert(0, 0)
        datas[i].append(0)
    # for i, data in enumerate(datas):
    #     datas[i] = [x - min_num for x in datas[i]]
    final_datas = datas
    y_news = []
    y_others = []
    for i in range(len(target)):
        if target[i] == 1:
            y_news.append(i)
        else:
            y_others.append(i)
    for i in range(len(y_news)):
        plt.plot(final_datas[y_news[i]], label=words[y_news[i]])
    plt.legend()
    plt.title("新闻关键词")
    plt.show()
    for i in range(len(y_others)):
        final_datas[y_others[i]]=[x/3 for x in final_datas[y_others[i]]]
        plt.plot(final_datas[y_others[i]], label=words[y_others[i]])
    plt.legend()
    plt.title("其他关键词")
    plt.show()

    return np.array(final_datas), target, words


def load_audio_instructions(method='energy'):
    path = "datasets/audioInstructions2/"
    audios = []
    files = []
    target = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        audio = load_voice(file_path)
        if method == 'energy':
            audios.append(audio_transform_to_energy(audio))
        else:
            audio = audio[np.where(audio >= 0)]
            # 截取相同长度
            common_length = 5500
            audio_length = len(audio)
            if audio_length > common_length:
                audio_outer = (audio_length - common_length) // 2
                audio = audio[audio_outer:audio_outer + common_length]
            else:
                audio_outer = (common_length - audio_length) // 2
                audio_new = np.zeros((common_length,), dtype=np.int16)
                audio_new[audio_outer:audio_outer + len(audio)] = audio
                audio = audio_new

            # 均值滤波
            box_length = 100
            audio_new = []
            for i in range(common_length // box_length):
                box = audio[i * box_length:i * box_length + box_length]
                audio_new.append(int(np.sum(box) / len(box)))
            audio = np.array(audio_new)
            audios.append(audio)

        files.append(file.split('.')[0])
        target.append(0)
    return np.array(audios), target, files


def audio_transform_to_energy(audio):
    common_length = 10000
    audio_length = len(audio)
    box_length = 200
    st_freq=0
    en_freq=100
    # 截取到相同长度的音频
    if audio_length > common_length:
        audio_outer = (audio_length - common_length) // 2
        audio = audio[audio_outer:audio_outer + common_length]
    else:
        audio_outer = (common_length - audio_length) // 2
        audio_new = np.zeros((common_length,), dtype=np.int16)
        audio_new[audio_outer:audio_outer + len(audio)] = audio
        audio = audio_new
    # 进行分时段的频谱能量计算生成矢量
    audio_new=[]
    for i in range(common_length // box_length):
        box = audio[i * box_length:i * box_length + box_length]
        fft_data = np.abs(fft(box))
        # plt.plot(box)
        # plt.plot(fft_data)
        # plt.show()
        audio_new.append(int(np.sum(fft_data[st_freq:en_freq]) / len(fft_data[st_freq:en_freq])))
    return audio_new
