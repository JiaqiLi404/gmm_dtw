import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import random
import dtw
from sklearn import datasets
import dataLoader as dataloader

from sklearn.cluster import KMeans
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset

from tslearn.generators import random_walks

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置


class GMM:
    data = None
    data_num = 0
    data_div = 0
    avg_k = None  # 均值
    cov_k = None  # 协方差
    alpha_k = None  # 权值
    class_num = 0  # 聚类数
    gamma = None

    def __init__(self, data, class_num, dtw_flag=False, kmeans_center=True, only_kmeans=True, avg_k=None):
        if dtw_flag:
            kmeans_iter_num = 20
        else:
            kmeans_iter_num = 1000
        self.only_kmeans = only_kmeans
        self.result = None
        self.category = None
        self.data = data
        self.avg_k = avg_k
        self._scale_data()
        self.data_num, self.data_div = self.data.shape
        self.class_num = class_num

        self.use_dtw = dtw_flag
        self.process = 0
        belong = np.zeros((len(data), class_num))
        if kmeans_center:
            center = self.data[0:class_num].copy()
            distances = np.zeros((len(data), class_num))
            for i in range(kmeans_iter_num):
                belong = np.zeros((len(data), class_num))
                for d in range(len(self.data)):
                    for c in range(class_num):
                        if dtw_flag:
                            distances[d, c] = dtw.get_dtw_distance_of_two_vactor(self.data[d], center[c])
                        else:
                            distances[d, c] = np.sum(np.abs(self.data[d] - center[c]))
                for b in range(len(data)):
                    max_ind = int(np.where(distances[b] == np.amin(distances[b]))[0][0])
                    belong[b, max_ind] = 1
                for c in range(class_num):
                    for d in range(self.data_div):
                        if np.sum(belong[:, c]) == 0:
                            center[c] = self.data[np.random.randint(0, self.data_num - 1)]
                            continue
                        center[c, d] = np.sum(np.matmul(self.data[:, d], belong[:, c])) / np.sum(belong[:, c])
            self.avg_k = center
            self.cov_k = np.zeros((self.class_num,))
            self.alpha_k = np.zeros((self.class_num,))
            for i in range(self.class_num):
                self.alpha_k[i] = np.sum(belong[:, i]) / self.data_num
                for j in range(self.data_num):
                    if self.use_dtw:
                        self.cov_k[i] += dtw.get_dtw_distance_of_two_vactor(self.data[j], self.avg_k[i]) ** 2 * belong[
                            j, i]
                    else:
                        self.cov_k[i] += np.sum((self.data[j] - self.avg_k[i]) ** 2) * belong[j, i]
            self.category = []
            for i in range(self.data_num):
                self.category.append((np.where(belong[i] == 1)[0][0]))
        else:
            if avg_k is None:
                # 基于class_num自动生成avg
                var_per_class = (np.amax(data, axis=0) - np.amin(data, axis=0)) / (self.class_num - 1)
                self.avg_k = [np.amin(data, axis=0)]
                for i in range(1, self.class_num):
                    self.avg_k.append(self.avg_k[i - 1] + var_per_class)
                self.avg_k = np.array(self.avg_k)
                # self.cov_k = np.array([np.eye(self.data_div)] * class_num)
                cov = self.data_num * np.sum((var_per_class / 2) ** 2)
                self.cov_k = []
                for i in range(self.class_num):
                    self.cov_k.append(cov.copy())
                self.cov_k = np.array(self.cov_k)
            else:
                self.avg_k = np.array(avg_k)
                self.cov_k = np.zeros((self.class_num,))
                for i in range(self.class_num):
                    for j in range(self.data_num):
                        if self.use_dtw:
                            self.cov_k[i] += dtw.get_dtw_distance_of_two_vactor(self.data[j], self.avg_k[i]) ** 2
                        else:
                            self.cov_k[i] += np.sum((self.data[j] - self.avg_k[i]) ** 2)
            self.alpha_k = np.array([1.0 / class_num] * class_num)
        print("init center:", self.avg_k)

    def _compute_exception(self):
        # self.gamma = np.zeros((self.data_num, self.class_num))
        # 计算概率
        self.gamma = self._phi(self.avg_k, self.cov_k)

        # 计算每个模型对每个样本的响应度
        for k in range(self.class_num):
            self.gamma[:, k] = self.alpha_k[k] * self.gamma[:, k]

        for i in range(self.data_num):
            if np.sum(self.gamma[i, :]) == 0:
                print("mu或者方差var过小导致概率计算时某一样本的概率和为0...")
                print(self.avg_k)
                print(self.cov_k)
            self.gamma[i, :] /= np.sum(self.gamma[i, :])

    def _phi(self, mu_k, cov_k):
        '''
        norm = multivariate_normal(mean=mu_k, cov=cov_k)
        return norm.pdf(self.data)

                    for j in range(self.class_num):
                dominator = dominator + np.exp(-1.0 / (2.0 * cov_k[j]) * np.sum((self.data[i] - mu_k[j]) ** 2))
            for j in range(self.class_num):
                numerator = np.exp(-1.0 / (2.0 * cov_k[j]) * np.sum((self.data[i] - mu_k[j]) ** 2))
                posterior[i, j] = numerator / dominator
        '''
        posterior = np.mat(np.zeros((self.data_num, self.class_num)))
        for i in range(self.data_num):
            for j in range(self.class_num):
                under = (2 * np.pi) ** (self.data_num / 2) * np.sqrt(abs(cov_k[j]))
                if self.use_dtw:
                    top = np.exp(-0.5 * ((dtw.get_dtw_distance_of_two_vactor(self.data[i], mu_k[j])) ** 2 / cov_k[j]))
                else:
                    top = np.exp(-0.5 * (np.sum(((self.data[i] - mu_k[j])) ** 2)) / cov_k[j])
                posterior[i, j] = top / under
        return posterior

    def _compute_maximize_normal(self):
        '''
        # 这是方差不为纯量的代码
        # 初始化参数值
        mu = np.zeros((self.class_num, self.data_div))
        cov = []
        alpha = np.zeros(self.class_num)

        # 更新每个模型的参数
        for k in range(self.class_num):
            # 第 k 个模型对所有样本的响应度之和
            Nk = np.sum(self.gamma[:, k])
            # 更新 mu
            # 对每个特征求均值
            for d in range(self.data_div):
                mu[k, d] = np.sum(np.multiply(self.gamma[:, k], self.data[:, d])) / Nk
            # print(mu)
            # 更新 cov
            cov_k = np.mat(np.zeros((self.data_div, self.data_div)))
            for i in range(self.data_num):
                cov_k += self.gamma[i, k] * (np.array([self.data[i] - mu[k]]).T * np.array([self.data[i] - mu[k]])) / Nk
            cov.append(cov_k)
            # 更新 alpha
            alpha[k] = Nk / self.data_num
        cov = np.array(cov)
        return mu, cov, alpha
        '''
        # 初始化参数值
        mu = np.zeros((self.class_num, self.data_div))
        cov = []
        alpha = np.zeros(self.class_num)

        # 更新每个模型的参数
        for k in range(self.class_num):
            # 第 k 个模型对所有样本的响应度之和
            Nk = np.sum(self.gamma[:, k])
            if Nk == 0:
                print("第{}个模型没有任何实例".format(k))
            # 更新 mu
            # 对每个特征求均值
            for d in range(self.data_div):
                mu[k, d] = np.sum(np.matmul(self.data[:, d], self.gamma[:, k])) / Nk
            # print(mu)
            # 更新 cov
            cov_k = 0
            for i in range(self.data_num):
                cov_k += self.gamma[i, k] * np.sum(np.array([self.data[i] - mu[k]]) ** 2) / Nk
            cov.append(cov_k)
            # 更新 alpha
            alpha[k] = Nk / self.data_num
        cov = np.array(cov)
        return mu, cov, alpha

    def _compute_maximize_dtw(self):
        # 初始化参数值
        mu = []
        for i in range(self.class_num):
            mu.append([])
        # mu = np.zeros((self.class_num, self.data_div))
        cov = []
        alpha = np.zeros(self.class_num)

        # 更新每个模型的参数
        for k in range(self.class_num):
            # 第 k 个模型对所有样本的响应度之和
            Nk = np.sum(self.gamma[:, k])
            if Nk == 0:
                print("第{}个模型没有任何实例".format(k))
            # 对每个特征求均值
            mu[k] = self.data[0]
            gamma_sum = self.gamma[0, k]

            for d in range(1, self.data_num):
                # vactor1 = list((gamma_sum / (self.gamma[d, k] + gamma_sum) * np.array(mu[k])))
                # if (np.isnan(vactor1[0])):
                #     vactor1 = list(np.zeros(len(vactor1)))
                # vactor2 = self.gamma[d, k] / (self.gamma[d, k] + gamma_sum) * self.data[d]
                # if (np.isnan(vactor2[0])):
                #     vactor2 = list(np.zeros(len(vactor2)))
                a1 = gamma_sum / (self.gamma[d, k] + gamma_sum)
                # a2 = self.gamma[d, k] / (self.gamma[d, k] + gamma_sum)
                a2 = 1 - a1
                mu[k] = dtw.get_dtw_sum_of_two_vactor(list(mu[k]), list(self.data[d]), a1, a2)[0]
                gamma_sum += self.gamma[d, k]
            # print(mu)
            # 更新 cov
            cov_k = 0
            if Nk == 0:
                alpha[k] = 0
                cov.append(00)
                continue
            for i in range(self.data_num):
                # cov_k += self.gamma[i, k] * (self.data[i] - mu[k]).T * (self.data[i] - mu[k]) / Nk
                cov_k += self.gamma[i, k] * (dtw.get_dtw_distance_of_two_vactor(self.data[i], mu[k]) ** 2) / Nk
            cov.append(cov_k)
            # 更新 alpha
            alpha[k] = Nk / self.data_num
        cov = np.array(cov)

        return mu, cov, alpha

    def _scale_data(self):
        # 对每一维特征分别进行缩放
        for i in range(self.data_num):
            max_ = self.data[:, i].max()
            min_ = self.data[:, i].min()
            self.data[:, i] = (self.data[:, i] - min_) / (max_ - min_)

    def fit(self, times, target=None):
        if self.only_kmeans:
            self.result = []
            for i in range(self.class_num):
                self.result.append([])
            for i in range(self.data_num):
                self.result[self.category[i]].append(self.data[i])
            return
        for i in range(times):
            self._compute_exception()
            if self.use_dtw:
                self.avg_k, self.cov_k, self.alpha_k = self._compute_maximize_dtw()
                self.category = self.gamma.argmax(axis=1).flatten().tolist()[0]
                self.result = []
                for j in range(self.class_num):
                    self.result.append([])
                for j in range(self.data_num):
                    self.result[self.category[j]].append(self.data[j])
                self.show()
                print("step {times} {sep} Result {sep}".format(sep="-" * 20, times=i + 1))
                print("mu:", self.avg_k)
                print("cov:", self.cov_k)
                print("alpha:", self.alpha_k)
                # self.check_result(target)
            else:
                self.avg_k, self.cov_k, self.alpha_k = self._compute_maximize_normal()
            # 被归为同一类了
            if np.sum(np.where(self.alpha_k == 1, 1, 0)) == 1:
                print("所有数据已经被归为同一类...")
                break
        self.category = self.gamma.argmax(axis=1).flatten().tolist()[0]
        self.result = []
        for i in range(self.class_num):
            self.result.append([])
        for i in range(self.data_num):
            self.result[self.category[i]].append(self.data[i])
        print("{sep} Result {sep}".format(sep="-" * 20))
        print("mu:", self.avg_k)
        print("cov:", self.cov_k)
        print("alpha:", self.alpha_k)

    def show(self):
        for i in range(self.class_num):
            x = [i[1] for i in self.result[i]]
            y = [i[2] for i in self.result[i]]
            plt.scatter(x, y)
        if self.use_dtw:
            plt.title("GMM Clustering By DTW-Joined EM Algorithm")
        else:
            plt.title("GMM Clustering By Classic EM Algorithm")
        plt.show()

    def check_result(self, target):
        if target is None:
            return
        correct_list = []
        wrong_list = []
        for i in range(self.data_num):
            if self.category[i] == target[i]:
                correct_list.append(self.data[i])
            else:
                wrong_list.append(self.data[i])
        if len(wrong_list) > len(correct_list):
            temp_list = wrong_list
            wrong_list = correct_list
            correct_list = temp_list
        x = [i[0] for i in correct_list]
        y = [i[1] for i in correct_list]
        plt.scatter(x, y, c='g')
        x = [i[0] for i in wrong_list]
        y = [i[1] for i in wrong_list]
        plt.scatter(x, y, c='r')
        plt.title("data correctness distribution")
        plt.show()
        print("correct:", len(correct_list) / self.data_num * 100, '%')


def run(data_flag='n', model_flag='dtw'):
    words = []
    if data_flag == 'rand':
        data_train, target = dataloader.generate_data()
    elif data_flag == 'ed':
        data_train, target = dataloader.load_eye_data(True)
    elif data_flag == 'e':
        data_train, target = dataloader.load_eye_data()
    elif data_flag == 'real':
        data_train, target = dataloader.load_real_data()
    elif data_flag == 'reald':
        data_train, target = dataloader.load_real_data(time_delay=True)
    elif data_flag == 'v':
        data_train, target = dataloader.load_voice_data()
    elif data_flag == 'h':
        # Temp：TP: 正判正    FN: 正判负     FP: 负判正     TN: 负判负
        # 改进      15          0               2           13
        # 传统      13          2               6           9
        data_train, target, words = dataloader.load_hot_words_data()
    elif data_flag == 'ai':
        data_train, target, words = dataloader.load_audio_instructions()
    else:
        data = datasets.load_wine()
        data_train = data['data']
    words = ['Digital currency ', 'Beirut explosion accident in 2020', 'Artificial neural network', 'Hantavirus',
             'Computer vision', 'GameStop', 'Tea culture', 'Black Lives Matter', 'Economic Times', 'WWDC',
             'Empire State Building', 'the withdrawal of the American occupying forces from Afghanistan',
             'market trends',
             'COVID-19 epidemic situation', 'Russia Ukraine war', 'Sino-Indian border war ', 'Monkeypox', 'Bundesliga',
             'Baidu', 'Ethereum', 'Brexit', 'Iran', 'Munich Security Conference', 'Apple Inc.',
             'Arecibo Radio Telescope', 'Fenerbahçe Spor Kulübü', 'Omicron', 'Firefox Browser',
             'the Korean nuclear issue', 'Google Translate']
    print(data_train.shape)
    print("avg:", np.average(data_train, axis=0))
    print("max:", np.amax(data_train, axis=0))
    print("min:", np.amin(data_train, axis=0))

    dataloader.show_data_distribution(data_train, target)

    avg = [data_train[1], data_train[2], data_train[26], data_train[22]]
    # avg = [data_train[1], data_train[2]]
    gmm_n = GMM(np.array(data_train), 7, dtw_flag=False, avg_k=avg)
    gmm_n.fit(1)
    gmm_n.show()
    # gmm_n.check_result(target)
    # 保存真实数据用于聚类
    # np.save("points.npy", gmm_n.result[0])

    if model_flag == 'kmeans':
        cluster = KMeans(n_clusters=7, random_state=9)
        cluster = cluster.fit(data_train)
        y_pred =  cluster.labels_
        print(y_pred)
    elif model_flag == 'kshape':
        print(data_train)
        data_train=to_time_series_dataset(data_train)
        cluster = KShape(n_clusters=7, n_init=1, random_state=9)
        cluster = cluster.fit(data_train)
        y_pred = cluster.labels_
        print(y_pred)
    else:
        dtw_flag = True
        if model_flag == 'gmm':
            dtw_flag = False
        gmm_dtw = GMM(np.array(data_train), 7, dtw_flag=dtw_flag, avg_k=avg)
        gmm_dtw.fit(1, target)
        # gmm_dtw.show()
        if data_flag == 'h' or data_flag == 'ai':
            cate = gmm_n.category
            result = [[], [], [], [], [], [], [], [], [], []]
            resultwords = [[], [], [], [], [], [], [], [], [], []]
            i = 0
            for cat in cate:
                result[cat].append(i)
                resultwords[cat].append(words[i])
                i += 1
            print('normal gmm results:', resultwords)
            cate = gmm_dtw.category
            result2 = [[], [], [], [], [], [], [], [], [], []]
            resultwords = [[], [], [], [], [], [], [], [], [], []]
            i = 0
            for cat in cate:
                result2[cat].append(i)
                resultwords[cat].append(words[i])
                i += 1

            for c in result2:
                if len(c) > 0:
                    for i in c:
                        plt.plot(data_train[i], label=words[i])
                        plt.title('dtw' + str(words[i]))
                    plt.legend()
                    plt.show()
            print('dtw gmm results:', resultwords)
            print(sum(gmm_dtw.cov_k))


if __name__ == '__main__':
    run('h','dtw')
