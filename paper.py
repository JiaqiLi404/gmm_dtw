import matplotlib.pyplot as plt
from sympy import *
import numpy as np
import dataLoader as dataloader

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 输出dtw示意图
def print_dtw_example():
    x = [10, 15, 18, 14, 12, 11, 10]
    y = [20, 23, 25, 28, 27, 22, 20]
    plt.plot(x, color='r')
    plt.plot(y, color='g')
    lines_x = [[0, 10], [0, 10], [1, 15], [2, 18], [2, 18], [3, 14], [4, 12], [5, 11], [6, 10]]
    lines_y = [[0, 20], [1, 23], [2, 25], [3, 28], [4, 27], [5, 22], [5, 22], [5, 22], [6, 20]]
    for i in range(len(lines_x)):
        plt.plot([lines_x[i][0], lines_y[i][0]], [lines_x[i][1], lines_y[i][1]], 'y:')
    plt.title("时间序列对齐")
    plt.show()

    xp = [10, 10, 15, 18, 18, 14, 12, 11, 10]
    yp = [20, 23, 25, 28, 27, 22, 22, 22, 20]

    plt.plot(xp, color='r')
    plt.plot(yp, color='g')
    for i in range(len(lines_x)):
        plt.plot([i, i], [xp[i], yp[i]], 'y:')
    plt.title("经过动态时间规整后的序列")
    plt.show()


# 输出插值示意图
def print_insert_value_example():
    def lagrange(x, y):
        p, la = symbols('p  la')
        n = len(x)
        s = 0
        for k in range(n):
            la = y[k]
            for j in range(k):
                la = la * (p - x[j]) / (x[k] - x[j])
            for j in range(k + 1, n):
                la = la * (p - x[j]) / (x[k] - x[j])
            s = s + la
        print(expand(simplify(s)))
        return s

    data = [4329.23, 4009.23, 4289.23, 4148.21, 4350.26, 4586.15, 4096.92, 4641.03, 4222.05, 4238.46, 4211.28, 4280.51,
            4635.9, 4393.85]

    data_aft = [[4329.23, 4009.23, 4289.23], [4148.21, 4350.26], [4586.15, 4096.92, 4641.03], [4222.05, 4238.46],
                [4211.28, 4280.51], [4635.9, 4393.85]]
    data_aft_x = [[0, 1, 2], [4, 5], [7, 8, 9], [11, 12], [14, 15], [17, 18]]

    data_insert = [[4289.23, 4148.21], [4350.26, 4586.15], [4641.03, 4222.05], [4238.46, 4211.28], [4280.51, 4635.9]]
    data_insert_x = [[2, 3, 4], [5, 6, 7], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
    data_insert_time = [2.5, 4.5, 7.5, 9.5, 11.5, 12.5]
    p = symbols('p')

    plt.plot(data, label="插值前", color='r')
    for i, v in enumerate(data_aft):
        line1, = plt.plot(data_aft_x[i], [x + 1000 for x in v], color='g')
    for i, v in enumerate(data_insert):
        fx = lagrange([0, 1], v)
        # line2, = plt.plot(data_insert_x[i], [v[0] + 1000, fx.subs(p, 0.5) + 1000, v[1] + 1000], color='y')
        line2, = plt.plot(data_insert_x[i], [v[0] + 1000, v[0] + 1000, v[1] + 1000], color='y')
    plt.title('样本时间偏移效果')
    plt.xlabel('时间')
    plt.ylabel('电位')
    plt.legend(handles=[line1, line2], labels=["原序列", '插值'], loc='best')
    plt.show()


def print_hot_words_example():
    data_train, target, words = dataloader.load_hot_words_data(process=False)
    plt.plot(data_train[0], label='Digital Currency')
    plt.plot(data_train[20], label='Brexit')
    plt.legend()
    plt.show()
    data_train, target, words = dataloader.load_hot_words_data(process=True)
    plt.plot(data_train[15], label='Sino-Indian Border War')
    plt.plot(data_train[20], label='Brexit')
    plt.legend()
    plt.show()


def print_cluster_num():
    x = [2, 3, 4, 5, 6, 7, 8, 9,10]
    y = [244682.28302560688, 200571.0535817111, 189781.0885829004, 168553.84673432514, 157907.1615183182,
         93161.18629921137, 83852.79042848328, 75973.49456428894,71727.59857106445]
    plt.plot(x[:], y[:])
    plt.xlabel("clusters num")
    plt.ylabel("cov")
    plt.show()
    plt.plot([0,69],[0,144],label='piex')
    plt.plot([0, 9], [0, 1683],label='gps(/10)')
    plt.legend()
    plt.show()
    plt.plot([0, 108], [0, 338], label='piex')
    plt.plot([0, -1], [0, -2179], label='gps(/10)')
    plt.legend()
    plt.show()


print_cluster_num()
