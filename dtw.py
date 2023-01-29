import numpy as np
import itertools
import copy


def _list_remove_nan(list):
    while np.isnan(list[-1]):
        list.pop()
    return list


def _get_dtw_model_of_two_vactor(vac1, vac2):
    list1 = _list_remove_nan(list(vac1))
    list2 = _list_remove_nan(list(vac2))
    matrix = [list1, list2]
    dtw_model = DTW(matrix)

    dtw_model.silence_flag = True
    return dtw_model


def get_dtw_center_of_two_vactor(vac1, vac2, weights=None):
    dtw_model = _get_dtw_model_of_two_vactor(vac1, vac2)
    return dtw_model.get_center_by_category(weights)


def get_dtw_half_of_two_vactor(vac, center, weights=0.5):
    dtw_model = _get_dtw_model_of_two_vactor(vac, center)
    return dtw_model.get_half_by_category(0, weights)


def get_dtw_sum_of_two_vactor(vac1, vac2,a_vac1=1,a_vac2=1):
    dtw_model = _get_dtw_model_of_two_vactor(vac1, vac2)
    return dtw_model.get_sum_by_category(a_vac1,a_vac2)


def get_dtw_distance_of_two_vactor(vac1, vac2):
    dtw_model = _get_dtw_model_of_two_vactor(vac1, vac2)
    return dtw_model.get_distance()


class DTW:
    def __init__(self, data, quick_flag=False):
        self.step_num = None
        self._distance = None
        self.categorys = None
        self.dim_length = []
        self.dim_sum = 0
        data_temp = copy.deepcopy(data)

        # 进行数据对齐以便numpy生成矩阵，长度不够的list补nan
        maxLength = 0
        for i in data_temp:
            self.dim_length.append((len(i)))
            if len(i) > maxLength:
                maxLength = len(i)

        self.dim_sum = sum(self.dim_length)
        for index, value in enumerate(data_temp):
            while len(data_temp[index]) < maxLength:
                data_temp[index].append(np.nan)

        self.data = np.array(data_temp)
        self.data_num = self.data.shape[0]
        self.quick_flag = quick_flag
        self.silence_flag = False

        # 低复杂度算法所需参数
        self.center = np.zeros(self.data.shape[1], np.float64)

    def _float_equlal(self, a, b):
        if np.abs(a - b) < 1e-5:
            return True
        return False

    def _get_cost(self, i, j, k=None):
        if k is None:
            return abs(self.data.item((0, i)) - self.data.item((1, j)))
        else:
            return abs(self.data.item((0, i)) - self.data.item((1, j))) + abs(
                self.data.item((0, i)) - self.data.item((2, k))) + abs(self.data.item((1, j)) - self.data.item((2, k)))

    # 支持探索所有可能的dtw路径
    def _trace_category_step_multipath(self, coord, costs):
        for i, c in enumerate(coord):
            if c.sum() == 0: continue
            self.step_num += 1
            if self.data_num == 2:
                cost = self._get_cost(c[0], c[1])  # 计算绝对距离
                last = costs[c[0] + 1, c[1] + 1]
            else:
                cost = self._get_cost(c[0], c[1], c[2])  # 计算绝对距离
                last = costs[c[0] + 1, c[1] + 1, c[2] + 1]
            last1 = last - cost
            last2 = -last - cost
            add_flag = False
            c_temp = copy.deepcopy(c)
            if self.data_num == 2:
                if self._float_equlal(last1, costs[c_temp[0], c_temp[1]]) or self._float_equlal(last2, costs[
                    c_temp[0], c_temp[1]]):
                    self.categorys[i].append([c[0] - 1, c[1] - 1])
                    c[0] -= 1
                    c[1] -= 1
                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0], c_temp[1] + 1]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[0] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[0] -= 1
                        self.categorys[i].append([c[0], c[1]])

                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0] + 1, c_temp[1]]) or self._float_equlal(last2, costs[
                    c_temp[0] + 1, c_temp[1]]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[1] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[1] -= 1
                        self.categorys[i].append([c[0], c[1]])
                    add_flag = True
            else:
                # [costs[i, j, k - 1], costs[i, j - 1, k], costs[i, j - 1, k - 1], costs[i - 1, j, k],
                # costs[i - 1, j, k - 1], costs[i - 1, j - 1, k], costs[i - 1, j - 1, k - 1]]
                if self._float_equlal(last, costs[c_temp[0], c_temp[1], c_temp[2]]):
                    self.categorys[i].append([c[0] - 1, c[1] - 1, c[2] - 1])
                    c[0] -= 1
                    c[1] -= 1
                    c[2] -= 1
                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0], c_temp[1], c_temp[2] + 1]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[0] -= 1
                        c_temp_temp[1] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[0] -= 1
                        c[1] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])

                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0], c_temp[1] + 1, c_temp[2]]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[0] -= 1
                        c_temp_temp[2] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[0] -= 1
                        c[2] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])

                    add_flag = True
                # [costs[i, j, k - 1], costs[i, j - 1, k], costs[i, j - 1, k - 1], costs[i - 1, j, k]
                if self._float_equlal(last, costs[c_temp[0], c_temp[1] + 1, c_temp[2] + 1]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[0] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[0] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])

                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0] + 1, c_temp[1], c_temp[2]]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[1] -= 1
                        c_temp_temp[2] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[1] -= 1
                        c[2] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])

                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0] + 1, c_temp[1], c_temp[2] + 1]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[1] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[1] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])

                    add_flag = True
                if self._float_equlal(last, costs[c_temp[0] + 1, c_temp[1] + 1, c_temp[2]]):
                    if add_flag:
                        c_temp_temp = copy.deepcopy(c_temp)
                        c_temp_temp[2] -= 1
                        coord.append(c_temp_temp)
                        cat_temp = copy.deepcopy(self.categorys[i])
                        cat_temp.pop()
                        cat_temp.append(c_temp_temp)
                        self.categorys.append(cat_temp)
                    else:
                        c[2] -= 1
                        self.categorys[i].append([c[0], c[1], c[2]])
                    add_flag = True
        return coord

    # 搜索一条最短的dtw路径,目前仅支持两个向量间的dtw计算
    def _trace_category_step_fastestpath(self, coord, costs):
        c = coord[0]
        # print(c)
        self.step_num += 1
        if self.data_num == 2:
            cost = self._get_cost(c[0], c[1])  # 计算绝对距离
            last = costs[c[0] + 1, c[1] + 1]
        else:
            print("搜索一条最短的dtw路径目前仅支持两个向量间的dtw计算")
            return [[0]]
        last1 = last - cost
        last2 = -last - cost
        c_temp = copy.deepcopy(c)
        if self._float_equlal(last1, costs[c_temp[0], c_temp[1]]) or self._float_equlal(last2,costs[c_temp[0], c_temp[1]]):
            self.categorys[0].append([c[0] - 1, c[1] - 1])
            c[0] -= 1
            c[1] -= 1
        elif self._float_equlal(last1, costs[c_temp[0], c_temp[1] + 1]) or self._float_equlal(last2, costs[
            c_temp[0], c_temp[1] + 1]):
            c[0] -= 1
            self.categorys[0].append([c[0], c[1]])
        elif self._float_equlal(last1, costs[c_temp[0] + 1, c_temp[1]]) or self._float_equlal(last2, costs[
            c_temp[0] + 1, c_temp[1]]):
            c[1] -= 1
            self.categorys[0].append([c[0], c[1]])
        else:
            print("路径回溯失败：")
            print(last, costs[c_temp[0], c_temp[1]], costs[c_temp[0], c_temp[1] + 1], costs[c_temp[0] + 1, c_temp[1]])
            return [[0]]
        return [c]

    def compute_distance_DP(self, skip_category=False):
        if not self.silence_flag:
            print("***starting*** computing distance by DP method")
        self.step_num = 0
        self._distance = 0
        self.categorys = [[[]]]
        for i in range(self.data_num): self.categorys[0][0].append(self.dim_length[i] - 1)
        coord = [np.array(self.categorys[0][0], np.int32)]
        # 生成代价矩阵
        dim_length_temp = [x + 1 for x in self.dim_length]
        costs = np.zeros(tuple(dim_length_temp), np.float64)
        costs.fill(np.inf)
        costs.itemset(0, 0)
        for i in range(1, self.dim_length[0] + 1):
            for j in range(1, self.dim_length[1] + 1):
                if self.data_num == 2:
                    cost = self._get_cost(i - 1, j - 1)  # 计算绝对距离
                    # 找到当前索引方块的top,left,top_left方向的累积损失的最小值
                    last_min = np.min(
                        [cost + costs[i, j - 1], cost + costs[i - 1, j], cost + costs[i - 1, j - 1]])
                    costs[i, j] = last_min
                else:
                    for k in range(1, self.dim_length[2] + 1):
                        cost = self._get_cost(i - 1, j - 1, k - 1)  # 计算绝对距离
                        # 找到当前索引方块的top,left,top_left方向的累积损失的最小值
                        last_min = np.min(
                            [costs[i, j, k - 1], costs[i, j - 1, k], costs[i, j - 1, k - 1], costs[i - 1, j, k],
                             costs[i - 1, j, k - 1], costs[i - 1, j - 1, k], costs[i - 1, j - 1, k - 1]])
                        costs[i, j, k] = cost + last_min
        if self.data_num == 2:
            self._distance = costs[-1, -1]
        else:
            self._distance = costs[-1, -1, -1]

        # 依据cost生成路径,可以捕捉重复的路径
        if skip_category:
            return
        while self._check_coord(coord):
            coord = self._trace_category_step_fastestpath(coord, costs)

        for cat in self.categorys: cat.reverse()

        if not self.silence_flag:
            print("data:")
            print(self.data)
            print("category:")
            print(self.categorys)
            print("costs:")
            print(costs)
            print("***finished*** total steps:", self.step_num + 1)

    def get_distance(self):
        if self._distance is None:
            self.compute_distance_DP(skip_category=True)
        return self._distance

    def _check_coord(self, coord):
        for c in coord:
            if c.sum() != 0:
                return True
        return False

    def get_center_by_category(self, weight=None):
        if self.categorys is None:
            self.compute_distance_DP()

        self.center = []
        for i, category in enumerate(self.categorys):
            self.center.append([])
            for cat in category:
                values = []
                for j, v in enumerate(cat):
                    values.append(self.data.item((j, v)))
                self.center[i].append(np.average(values, weights=weight))
        return self.center

    def get_sum_by_category(self,a_vac1,a_vac2):
        if self.categorys is None:
            self.compute_distance_DP()

        sum_list = []
        for i, category in enumerate(self.categorys):
            sum_list.append([])
            for cat in category:
                values = []
                for j, v in enumerate(cat):
                    values.append(self.data.item((j, v)))
                sum_list[i].append(a_vac1*values[0]+a_vac2*values[1])
        return sum_list

    def get_half_by_category(self, index, weights):
        if self.categorys is None:
            self.compute_distance_DP()
        half = []
        for i, category in enumerate(self.categorys):
            half.append([])
            for cat in category:
                half[i].append(self.data.item((index, cat[index])))
            half[i] = [x * weights for x in half[i]]
        return half

    def compute_center_low_complexity(self, iter_num, init_center_flag='avg', force_iter_flag=False):
        '''
        :param iter_num: 最大迭代次数
        :param init_center_flag: ‘avg’表示均值得到初始化中心，‘dtw’表示dtw相加得到初始化中心
        :param force_iter_flag: True、False表示是否在收敛后强制继续迭代
        :return:
        '''
        print("***starting*** computing distance by Low Complexity method")
        self._get_init_center(init_center_flag)
        # self.center = [[30, 60, 60]]
        delta = []
        # self.center = [80, 40, 50, 30]

        print("init center:", self.center)
        # 迭代次数
        for iter_i in range(iter_num):
            # 判断退出条件
            if iter_i >= 1:
                if (min(delta) < 1e-5 or len(self.center) > 10) and not force_iter_flag:
                    print("delta is ", delta)
                    print("delta is %.5f,stop iterating" % min(delta))
                    break
                print("delta is ", delta)
            print("--starting iterator %d--" % iter_i)
            # 获取各样本与中心的对齐份额
            next_center = []
            add_center = []
            for index, c in enumerate(self.center):
                next_center.append([c])
                # 各矢量对齐
                for n in range(self.data_num):
                    # 取n份额，可能有多种可能
                    new_c = get_dtw_half_of_two_vactor(self.data[n, :], c, 1 / self.data_num)
                    for nc_i, nc in enumerate(new_c):
                        if nc_i == 0:
                            next_center[index].append(nc)
                        else:
                            next_center_temp = copy.deepcopy(next_center[index])
                            next_center_temp.pop()
                            next_center_temp.append(nc)
                            add_center.append(next_center_temp)
                    # print(next_center[-1], self.data[n, :], self.center)
            next_center.extend(add_center)
            # 各中心依次对齐
            self.center = []
            delta = []
            print(next_center)
            # next_center:[[路径1[上个中心]，[向量1],[向量2]] , [路径2[向量1],[向量2]]]
            # center:[[路径1[中心1],[中心2]] , [路径2[中心1],[中心2]]]
            for path_i, path in enumerate(next_center):
                self.center.append([])
                last_center = None
                for vac_i, vac in enumerate(path):
                    if vac_i == 0:
                        last_center = vac
                    elif vac_i == 1:
                        self.center[path_i] = [vac]
                    else:
                        new_centers = []
                        for alter_center in self.center[path_i]:
                            # print(vac, self.center)
                            new_centers.extend(get_dtw_sum_of_two_vactor(vac, alter_center))
                        self.center[path_i] = new_centers
                for alter_center in self.center[path_i]:
                    delta.append(get_dtw_distance_of_two_vactor(alter_center, last_center))
            new_centers = []
            for path in self.center:
                for c in path:
                    # 去除重复center
                    if c not in new_centers:
                        new_centers.append(c)

            self.center = new_centers
            print("center:", self.center)
        print("***finished*** center:", self.center)

    def _get_init_center(self, flag):
        data_temp = np.nan_to_num(self.data)
        if flag == 'avg':
            self.center = [list(np.average(data_temp, axis=0))]
        elif flag == 'dtw':
            for index, vac in enumerate(self.data):
                vac = vac[:self.dim_length[index]]
                if index == 0:
                    self.center = [list(vac)]
                else:
                    new_center = []
                    for center_i, center in enumerate(self.center):
                        new_center.extend(get_dtw_sum_of_two_vactor(center, vac))
                    self.center = new_center
            for i, center in enumerate(self.center):
                self.center[i] = [x / self.data_num for x in center]


# 验证两个矢量的em算法正确性
def exp_em_2():
    # data = [[100, 80, 60, 90], [2, 50, 60]]
    # data = [[8, 9, 10], [14, 13, 11]]
    # data = [[94, 10, 65], [21, 77, 10]]
    # data = [[73, 2, 27], [45, 50, 77]]
    data = [[44, 9, 54], [70, 94, 72]]
    # data = np.random.randint(0, 100, size=(2, 3))
    print("data:", data)
    dtw_s = DTW(data)
    dtw_s.compute_distance_DP()
    print("real center:", dtw_s.get_center_by_category())
    dtw_l = DTW(data)
    dtw_l.compute_center_low_complexity(5, force_iter_flag=True)
    print("***summary***")
    print("real center:", dtw_s.center)
    print("guess center:", dtw_l.center)
    effective_flag = False
    for real_center in dtw_s.center:
        for guess_center in dtw_l.center:
            if get_dtw_distance_of_two_vactor(real_center, guess_center) == 0:
                effective_flag = True
                print("SUCCESS, distance:",
                      get_dtw_distance_of_two_vactor(real_center, guess_center), real_center, guess_center)
    if not effective_flag:
        print("FAIL, guess center is wrong")


# 验证交换律
def exp_law_of_commutation():
    data = np.random.randint(0, 100, size=(3, 3))
    a = data[0]
    b = data[1]
    c = data[2]
    print('a', a, 'b', b, 'c', c)
    dtw_plus_ab = get_dtw_sum_of_two_vactor(a, b)
    dtw_plus_abc = []
    for ab in dtw_plus_ab:
        dtw_plus_abc.extend(get_dtw_sum_of_two_vactor(ab, c))
    print("a+b+c=", dtw_plus_abc)
    dtw_plus_ac = get_dtw_sum_of_two_vactor(a, c)
    dtw_plus_acb = []
    for ac in dtw_plus_ac:
        dtw_plus_acb.extend(get_dtw_sum_of_two_vactor(ac, b))
    print("a+c+b=", dtw_plus_acb)
    print("distance:", get_dtw_distance_of_two_vactor(dtw_plus_abc[0], dtw_plus_acb[0]))


# 验证三个矢量的em算法，直接均值初始化中心与dtw求初始化中心的效果差异
def exp_em_3():
    # data = [[100, 80, 60, 90], [2, 50, 60]]
    # data = [[8, 9, 10], [14, 13, 11]] # 和为5维向量
    # data = [[94, 10, 65], [21, 77, 10]] # 存在多个dtw和向量
    # data = [[73, 2, 27], [45, 50, 77]]
    # data = [[44, 9, 54], [70, 94, 72]]
    # data = np.random.randint(0, 100, size=(3, 3))
    data = [[31, 37, 83], [79, 98, 59], [42, 99, 16]]

    print("data:", data)
    dtw_s = DTW(data)
    dtw_s.compute_distance_DP()
    print("real center:", dtw_s.get_center_by_category())
    dtw_l_avg = DTW(data)
    dtw_l_avg.compute_center_low_complexity(5, init_center_flag='avg', force_iter_flag=False)
    dtw_l_dtw = DTW(data)
    dtw_l_dtw.compute_center_low_complexity(5, init_center_flag='dtw', force_iter_flag=False)
    print("***summary***")
    print("real center:", dtw_s.center)
    print("guess center init with avg:", dtw_l_avg.center)
    print("guess center init with dtw:", dtw_l_dtw.center)
    min_avg_distance = np.inf
    min_avg_center = None
    for real_center in dtw_s.center:
        for guess_center in dtw_l_avg.center:
            distance = get_dtw_distance_of_two_vactor(real_center, guess_center)
            if distance < min_avg_distance:
                min_avg_distance = distance
                min_avg_center = guess_center
    min_dtw_distance = np.inf
    min_dtw_center = None
    for real_center in dtw_s.center:
        for guess_center in dtw_l_dtw.center:
            distance = get_dtw_distance_of_two_vactor(real_center, guess_center)
            if distance < min_dtw_distance:
                min_dtw_distance = distance
                min_dtw_center = guess_center
    if min_dtw_distance <= min_avg_distance + 1e-5:
        effective_flag = 'better'
        print("SUCCESS, distance of center with init with avg method:",
              min_avg_distance, min_avg_center)
        print("SUCCESS, distance of center with init with dtw method:",
              min_dtw_distance, min_dtw_center)
        if min_avg_distance > 1e-2 and (min_avg_distance - min_dtw_distance) > 1e-2:
            print("IMPROVED,improved %.2f" % ((float(min_avg_distance) - float(min_dtw_distance)) / float(
                min_avg_distance)))
        if abs(min_dtw_distance - min_avg_distance) <= 1e-5:
            effective_flag = 'equal'
    else:
        effective_flag = 'worse'
        print("FAIL, distance of center with init with avg method:",
              min_avg_distance, min_avg_center)
        print("FAIL, distance of center with init with dtw method:",
              min_dtw_distance, min_dtw_center)
    return effective_flag


# 重复多次进行试验3，证明：验证三个矢量的em算法，直接均值初始化中心与dtw求初始化中心的效果差异
def exp_em_4():
    result = []
    result_num = 0
    better_num = 0
    worse_num = 0
    while result_num < 100:
        result_num += 1
        re = exp_em_3()
        result.append(re)
        if re == 'worse':
            worse_num += 1
        elif re == 'better':
            better_num += 1
    print("***final summary***")
    print(result_num, result)
    print('better', better_num)
    print('worse', worse_num)


# 测试多维矢量集的效率
def exp_em_5():
    data = np.random.randint(0, 100, size=(100, 15))
    dtw_l_avg = DTW(data)
    dtw_l_avg.compute_center_low_complexity(5, init_center_flag='avg', force_iter_flag=False)


# 测试
def exp_em_6():
    data = np.random.randint(0, 100, size=(2, 5))
    dtw_l_avg = DTW(data)
    dtw_l_avg.compute_distance_DP()


if __name__ == '__main__':
    exp_em_6()
