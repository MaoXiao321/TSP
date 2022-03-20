import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt


class GA(object):
    """遗传算法确定最佳路径"""

    def __init__(self, num_city, num_total, iteration, data, dis_mat):
        self.num_city = num_city
        self.num_total = num_total  # 候选种群数
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2  # 选择率
        self.mutate_ratio = 0.05  # 变异率
        self.e = num_city-1  # 指定的终点顺序

        # fruits中存路径顺序
        self.dis_mat = dis_mat
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city, self.e)

        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)  # 算适应度
        sort_index = np.argsort(-scores)  # 适应度排序由大到小排序
        init_best = self.fruits[sort_index[0]]  # 最佳适应度的路径顺序
        init_best = self.location[init_best]  # 最佳适应度的坐标顺序

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]  # 迭代次数
        self.iter_y = [1. / scores[sort_index[0]]]  # 路径长度

    # def random_init(self, num_total, num_city):
    #     tmp = [x for x in range(num_city)]
    #     result = []
    #     for i in range(num_total):
    #         random.shuffle(tmp)
    #         result.append(tmp.copy())
    #     return result

    def greedy_init(self, dis_mat, num_total, num_city, e):
        """生成num_total条路线"""
        start_index = 0
        result = []
        for i in range(num_total):
            # rest里存除start_index以及终点之外的点
            rest = [x for x in range(0, e)]
            # 如果start_index已经取到到每个点，那么随机生成一个start_index，
            # 把start_index对应的结果加到result中，最终result会生成num_total个结果
            if start_index >= e:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)

            # 从start_index出发每次都找最近的点，形成一条路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                # 找离current最近的点tmp_choose，然后current = tmp_choose，继续找最近点
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)

            result_one.append(e)  # 为路径加上终点
            result.append(result_one)  # 添加到种群中
            start_index += 1
        return result

    def compute_pathlen(self, path, dis_mat, back=False):
        """计算路径长度,默认不返回起点"""
        a = path[0]
        b = path[-1]
        if back:
            result = dis_mat[a][b]
        else:
            result = 0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_adp(self, fruits):
        """计算种群适应度:适应度用距离表示，距离越短表示适应度越高"""
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                # 这里会进入pdb调试环境，可以忽略
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        """变异方式：合并两个结果，倒序后截断"""
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        """交叉生成两条新路径"""
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        # 在城市编号中随机选两个数
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y与x冲突的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # 交叉，交换x与y中的切片数据
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        """保留一定比例的种群，记录路径和适应度"""
        sort_index = np.argsort(-scores).copy() # 适应度从大到小排序并记录索引
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]  # 保留ga_choose_ratio比例的种群
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        """轮盘赌方式对父代进行选择,返回选择两条路径"""
        # 算适应度占比
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        """对gene中选定的片段进行变异操作：此处的变异操作是倒序"""
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]  # 倒序
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        """获取最优路径与最佳适应度"""
        # 生成指定数量的种群，并计算种群适应度
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式选择两个父代
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 对父代进行交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            # 算适应度
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None  # 存路径顺序
        best_score = -math.inf  # 存路径长度
        self.best_record = []  # 存每轮迭代的路径长度
        for i in range(1, self.iteration + 1):
            # 获取最优秀的路径以及最佳适应度
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)  # 将适应度转回到距离
            # 更新
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1. / best_score)  # best_record记录多轮迭代的路径距离
            if i % 100 == 0:
                print(i, 1. / best_score)
        # print(1. / best_score)
        # return self.location[BEST_LIST], 1. / best_score
        return BEST_LIST, 1. / best_score


def read_tsp(path):
    """读取数据"""
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


def compute_dis_mat(num_city, location):
    """算城市间的距离矩阵"""
    dis_mat = np.zeros((num_city, num_city))
    for i in range(num_city):
        for j in range(num_city):
            if i == j:
                dis_mat[i][j] = np.inf
                continue
            a = location[i]
            b = location[j]
            tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
            dis_mat[i][j] = tmp
    return dis_mat


if __name__ == '__main__':
    # 获取城市的坐标位置
    data = read_tsp('data/st70.tsp')
    data = np.array(data)
    data = data[:11, 1:]

    # 算dis_mat
    num_city = len(data)
    dis_mat = compute_dis_mat(num_city, data)

    # 算串联所有点回到指定终点的最佳路径
    start = time.clock()  # 程序计时开始
    model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy(), dis_mat=dis_mat)
    Best_path, Best = model.run()
    end = time.clock()  # 程序计时结束

    print(f"程序的运行时间:{end - start}, 最短路径长：{Best}")
    print(f"节点顺序：{Best_path}")


    # # 加上一行因为会回到起点
    # fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    # axs[0].scatter(Best_path[:, 0], Best_path[:,1])
    # Best_path = np.vstack([Best_path, Best_path[0]])
    # axs[0].plot(Best_path[:, 0], Best_path[:, 1])
    # axs[0].set_title('规划结果')
    # iterations = range(model.iteration)
    # best_record = model.best_record
    # axs[1].plot(iterations, best_record)
    # axs[1].set_title('收敛曲线')
    # plt.show()
