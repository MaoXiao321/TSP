import math
import time
import matplotlib.pyplot as plt
import numpy as np


class DP(object):
    """动态规划法"""

    def __init__(self, num_city, data, dis_mat, e):
        self.num_city = num_city
        self.location = data
        self.dis_mat = dis_mat
        self.num_city = len(dis_mat)
        self.e = e  # 指定的终点

    def run(self):
        """动态规划过程"""
        # restnum存候选点集
        restnum = [x for x in range(self.num_city)]
        restnum.remove(0)
        restnum.remove(self.e)

        tmppath = [0, self.e]
        tmplen = self.dis_mat[0][self.e]
        while len(restnum) > 0:
            # c是候选点
            c = restnum[0]
            # 将候选点从restnum中删除
            restnum = restnum[1:]

            # 决定候选点的插入位置(只能在终点前插)
            insert = 0
            minlen = math.inf
            for i, num in enumerate(tmppath):
                # 获取已确定路径中最后一个到达点
                a = tmppath[-1] if i == 0 else tmppath[i - 1]
                # 获取已确定路径中第i个点
                b = tmppath[i]

                # 算候选点到a,b的距离
                tmp1 = 0 if i == 0 else self.dis_mat[a][c]
                tmp2 = self.dis_mat[c][b]

                curlen = (tmplen + tmp1 + tmp2) if i == 0 else (tmplen + tmp1 + tmp2 - self.dis_mat[a][b])
                if curlen < minlen:
                    minlen = curlen
                    insert = i

            tmppath = tmppath[0:insert] + [c] + tmppath[insert:]
            tmplen = minlen
        # return self.location[tmppath], tmplen
        return tmppath, tmplen


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

    e = len(dis_mat) - 1
    start = time.clock()  # 程序计时开始
    model = DP(num_city=data.shape[0], data=data.copy(), dis_mat=dis_mat, e=e)
    Best_path, Best = model.run()
    end = time.clock()  # 程序计时结束

    print(f"程序的运行时间:{end - start}, 最短路径长：{Best}")
    print(f"节点顺序：{Best_path}")

    # 显示规划结果
    Best_path = data[Best_path]
    plt.scatter(Best_path[:, 0], Best_path[:, 1])
    Best_path = np.vstack([Best_path, Best_path[0]])
    plt.plot(Best_path[:, 0], Best_path[:, 1])
    plt.title('st70:动态规划规划结果')
    plt.show()
