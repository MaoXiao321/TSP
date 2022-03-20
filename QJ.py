"""
代码功能：连接所有节点，并自主确定起点的最短路问题
此处暴力求解
"""

import pandas as pd
import json
from urllib.request import urlopen
import numpy as np
import time


def get_min_path(graph, s, e):
    """算从给定起点出发串联所有点回到中欣大厦的最短路径"""
    from itertools import permutations

    # vertex是候选点集合
    V = len(graph)  # 节点数
    vertex = list(np.arange(V))
    vertex.remove(s)  # 删掉起点
    vertex.remove(e)  # 删掉终点

    min_path = np.inf
    next_permutation = permutations(vertex)

    ls, path = [], (0, 0)
    # next_permutation用于对vertex中的节点进行全排列，i是返回的排列结果
    for i in next_permutation:
        current_pathweight = 0
        k = s
        # 算当前排列的总距离
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][e]
        # ls.append([i,current_pathweight])

        # 更新最短距离，存储路径
        if current_pathweight < min_path:
            min_path = current_pathweight
            path = i

    result = [s] + list(path)
    result.append(e)
    return result, min_path


def min_path(graph, e):
    """确定最优起点"""
    V = len(graph)
    min_path, path, origin = np.inf, (np.inf, np.inf), np.inf

    # 删掉终点
    ls = list(np.arange(V))
    ls.remove(e)
    for s in ls:
        path, current_pathweight = get_min_path(graph, s, e)
        if current_pathweight < min_path:
            min_path = current_pathweight
            result = path
    return result, min_path


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

    # 算串联所有点回到指定终点e的最佳路径
    e = len(dis_mat) - 1
    start = time.clock()  # 程序计时开始
    # result, min_path = get_min_path(dis_mat, 0, e)
    Best_path, Best = min_path(dis_mat, e)
    end = time.clock()  # 程序计时结束

    print(f"程序的运行时间:{end - start}, 最短路径长：{Best}")
    print(f"节点顺序：{Best_path}")
