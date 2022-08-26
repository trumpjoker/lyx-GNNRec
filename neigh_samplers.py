from __future__ import division
from __future__ import print_function

import numpy as np

"""
Classes that are used to sample node neighborhoods
"""
# 邻居聚合函数
class UniformNeighborSampler(object):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, visible_time, deg):
        self.adj_info = adj_info
        self.visible_time = visible_time
        self.deg = deg

    def __call__(self, inputs):
        nodeids, num_samples, timeids, first_or_second, support_size = inputs
        adj_lists = []
        for idx in range(len(nodeids)):
            ## 遍历节点
            node = nodeids[idx]
            ##  // 整除(向小取整) 7//2=3   support_size:[1,5,50]
            timeid = timeids[idx // support_size]
            ## 用户节点node的关注者（邻居）
            adj = self.adj_info[node, :]
            neighbors = []
            for neighbor in adj:
                ## 对于用户node的每一个邻居：
                ## 如果是二阶邻居，则必须可初见时间 小于等于timeid才加入neighbors
                if first_or_second == 'second':
                    if self.visible_time[neighbor] <= timeid:
                        neighbors.append(neighbor)
                elif first_or_second == 'first':
                ##？ 如果是一阶邻居，则必须可初见时间 小于等于timeid 且deg大于0？
                    if self.visible_time[neighbor] <= timeid and self.deg[neighbor] > 0:
                        ## 遍历每个邻居的二阶邻居，
                        ## 如果其二阶邻居须可初见时间 小于等于timeid才加入neighbors
                        for second_neighbor in self.adj_info[neighbor]:
                            if self.visible_time[second_neighbor] <= timeid:
                                neighbors.append(neighbor)
                                break
            assert len(neighbors) > 0
            ## replace = True 在一次抽取中，抽取的样本可重复出现。replace = False 再一次抽取中，抽样的样本不可重复出现。
            ## 邻居的数量小于num_samples：则n次取样（可重复）
            ## 邻居的数量小于num_samples：则n次取样（不可重复）
            if len(neighbors) < num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=True)
            elif len(neighbors) > num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=False)
            ## 将每个节点的邻居信息都拼接在adj_lists列表中
            ## adj_lists是一个二维列表，第一维
            # 代表节点数，第二维记录了节点对应的每个邻居节点信息
            adj_lists.append(neighbors)
        ## 输出inputs对应的邻居节点列表
        return np.array(adj_lists, dtype=np.int32)
