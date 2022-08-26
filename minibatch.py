#coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import sys
from neigh_samplers import UniformNeighborSampler
from utils import *

np.random.seed(123)

class MinibatchIterator(object):
    def __init__(self, 
                adj_info, # in pandas dataframe
                latest_sessions,
                data, # data list, either [train, valid] or [train, valid, test].
                placeholders,
                batch_size,
                max_degree,
                num_nodes,
                max_length=20,
                samples_1_2=[10,5],
                training=True):
        self.num_layers = 2 # Currently, only 2 layer is supported.
        self.adj_info = adj_info
        self.latest_sessions = latest_sessions
        self.training = training
        ## 代入训练集 验证集 测试集数据
        self.train_df, self.valid_df, self.test_df = data
        self.all_data = pd.concat(data)
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.num_nodes = num_nodes
        self.max_length = max_length ## 20
        self.samples_1_2 = samples_1_2
        ## 记录每一层的support_size数量
        self.sizes = [1, samples_1_2[1], samples_1_2[1]*samples_1_2[0]]
        self.visible_time = self.user_visible_time()
        self.test_adj, self.test_deg = self.construct_test_adj()
        ## 如果是训练集集合
        if self.training:
            self.adj, self.deg = self.construct_adj()
            self.train_session_ids = self._remove_infoless(self.train_df, self.adj, self.deg)
            self.valid_session_ids = self._remove_infoless(self.valid_df, self.test_adj, self.test_deg)
            ## 创建一个邻居聚合函数实例（邻居聚合器）
            self.sampler = UniformNeighborSampler(self.adj, self.visible_time, self.deg)
        
        self.test_session_ids = self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
        ## padded_data是new_data，即会话的扩展序列，mask是会话对应的掩码
        self.padded_data, self.mask = self._padding_sessions(self.all_data)
        self.test_sampler = UniformNeighborSampler(self.test_adj, self.visible_time, self.test_deg)
        
        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0

    ## 记录每个用户第一次点击（即有记录）的时间为visible_time
    def user_visible_time(self):

        '''
            Find out when each user is 'visible' to her friends, i.e., every user's first click/watching time.
        '''
        visible_time = []
        for l in self.latest_sessions:
            ## l是一个用户的历史会话信息
            ## loc定位会话的timeId
            ## timeid表示该用户第一条历史会话的时间
            timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
            visible_time.append(timeid)
            assert timeid > 0 and timeid < len(l), 'Wrong when create visible time {}'.format(timeid)
        ## visible_time记录每个用户第一条会话的timeid
        return visible_time

    ## 输入数据，社交关系，用户的邻居数组，输出得到筛选后的会话id
    def _remove_infoless(self, data, adj, deg):
        '''
        Remove users who have no sufficient friends.
        '''
        ## data.loc[index,column]第一个行索引 第二个列索引
        ## 这里选出有社交联系的用户的社交数据
        data = data.loc[deg[data['UserId']] != 0]
        reserved_session_ids = []
        print('sessions: {}\tratings: {}'.format(data.SessionId.nunique(), len(data))) ##
        for sessid in data.SessionId.unique():
            ## 选择一个会话
            userid, timeid = sessid.split('_')
            userid, timeid = int(userid), int(timeid)
            cn_1 = 0
            for neighbor in adj[userid, : ]:
                ## 选出该会话对应的用户的一阶邻居
                if self.visible_time[neighbor] <= timeid and deg[neighbor] > 0:
                    ## 如果邻居的可见时间小于等于timeid（在此会话之前发生），并且邻居的邻居（二阶邻居）数量大于0，添加cn_2
                    cn_2 = 0
                    for second_neighbor in adj[neighbor, : ]:
                        ## 对于用户的二阶邻居，如果可见时间小于等于timeid，跳出二阶邻居的循环
                        ## 否则cn_2++
                        if self.visible_time[second_neighbor] <= timeid:
                            break
                        cn_2 += 1
                    ## 如果cn_2 小于等于max_degree，跳出循环
                    if cn_2 < self.max_degree:
                        break
                ## 每遍历一次邻居节点，cn_1++；
                cn_1 += 1
            ## 如果小于cn_1小于max_degree，保存此会话id
            if cn_1 < self.max_degree:
                reserved_session_ids.append(sessid)
        return reserved_session_ids

    ## 将每个长度不足max_length的会话补充零
    def _padding_sessions(self, data):
        '''
        Pad zeros at the end of each session to length self.max_length for batch training.
        '''
        ## （详见Test03）data是一个字典，key值为SessionId，value值是其item，根据交互时间排序
        data = data.sort_values(by=['TimeId']).groupby('SessionId')['ItemId'].apply(list).to_dict()
        new_data = {}
        data_mask = {}
        for k, v in data.items():
            ## data对应了一个会话，mask为掩码，其长度为max_length,全置为1
            mask = np.ones(self.max_length, dtype=np.float32)
            ## x表示会话去掉最后一个item
            ## y表示去掉第一个item
            x = v[:-1]
            y = v[1: ]
            assert len(x) > 0
            ## 设定补充的长度
            padded_len = self.max_length - len(x)
            ## 如果补充了多个0：
            if padded_len > 0:
                ## extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
                x.extend([0] * padded_len)
                y.extend([0] * padded_len)
                ## mask是对x（y）的掩码
                mask[-padded_len: ] = 0.
            v.extend([0] * (self.max_length - len(v)))
            x = x[:self.max_length]
            y = y[:self.max_length]
            v = v[:self.max_length]
            ## new_data和data_mask的行数都是k（原数据中的会话数目）
            ## new_data每一行有三列，分别是x y v（原始序列） new_data是对原始会话的扩展
            new_data[k] = [np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(v, dtype=np.int32)]
            ## mask
            data_mask[k] = np.array(mask, dtype=bool)
        return new_data, data_mask

    ## 输入当前批量（当前批量的会话id，用户的采样邻居节点，支持节点尺寸），得到对应每一层的支持节点列表、支持会话列表、支持会话长度
    def _batch_feed_dict(self, current_batch):
        '''
        Construct batch inputs.
        '''
        ## 当前批量分为会话id集合、支持节点的列表！！、支持节点尺寸数组[5,50]
        current_batch_sess_ids, samples, support_sizes = current_batch
        feed_dict = {}
        input_x = []
        input_y = []
        mask_y = []
        timeids = []
        for sessid in current_batch_sess_ids:
            ## 遍历当前批量的一个会话ID
            ## 分割会话为user和time两部分
            nodeid, timeid = sessid.split('_')
            ## 将timeid拼接到timeids中
            timeids.append(int(timeid))
            ## 将填充后的SessionId分为x y v（原始数据）
            x, y, _ = self.padded_data[sessid]
            mask = self.mask[sessid]
            input_x.append(x)
            input_y.append(y)
            mask_y.append(mask)

        x_in, x_out,  = [], []
        for input in input_x:
            node = np.unique(input)
            ## 将会话的去重序列填充0
            # items.append(node.tolist() + (self.max_length - len(node)) * [0])
            # items.append(input)
            ## 初始化邻接矩阵u_A
            u_A = np.zeros((self.max_length, self.max_length))
            ## 为什么要是len-1？？
            for i in np.arange(len(input) - 1):
                ## 如果下一个项目为0，则跳出循环
                if input[i + 1] == 0:
                    break
                u = np.where(node == input[i])[0][0]
                v = np.where(node == input[i + 1])[0][0]
                ## 节点u和v有连接，则元素值为1
                u_A[u][v] = 1
            ## 对列进行相加
            u_sum_in = np.sum(u_A, axis=0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            ## 对行进行相加
            u_sum_out = np.sum(u_A, axis=1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            ## 最后得到的A_in=A A_out=A.transpose
            x_in.append(u_A_in)
            x_out.append(u_A_out)

            ## support_lengths是会话长度列表
        # print("输入x是:",input_x)
        feed_dict.update({self.placeholders['input_x']: input_x})
        feed_dict.update({self.placeholders['input_y']: input_y})
        feed_dict.update({self.placeholders['x_in']: x_in})
        feed_dict.update({self.placeholders['x_out']: x_out})
        feed_dict.update({self.placeholders['mask_y']: mask_y})
        ## 将samples[1]和[2]填充到support_nodes_layer1和support_nodes_layer2中
        feed_dict.update({self.placeholders['support_nodes_layer1']: samples[2]})
        feed_dict.update({self.placeholders['support_nodes_layer2']: samples[1]})
        # prepare sopportive user's recent sessions.
        support_layers_session = []
        support_layers_length = []
        adj_in_layers = []
        adj_out_layers = []

        for layer in range(self.num_layers):  ## num_layers=2
            ## 遍历每一层 layer= 0、1
            start = 0
            t = self.num_layers - layer ## 2.1
            support_sessions = []
            support_lengths = []

            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            ## mask?  alias?   fin_state?   nasr_w1?
            for batch in range(self.batch_size): ##200
                ## 对于当前批量的每一个计算
                ## 得到需要计算的timeid列表
                timeid = timeids[batch]
                ## 1:samples[2][0:50],   2: samples[2][50,100]...  100:

                support_nodes = samples[t][start: start + support_sizes[t]]
                ## support_node是指用户id
                for support_node in support_nodes:  ##
                    ## 遍历该timeid下每一个支持节点
                    ## 得到每个支持节点(UserId)对应TimeId下的会话Id()
                    support_session_id = str(self.latest_sessions[support_node][timeid])
                    ##！ 得到该会话原序列（item）
                    support_session = self.padded_data[support_session_id][2]
                    # item =support_session
                    ## node为一个会话中去重后的节点列表
                    node = np.unique(support_session)
                    ## 将会话的去重序列填充0
                    # items.append(node.tolist() + (self.max_length - len(node)) * [0])
                    items.append(support_session)
                    ## 初始化邻接矩阵u_A
                    u_A = np.zeros((self.max_length,self.max_length))
                    ## 为什么要是len-1？？
                    for i in np.arange(len(support_session) - 1):
                        ## 如果下一个项目为0，则跳出循环
                        if support_session[i + 1] == 0:
                            break
                        u = np.where(node == support_session[i])[0][0]
                        v = np.where(node == support_session[i + 1])[0][0]
                        ## 节点u和v有连接，则元素值为1
                        u_A[u][v] = 1
                    ## 对列进行相加
                    u_sum_in = np.sum(u_A, axis=0)
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)
                    ## 对行进行相加
                    u_sum_out = np.sum(u_A, axis=1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)
                    ## 最后得到的A_in=A A_out=A.transpose
                    A_in.append(u_A_in)
                    A_out.append(u_A_out)

                    # 返回此批量中的：输出入度矩阵 出度矩阵 索引序列 去重重拍升序序列 掩码 标签序列
                    # return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
                    ## support_lengths是会话长度列表
                    length = np.count_nonzero(support_session)
                    ## support_sessions为items
                    support_sessions.append(support_session)
                    ## support_lengths为每个会话包含项目的长度
                    support_lengths.append(length)
                ## support_nodes的start向后移support_sizes[t]
                start += support_sizes[t]
            ## 将每一层的每个批量的支持节点的会话拼接在support_layers_session

            adj_in_layers.append(A_in)
            adj_out_layers.append(A_out)

            support_layers_session.append(support_sessions)
            support_layers_length.append(support_lengths)
        ## 第一层的支持会话喂入support_sessions_layer1 ->feed_dict
        ## 第二层的支持会话喂入support_sessions_layer2 ->feed_dict

        feed_dict.update({self.placeholders['support_sessions_layer1']:support_layers_session[0]}) ## [10000,20]
        feed_dict.update({self.placeholders['support_sessions_layer2']:support_layers_session[1]}) ## [1000,20]
        feed_dict.update({self.placeholders['support_lengths_layer1']:support_layers_length[0]})
        feed_dict.update({self.placeholders['support_lengths_layer2']:support_layers_length[1]})
        feed_dict.update({self.placeholders['adj_in_layer1']: adj_in_layers[0]})
        feed_dict.update({self.placeholders['adj_in_layer2']: adj_in_layers[1]})
        # print("adj_in_layer1 is: \n",adj_in_layers[0])
        # print("adj_in_layer2 is: \n",adj_in_layers[1])
        feed_dict.update({self.placeholders['adj_out_layer1']: adj_out_layers[0]})
        feed_dict.update({self.placeholders['adj_out_layer2']: adj_out_layers[1]})
        ## 将support_nodes_layer1 support_sessions_layer1 support_sessions_layer1 support_lengths_layer1 喂入feed_dict
        return feed_dict

    ## sample函数输入用户id列表，timeids，邻居聚合器
    ## sample函数最终返回包含各深度下采样节点的samples数组，与各深度下各点受支持节点数目的support_sizes数组。
    def sample(self, nodeids, timeids, sampler):
        '''
        Sample neighbors recursively. First-order, then second-order, ...
        '''
        ## 将节点Id输入samples列表
        samples = [nodeids]
        support_size = 1
        support_sizes = [support_size]
        first_or_second = ['second', 'first']
        for k in range(self.num_layers):
            ## 对于每一层 k=[0,1]
            ## 随着层数扩散，主键t减小
            t = self.num_layers - k - 1 ## [1，0]
            ## 对应 邻居聚合器中call函数的 nodeids, num_samples, timeids, first_or_second, support_size
            ## 返回inputs对应的邻居节点列表 node = adj_lists
            adj_lists = sampler([samples[k], self.samples_1_2[t], timeids, first_or_second[t], support_size])
            ## support_size数组表示当前节点u的embedding受多少节点信息的影响，是到目前深度为止的各深度下num_samples的连乘积support_size * batch_size。
            support_size *= self.samples_1_2[t] ## 5、50
            ## samples += reshape(adj_lists)
            samples.append(np.reshape(adj_lists, [support_size * self.batch_size,])) ## [1000,]、[10000,]
            support_sizes.append(support_size) ## [5,50]
        ## samples：[nodeids，nodeids对应的samples[1]，nodeids对应的samples[2]]
        ## support_sizes：支持节点尺寸：[5,50]
        return samples, support_sizes

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        '''
        Construct evaluation or test inputs.
        '''
        if val_or_test == 'val':
            start = self.batch_num_val * self.batch_size
            self.batch_num_val += 1
            data = self.valid_session_ids
        elif val_or_test == 'test':
            start = self.batch_num_test * self.batch_size
            self.batch_num_test += 1
            data = self.test_session_ids
        else:
            raise NotImplementedError
        
        current_batch_sessions = data[start: start + self.batch_size]
        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        samples, support_sizes = self.sample(nodes, timeids, self.test_sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    ## 构建当前批量的：会话Id集合  用户Id集合 TimeId，得到
    def next_train_minibatch_feed_dict(self):
        '''
        Generate next training batch data.
        '''
        ## 批量的起始序号设为batch_num * batch_size
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        ## 由remove_infoless函数得到的会话id，选取切片作为当前batch中的会话ID集合
        current_batch_sessions = self.train_session_ids[start: start + self.batch_size]
        ## node存储当前batch中UserID列表
        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        ## timeids存储当前batch中TimeID列表
        timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        ## 通过此批量的nodes获取其邻居
        samples, support_sizes = self.sample(nodes, timeids, self.sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    ## 构建用户的社交关系图，返回adj：用户社交关系列表、deg：用户的邻居长度列表
    def construct_adj(self):
        '''
        Construct adj table used during training.
        '''
        ## 初始化边集（shape=[n+1,max_degree]） 数组内数字全为n
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        ## 初始化全为0
        ## deg= n个0
        deg = np.zeros((self.num_nodes,))
        missed = 0
        for nodeid in self.train_df.UserId.unique():
            ## 对于每一个唯一的UserID:
            ## 选出其关注者Followee.作为其邻居
            neighbors = np.array([neighbor for neighbor in
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            ## deg[UserId]=邻居的数量
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            ## 如果邻居数量大于max_degree，则随机采样邻居时，不重复
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            ## 如果邻居数量小于max_degree，则随机采样邻居时，重复
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            ## 将每一个用户的邻居列表放在adj的一行
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        ## adj矩阵表示每个社交关系矩阵（出度矩阵），deg数组记录了每个用户的直接邻居数
        return adj, deg

    def construct_test_adj(self):
        '''
        Construct adj table used during evaluation or testing.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        data = self.all_data
        for nodeid in data.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def end(self):
        '''
        Indicate whether we finish a pass over all training samples.
        '''
        ## 左边是已经训练好的批量，右边是(n-1)个批量内的会话量，如果左边大于右边，即训练至最后一个批量，返回true
        return self.batch_num * self.batch_size > len(self.train_session_ids) - self.batch_size
    
    def end_val(self, val_or_test='val'):
        '''
        Indicate whether we finish a pass over all testing or evaluation samples.
        '''
        batch_num = self.batch_num_val if val_or_test == 'val' else self.batch_num_test
        data = self.valid_session_ids if val_or_test == 'val' else self.test_session_ids
        end = batch_num * self.batch_size > len(data) - self.batch_size
        if end:
            if val_or_test == 'val':
                self.batch_num_val = 0
            elif val_or_test == 'test':
                self.batch_num_test = 0
            else:
                raise NotImplementedError
        if end:
            self.batch_num_val = 0
        ## 如果训练完毕，返回true；否则返回false
        return end

    def shuffle(self):
        '''
        Shuffle training data.
        '''
        ## np.random.permutation（）对list随机排序
        self.train_session_ids = np.random.permutation(self.train_session_ids)
        self.batch_num = 0


if __name__ == '__main__':
    data = load_data('data/data')
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]
    minibatch = MinibatchIterator(adj_info,
                latest_per_user_by_time,
                [train_df, valid_df, test_df],
                None, #placeholders,
                batch_size=1,
                max_degree=50,
                num_nodes=len(user_id_map),
                max_length=20,
                samples_1_2=[10, 5])
    
