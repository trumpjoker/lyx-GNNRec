import pandas as pd
import numpy as np
import math
import argparse
import random
from collections import Counter

'''
The original DoubanMovie data can be found at:
https://www.dropbox.com/s/tmwuitsffn40vrz/Douban.tar.gz?dl=0
'''

PATH_TO_DATA = 'data/Douban/'

SOCIAL_NETWORK_FILE = PATH_TO_DATA + 'socialnet/socialnet.tsv'
RATING_FILE = PATH_TO_DATA + 'movie/douban_movie.tsv'
max_length = 30
## 处理用户行为信息数据集
def process_rating(day=7): # segment session in every $day days.
    ## 读取douban_movie.tsv文件
    df = pd.read_csv(RATING_FILE, sep='\t', dtype={0:str, 1:str, 2:np.int32, 3: np.float32})
    ## 提取评分在[1,6]之间的数据
    df = df[df['Rating'].between(1,6,inclusive=True)]
    span_left = 1.2e9 ## 2008-01-11 05:20:00
    span_right = 1.485e9 ## 2017-01-21 20:00:00
    ## 提取Timestamp在[span_left, span_right]之间的数据
    df = df[df['Timestamp'].between(span_left, span_right, inclusive=True)]

    min_timestamp = df['Timestamp'].min()
    ## math.floor返回数字的下舍整数。表示根据min_timestamp划分time_id，七天内的数据time_Id相同
    time_id = [int(math.floor((t-min_timestamp) / (86400*day))) for t in df['Timestamp']]
    df['TimeId'] = time_id
    ## session_Id =UserId+TimeId
    session_id = [str(uid)+'_'+str(tid) for uid, tid in zip(df['UserId'], df['TimeId'])]
    df['SessionId'] = session_id
    print('Statistics of user ratings:')
    ## format为格式化函数，将数据按默认顺序存在大括号中
    ## nunique() 返回唯一值的数量
    print('\tNumber of total ratings: {}'.format(len(df)))  ## 数据总量：8483267
    print('\tNumber of users: {}'.format(df.UserId.nunique())) ## 用户数量：82901
    print('\tNumber of items: {}'.format(df.ItemId.nunique())) ## 项目数量：73677
    print('\tAverage ratings per user:{}'.format(df.groupby('UserId').size().mean()))
    return df
## 处理社交关系数据集
def process_social(): # read in social network.
    ## 读取socialnet.tsv文件
    net = pd.read_csv(SOCIAL_NETWORK_FILE, sep='\t', dtype={0:str, 1: str})
    ## 删除特定的列['Follower', 'Followee']的重复行
    net.drop_duplicates(subset=['Follower', 'Followee'], inplace=True)
    ##
    friend_size = net.groupby('Follower').size()
    #net = net[np.in1d(net.Follower, friend_size[friend_size>=5].index)]
    print('Statistics of social network:')
    ## 输出社交网络中的用户数量：Follower->112679
    ## 输出社交网络中边的数量：1758302
    print('\tTotal user in social network:{}.\n\tTotal edges(links) in social network:{}.'.format(\
        net.Follower.nunique(), len(net)))
    ## 一个用户的平均朋友数目：15
    print('\tAverage number of friends for users: {}'.format(net.groupby('Follower').size().mean()))
    return net
## 将用户和项目id重新划分，用户从0开始，项目从1开始
def reset_id(data, id_map, column_name='UserId'):
    ## 通过id_map表中的UserId找到对应的新索引，即mapped_id
    mapped_id = data[column_name].map(id_map)
    ## 将新的mapped_id放到UserId中
    data[column_name] = mapped_id
    if column_name == 'UserId':
        ## 如果列名为 'UserId‘，则加上Session_id
        session_id = [str(uid)+'_'+str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
        data['SessionId'] = session_id
    return data
## 筛选数据集中的会话、项目、用户，并且以周为单位时间划分数据集
def split_data(day): #split data for training/validation/testing.
    ## 处理用户-项目数据，返回df_data
    df_data = process_rating(day)
    ## 处理用户社交数据，返回df_net
    df_net = process_social()
    print("\n(intial) Number of social links:",len(df_net))
    ## 因为data和net中的UseId和follower和followee不同，所以要筛选
    ## 筛选出df_net中 Follower 有UserId部分
    ## 筛选出df_net中 Followee 有UserId部分
    ## 筛选结果：df_net中存在的UserId必然对应Follower和Followee
    ## 筛选出df_data中 UserId 有 Follower部分
    ## 注意 只是筛选Follower 没有管Followee ？？？？ 难道只是以Follower划分？
    df_net = df_net.loc[df_net['Follower'].isin(df_data['UserId'].unique())]
    df_net = df_net.loc[df_net['Followee'].isin(df_data['UserId'].unique())]
    df_data = df_data.loc[df_data['UserId'].isin(df_net.Follower.unique())]
    
    #restrict session length in [2, max_length]. We set a max_length because too long sequence may come from a fake user.
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size')>1]
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size')<=max_length]
    #length_supports = df_data.groupby('SessionId').size()
    #df_data = df_data[np.in1d(df_data.SessionId, length_supports[length_supports<=max_length].index)]
    
    # split train, test, valid.
    tmax = df_data.TimeId.max()  ## 471
    ## session_max_times：依据sessionId（同一个用户在），取出每个会话内最大的TimeId  这不是SessID的后缀吗？？？
    session_max_times = df_data.groupby('SessionId').TimeId.max() ## 718147个会话，每个会话选出它的Time_Id


    # 以最大TimeID的26周前作为基准划分训练集，训练集是由Session_ID组成
    session_train = session_max_times[session_max_times < tmax - 26].index ## 682281
    session_holdout = session_max_times[session_max_times >= tmax - 26].index ## 35266
    # 由ID可以得到对应的训练集和hold_out集
    train_tr = df_data[df_data['SessionId'].isin(session_train)]
    holdout_data = df_data[df_data['SessionId'].isin(session_holdout)] 
    
    print('Number of train/test: {}/{}'.format(len(train_tr), len(holdout_data))) ## 2916327/134476
    # 筛选项目出现次数大于等于20的数据和会话长度大于1的数据
    train_tr = train_tr[train_tr['ItemId'].groupby(train_tr['ItemId']).transform('size')>=20]
    train_tr = train_tr[train_tr['SessionId'].groupby(train_tr['SessionId']).transform('size')>1]  ## 2717639
    
    print('Item size in train data: {}'.format(train_tr['ItemId'].nunique()))  ## 12591
    ## 统计每个Item出现的次数
    train_item_counter = Counter(train_tr.ItemId) ## dict{"Item_Id",出现次数}
    ## 选择训练集中出现次数大于等于50的ItemId
    to_predict = Counter(el for el in train_item_counter.elements() if train_item_counter[el] >= 50).keys() #7531
    print('Size of to predict: {}'.format(len(to_predict))) #7531
    
    # split holdout to valid and test.
    ## 将holdout数据集中，验证集和测试集五五开
    holdout_cn = holdout_data.SessionId.nunique() # 35266
    holdout_ids = holdout_data.SessionId.unique()
    np.random.shuffle(holdout_ids)
    valid_cn = int(holdout_cn * 0.5) # 17633
    session_valid = holdout_ids[0: valid_cn]
    session_test = holdout_ids[valid_cn: ]
    ## 获得打乱holdout后随机得到的验证集valid和测试集test
    valid = holdout_data[holdout_data['SessionId'].isin(session_valid)]
    test = holdout_data[holdout_data['SessionId'].isin(session_test)]
    ## 筛选在to_predict中出现的验证集数据
    ## 筛选会话长度大于1的会话
    valid = valid[valid['ItemId'].isin(to_predict)]
    valid = valid[valid['SessionId'].groupby(valid['SessionId']).transform('size')>1]
    ## 筛选在to_predict中出现的测试集数据
    ## 筛选会话长度大于1的会话
    test = test[test['ItemId'].isin(to_predict)]
    test = test[test['SessionId'].groupby(test['SessionId']).transform('size')>1]
    ## 拼接训练集 验证集 测试集
    total_df = pd.concat([train_tr, valid, test])
    ## 筛选出df_net中 Follower 有UserId部分
    ## 筛选出df_net中 Followee 有UserId部分
    df_net = df_net.loc[df_net['Follower'].isin(total_df['UserId'].unique())]
    df_net = df_net.loc[df_net['Followee'].isin(total_df['UserId'].unique())]
    ## 用户map 项目map。key值为原ID，value值从0开始（item是从1开始）
    user_map = dict(zip(total_df.UserId.unique(), range(total_df.UserId.nunique()))) # dict：26511 value从0开始
    item_map = dict(zip(total_df.ItemId.unique(), range(1, 1+total_df.ItemId.nunique()))) # dict：12591 value从1开始
    with open('user_id_map.tsv', 'w') as fout:
        for k, v in user_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    with open('item_id_map.tsv', 'w') as fout:
        for k, v in item_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    ## 用户map的数量、项目map的数量
    num_users = len(user_map) #26511
    num_items = len(item_map) #12591

    ## 通过user_map和item_map重写设定数据集中的用户和项目ID
    ## 将所有的数据集：总数据集、训练集、验证集、测试集重新设置ID
    reset_id(total_df, user_map)
    reset_id(train_tr, user_map)
    reset_id(valid, user_map)
    reset_id(test, user_map)
    reset_id(df_net, user_map, 'Follower')
    reset_id(df_net, user_map, 'Followee')
    reset_id(total_df, item_map, 'ItemId')
    reset_id(train_tr, item_map, 'ItemId')
    reset_id(valid, item_map, 'ItemId')
    reset_id(test, item_map, 'ItemId')
    ## nunique()返回的是唯一值的个数
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(train_tr),
                                                                                          train_tr.SessionId.nunique(),
                                                                                          train_tr.ItemId.nunique(),
                                                                                          train_tr.groupby(
                                                                                              'SessionId').size().mean()))
    print ('Valid set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(valid),
                                                                                          valid.SessionId.nunique(),
                                                                                          valid.ItemId.nunique(),
                                                                                          valid.groupby(
                                                                                              'SessionId').size().mean()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(test),
                                                                                         test.SessionId.nunique(),
                                                                                         test.ItemId.nunique(),
                                                                                         test.groupby(
                                                                                             'SessionId').size().mean()))

    ## user2sessions key值表示UserId value值表示会话Id，它划分了每个用户对应的会话
    user2sessions = total_df.groupby('UserId')['SessionId'].apply(set).to_dict() ## dict：26511
    user_latest_session = []
    for idx in range(num_users):
        ## 对于每个用户，取出其对应的会话sessions和latest空列表
        sessions = user2sessions[idx]
        latest = []
        for t in range(tmax+1):
            ## 对于每个用户，遍历每个会话
            if t == 0:
                latest.append('NULL')
            else:
                sess_id_tmp = str(idx) + '_' + str(t-1)
                if sess_id_tmp in sessions:
                    latest.append(sess_id_tmp)
                else:
                    ## 如果此sess_id没有出现，则此位置补充上一个位置的数据，会保存上一个数据
                    latest.append(latest[t-1])
        ## 将最近的会话放置于user_latest_session
        user_latest_session.append(latest)
    ## 将训练集 验证集 测试集 社交关系 保存在不同的文件中
    train_tr.to_csv('data/train.tsv', sep='\t', index=False)
    valid.to_csv('data/valid.tsv', sep='\t', index=False)
    test.to_csv('data/test.tsv', sep='\t', index=False)
    df_net.to_csv('data/adj.tsv', sep='\t', index=False)
    with open('data/latest_sessions.txt', 'w') as fout:
        ## 根据UserId来将每个用户的会话数据写入
        for idx in range(num_users):
            fout.write(','.join(user_latest_session[idx]) + '\n')

    print('Number of users:', num_users)
    print('Number of items:', num_items)
    print('Number of events:', len(train_tr)+len(valid)+len(test))
    print('Number of social links:',len(df_net))

if __name__ == '__main__':
    day = 7
    split_data(day)