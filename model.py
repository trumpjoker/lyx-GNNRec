import tensorflow as tf
import numpy as np
import math

from aggregators import *
from layers import Dense

class GNNRec(object):
    def __init__(self, args, support_sizes, placeholders):
        self.support_sizes = support_sizes
        # 确定邻居聚合方式
        ##self.out_size=100;self.batch_size=batch_size;
        if args.aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        # elif args.aggregator_type == "seq":
        #     self.aggregator_cls = SeqAggregator
        elif args.aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif args.aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif args.aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif args.aggregator_type == "attn":
            self.aggregator_cls = AttentionAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        ## 创建容器
        self.input_x = placeholders['input_x']
        self.input_y = placeholders['input_y']
        self.x_in = placeholders['x_in']
        self.x_out = placeholders['x_out']
        self.mask_y = placeholders['mask_y']
        # tf.cast()执行 tensorflow 中张量数据类型转换
        self.mask = tf.cast(self.mask_y, dtype=tf.float32)
        # 对self.mask求和
        self.point_count = tf.reduce_sum(self.mask)
        self.support_nodes_layer1 = placeholders['support_nodes_layer1']
        self.support_nodes_layer2 = placeholders['support_nodes_layer2']
        self.support_sessions_layer1 = placeholders['support_sessions_layer1']
        self.support_sessions_layer2 = placeholders['support_sessions_layer2']
        self.support_lengths_layer1 = placeholders['support_lengths_layer1']
        self.support_lengths_layer2 = placeholders['support_lengths_layer2']
        self.adj_in_layer1 = placeholders['adj_in_layer1']
        self.adj_in_layer2 = placeholders['adj_in_layer2']
        self.adj_out_layer1 = placeholders['adj_out_layer1']
        self.adj_out_layer2 = placeholders['adj_out_layer2']

        self.training = args.training
        self.concat = args.concat
        if args.act == 'linear':
            self.act = lambda x:x
        elif args.act == 'relu':
            self.act = tf.nn.relu
        elif args.act == 'elu':
            self.act = tf.nn.elu
        else:
            raise NotImplementedError
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        ##？？ 确定吗
        self.out_size = 100
        self.samples_1 = args.samples_1
        self.samples_2 = args.samples_2
        self.num_samples = [self.samples_1, self.samples_2]
        self.n_items = args.num_items
        self.n_users = args.num_users
        ## 这是self embedding
        self.emb_item = args.embedding_size
        ## 这是user embedding
        self.emb_user = args.emb_user
        self.max_length = args.max_length
        self.model_size = args.model_size
        self.dropout = args.dropout
        self.dim1 = args.dim1
        self.dim2 = args.dim2
        self.weight_decay = args.weight_decay
        self.global_only = args.global_only
        self.local_only = args.local_only
        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.dims = [self.hidden_size, args.dim1, args.dim2]
        self.dense_layers = []
        self.loss = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(args.learning_rate,
                                                            self.global_step,
                                                            args.decay_steps,
                                                            args.decay_rate,
                                                            staircase=True))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.build()

    ## 求用户朋友的长期静态兴趣
    def global_features(self):
        # 会话的全局嵌入  self.emb_user=100
        self.user_embedding = tf.get_variable('user_embedding', [self.n_users, self.emb_user],  ##[26511.50]
                                        initializer=tf.glorot_uniform_initializer())
        ## tf.nn.embedding_lookup(tensor,id)：即tensor就是输入的张量，id 就是张量对应的索引
        feature_layer1 = tf.nn.embedding_lookup(self.user_embedding, self.support_nodes_layer1) ## [10000,50]
        feature_layer2 = tf.nn.embedding_lookup(self.user_embedding, self.support_nodes_layer2) ## [1000,50]

        dense_layer = Dense(self.emb_user, 
                            self.hidden_size if self.global_only else self.hidden_size // 2,
                            act=tf.nn.relu,
                            dropout=self.dropout if self.training else 0.)
        self.dense_layers.append(dense_layer)
        feature_layer1 = dense_layer(feature_layer1)
        feature_layer2 = dense_layer(feature_layer2)
        return [feature_layer2, feature_layer1]

    #    修改的代码
    #    self.batch_size修改为support_lengths_layer
    # initial_state_layer1 = self.lstm_cell.zero_state(self.batch_size * self.samples_1 * self.samples_2,
    #                                                  dtype=tf.float32)
    # initial_state_layer2 = self.lstm_cell.zero_state(self.batch_size * self.samples_2, dtype=tf.float32)
    def ggnn_layer(self,inputs):
        support_sessions_layer, adj_in, adj_out, session_nums = inputs
        fin_state = tf.nn.embedding_lookup(self.embedding, support_sessions_layer)  ## fin_state: [10000,20,50]
        cell = tf.nn.rnn_cell.GRUCell(self.out_size,reuse=tf.AUTO_REUSE)
        with tf.variable_scope('gru'):
            ## 这里私自将step设置为1
            for i in range(1):
                fin_state = tf.reshape(fin_state, [session_nums,-1, self.out_size])
                ## fin_state_in =W_in*fin_state+b_in
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [session_nums, -1, self.emb_item ])
                ## fin_state_out =W_out*fin_state+b_out
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [session_nums, -1, self.emb_item ])
                ## av = concat(adj_in*fin_state_in,adj_out*fin_state_out,axis=-1)
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)
                # print("入度出度矩阵fin_state_in :\n",fin_state_in)  [10000/1000,20,50]
                ## av = concat(adj_in*fin_state_in,adj_out*fin_state_out,axis=-1)
                ## 最后一个维度数维度增加
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)

                state_output, fin_state =tf.nn.dynamic_rnn (cell=cell,
                                          inputs=tf.reshape(av,[session_nums,-1,2 * self.out_size]),
                                        # sequence_length=self.max_length,
                                          initial_state=cell.zero_state(batch_size=session_nums,dtype=tf.float32)
                                         )
                print("fin_state: \n",fin_state) #[10000,100]、[1000,100]
                # 返回最终状态h,-1所代表的含义是我们不用亲自去指定这一维的大小，函数会自动进行计算
        return tf.reshape(fin_state, [session_nums,self.out_size])

    def ggnn_layer2(self, inputs):
        support_sessions_layer, adj_in, adj_out, session_nums = inputs
        fin_state = tf.nn.embedding_lookup(self.embedding, support_sessions_layer)  ## fin_state: [1000,20,50]
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru2'):
            ## 这里私自将step设置为1
            for i in range(1):
                # fin_state = tf.reshape(fin_state, [self.batch_size,-1, self.out_size])
                ## fin_state_in =W_in*fin_state+b_in
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [session_nums, -1, self.emb_item ])
                ## fin_state_out =W_out*fin_state+b_out
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [session_nums, -1, self.emb_item ])
                ## av = concat(adj_in*fin_state_in,adj_out*fin_state_out,axis=-1)
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)
                # tf.nn.dynamic_rnn(cell=self.lstm_cell,
                #                                                       inputs=inputs_2,
                #                                                       sequence_length=self.support_lengths_layer2,
                #                                                       initial_state=initial_state_layer2,
                #                                                       dtype=tf.float32)

                output, fin_state = tf.nn.dynamic_rnn(cell=cell,
                                                            inputs=
                                                                tf.reshape(av,[session_nums,-1, 2*self.out_size]),
                                                            # sequence_length=self.max_length,
                                                            initial_state=cell.zero_state(batch_size=session_nums, dtype=tf.float32)
                                                            )

                # 返回最终状态h,-1所代表的含义是我们不用亲自去指定这一维的大小，函数会自动进行计算
                print("output: \n", output)  # [200,?,100]
        return output
    ## 求用户朋友的动态兴趣
    def local_features(self):
        ## 会话的局部嵌入

        '''
        Use the same ggnn in decode function
        '''

        ## GGNN model 融合用户朋友的动态偏好
        inputs_1 = self.support_sessions_layer1,self.adj_in_layer1,self.adj_out_layer1,self.batch_size * self.samples_1 * self.samples_2
        inputs_2 = self.support_sessions_layer2,self.adj_in_layer2,self.adj_out_layer2,self.batch_size * self.samples_2
        states1 = self.ggnn_layer(inputs_1)
        states2 = self.ggnn_layer(inputs_2)
        #outputs: shape[batch_size, max_time, depth] output此时没有利用！
        # 将状态1、2分别转换为 局部层1、2
        local_layer1 = states1
        local_layer2 = states2

        # Dense 类是一个全连接层类，dense_layer是它的一个实例
        dense_layer = Dense(self.hidden_size, 
                            self.hidden_size if self.local_only else self.hidden_size // 2,
                            act=tf.nn.relu,
                            dropout=self.dropout if self.training else 0.)
        self.dense_layers.append(dense_layer)
        # 将局部层1、2都加上全连接层
        local_layer1 = dense_layer(local_layer1)
        local_layer2 = dense_layer(local_layer2)
        # 返回局部层
        return [local_layer2, local_layer1]

    ## 得到用户朋友的总体嵌入
    def global_and_local_features(self):
        # 将局部层的两层特征，和全局层的两层特征分别融合
        global_feature_layer2, global_feature_layer1 = self.global_features()
        local_feature_layer2, local_feature_layer1 = self.local_features()
        global_local_layer1 = tf.concat([global_feature_layer1, local_feature_layer1], -1)
        global_local_layer2 = tf.concat([global_feature_layer2, local_feature_layer2], -1)
        return [global_local_layer2, global_local_layer1]

    def aggregate(self, hidden, dims, num_samples, support_sizes, 
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
                每一个卷积层的实例，长度为层数+1，每一个都是节点索引的嵌入
            input_features: the input features for each sample of various hops away.
                不同跳的每个实例的输入特征
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
                从输入层到输出层的隐藏层维度，长度为层数+1
            num_samples: list of number of samples for each layer. [1,5.10]
                每一层节点u的邻居节点的数量列表
            support_sizes: the number of nodes to gather information from for each layer. [1,5,50]
                每一层的节点u受到多少节点影响（既受当前层num_samples个直接邻居的影响，其邻居也受更先前深度num_samples个邻居的影响）
                support_size是到目前深度为止的各深度下num_samples的连乘积
            batch_size: the number of inputs (different for batch inputs and negative samples).
                输入数量（不同于批量输入和负样本数量）
        Returns:
            返回在此批次中，所有节点在最后一层结果中的隐藏层表示
            The hidden representation at the final layer for all nodes in batch
        """

        # length: number of layers + 1
        hidden = hidden
        # aggregators默认为None
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                # concat默认为false
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer，act=lambda x：x
                if layer == len(num_samples) - 1:
                    # aggregator_cls是选择的聚合方式 如果是最后一层，返回x，否则返回elu激励函数
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.dropout if self.training else 0., 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=self.act,
                            dropout=self.dropout if self.training else 0., 
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [self.batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                ## 对应 outputs=aggregator(self_vecs, neigh_vecs)
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    ## 得到用户的自身嵌入表示
    def decode(self):
        self.lstm_cell = lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        inputs_self= self.input_x,self.x_in,self.x_out,self.batch_size
        # print("self 的输入是:\n",inputs_self)
        outputs = self.ggnn_layer2(inputs_self)
        outputs = tf.transpose(outputs,perm=[1,0,2])
        # outputs: shape[max_length, batch_size, depth]
        # print("self 的输出是:",outputs)
        slices = tf.split(outputs, num_or_size_splits=self.max_length, axis=0)
        ## tf.squeeze(t,[0])将数据t中维度=1的压缩，这里指定为0维
        return [tf.squeeze(t,[0]) for t in slices]

    ## 目标用户的会话经过GGNN得到20个st，每个st都与社交层嵌入拼接，一共得到20个hidden
    ## hidden输入aggregate经过社交层聚合得到output，一共有20个output
    def step_by_step(self, features_0, features_1_2, dims, num_samples, support_sizes, 
            aggregators=None, name=None, concat=False, model_size="small"):
        self.aggregators = None
        outputs = []
        for feature0 in features_0:
            hidden = [feature0, features_1_2[0], features_1_2[1]]
            output1, self.aggregators = self.aggregate(hidden, dims, num_samples, support_sizes,
                                        aggregators=self.aggregators, concat=concat, model_size=self.model_size)
            outputs.append(output1)
        return tf.stack(outputs, axis=0)

    def build(self):
        self.embedding = embedding = tf.get_variable('item_embedding', [self.n_items, self.emb_item],\
                                        initializer=tf.glorot_uniform_initializer())   ## (12592, 100)

        print("embedding的维度：/n",self.embedding)
        ## 0层特征，表示用户本身的特征
        features_0 = self.decode() # features of zero layer nodes. 
        #outputs with shape [max_time, batch_size, dim2]
        ## 如果只设置融合全局嵌入：融合全局嵌入
        if self.global_only:
            features_1_2 = self.global_features()
        ## 如果只设置融合局部嵌入：融合局部嵌入
        elif self.local_only:
            features_1_2 = self.local_features()
        ## 融合局部嵌入和全局嵌入：
        else:
            features_1_2 = self.global_and_local_features()
        ## 将0 1 2层特征聚合
        outputs = self.step_by_step(features_0, features_1_2, self.dims, self.num_samples, self.support_sizes,
                                concat=self.concat)
        concat_self = tf.concat([features_0, outputs], axis=-1)

        # exchange first two dimensions.
        # transposed_outputs代表会话的最终嵌入
        self.transposed_outputs = tf.transpose(concat_self, [1,0,2])

        self.loss = self._loss()
        self.sum_recall = self._recall()
        self.sum_ndcg = self._ndcg()
        self.sum_mrr = self._mrr()

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                        for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
    
    def _loss(self):
        reg_loss = 0.
        xe_loss = 0.
        fc_layer = Dense(self.dim2 + self.hidden_size, self.emb_item, act=lambda x:x, dropout=self.dropout if self.training else 0.)
        self.dense_layers.append(fc_layer)
        ## logits = hn*zy，为每个项目被选中的概率
        self.logits = logits = tf.matmul(fc_layer(tf.reshape(self.transposed_outputs, [-1, self.dim2+self.hidden_size])), self.embedding, transpose_b=True)
        for dense_layer in self.dense_layers:
            for var in dense_layer.vars.values():
                reg_loss += self.weight_decay * tf.nn.l2_loss(var)
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                reg_loss += self.weight_decay * tf.nn.l2_loss(var)
        reshaped_logits = tf.reshape(logits, [self.batch_size, self.max_length, self.n_items])
        ## 计算logits 和 labels 之间的稀疏softmax 交叉熵
        xe_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                            logits=reshaped_logits,
                                                            name='softmax_loss')
        xe_loss *= self.mask
        return tf.reduce_sum(xe_loss) / self.point_count + reg_loss

    def _ndcg(self):
        # 推荐指标为ndcg的计算公式
        ## logits的转置
        predictions = tf.transpose(self.logits)
        ## 表示将张量input_y展平为一维
        targets = tf.reshape(self.input_y, [-1])
        ## 返回predictions的targets行，tf.diag_part()取出对角线元素,便是正样本对应的预测概率
        ## tf.expand_dims表示加了最后一个维度
        ## 对角线元素pred_values代表targets预测的概率
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(predictions, targets)), -1)
        ## tf.tile()扩展张量
        tile_pred_values = tf.tile(pred_values, [1, self.n_items-1])
        ## ranks表示正样本的排名 self.logits[:,1:] > tile_pred_values？？啥意思
        ranks = tf.reduce_sum(tf.cast(self.logits[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1
        ndcg = 1. / (log2(1.0 + ranks))
        mask = tf.reshape(self.mask, [-1])
        ndcg *= mask
        return tf.reduce_sum(ndcg)

    """
     tf.nn.in_top_k()函数:
     predictions: 你的预测结果（你的网络输出值）大小是预测样本的数量乘以输出的维度
     target:      实际样本类别的标签，大小是样本数量的个数
     k:           每个样本中前K个最大的数里面（序号）是否包含对应target中的值
     """
    def _recall(self):
         # 推荐指标为recall的计算公式
        predictions = self.logits
        targets = tf.reshape(self.input_y, [-1])
        recall_at_k = tf.nn.in_top_k(predictions, targets, k=20)
        recall_at_k = tf.cast(recall_at_k, dtype=tf.float32)
        mask = tf.reshape(self.mask, [-1])
        recall_at_k *=mask
        return tf.reduce_sum(recall_at_k)

    def _mrr(self):
        # 推荐指标为MRR@20的计算公式
        mrr = []
        ## 预处理数据
        predicitons = predictions = tf.transpose(self.logits)
        targets = tf.reshape(self.input_y, [-1])
        ## 取出每个会话target项目对应的概率pred_values
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(predictions, targets)), -1)
        tile_pred_values = tf.tile(pred_values, [1, 20])
        ## 每个会话中，与前20个项目进行比较，得到ranks
        logits_k = tf.nn.top_k(self.logits, 20).values
        ranks = tf.reduce_sum(tf.cast(logits_k > tile_pred_values, dtype=tf.float32), -1) + 1
        logits = tf.cast(self.logits, tf.float32)
        # 排名不在前20的项目通过recall_at_k取0
        recall_at_k = tf.nn.in_top_k(logits, targets, k=20)
        recall_at_k = tf.cast(recall_at_k, dtype=tf.float32)
        mrr = 1.0 / (ranks)  ## 注意！！
        mrr *= recall_at_k
        mask = tf.reshape(self.mask, [-1])
        mrr *=mask
        return tf.reduce_sum(mrr)

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
