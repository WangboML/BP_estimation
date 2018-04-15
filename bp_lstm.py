import os
from get_parser import FLAGS
import tensorflow as tf
from data_input import input_data
from data_input_pca import input_data_pca
from data_batch_input import batched_input
import matplotlib as plt
import numpy as np


def get_batch(train_data_features,train_data_dia_bp,train_data_sys_bp):
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)


    # xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE)
    features_batch = train_data_features[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE]  # 这个是输入数据
    dia_bp_batch = train_data_dia_bp[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE]
    sys_bp_batch = train_data_sys_bp[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE]  #这个是输出数据

    BATCH_START += TIME_STEPS
    return features_batch,dia_bp_batch,sys_bp_batch
    # 我们的输入数据必须是：(batch, step, input)，所以returned seq, res and xs: shape (batch, step, input)

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        #self.standard_scaler_DIA=standard_scaler_DIA
        #self.standard_scaler_SYS=standard_scaler_SYS
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('accuracy'):
            self.compute_accuracy()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # 设置 add_input_layer 功能, 添加 input_layer: #输入rnn之前先加一层线性变换，可选
    def add_input_layer(self, ):
        # 隐藏层的输入数据，首先要将三维数据变成二维数据
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws 形状是：(in_size, cell_size)  实现方法是下面的方程
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # 设置 add_cell 功能, 添加 cell, 注意这里的 self.cell_init_state, 因为我们在 training 的时候, 这个地方要特别说明
    def add_cell(self):

        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        dropout_cell=tf.contrib.rnn.DropoutWrapper(cell=cell,input_keep_prob=1.0)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([dropout_cell] * 2)
        #lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([dropout_cell] * 1)
        #outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_cell_fw,
                                                                #lstm_cell_bw, x,
                                                                #dtype=tf.float32)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            #self.cell_init_state = lstm_cell_bw.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
            # 因为time_step在输入数据的第二个维度，所以time_major=False
            # cell_outputs是一个list，包含所有的cell的输出；cell_final_state是一个batch的最后一个state，之后再训练就会取代
            # cell_init_state
    # 设置 add_output_layer 功能, 添加 output_layer:
    def add_output_layer(self):
        # 所接收的cellshape还是一个三维数据，要变二维 (batch * steps, cell_size)
        # 只要用二维才可以使用Wx_plus_b
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # 添加 RNN 中剩下的部分
    def compute_cost(self):
        # 对于整个batch，这个losses是每一个的loss的列表
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[tf.reshape(self.pred, [-1], name='reshape_pred')],
            targets=[tf.reshape(self.ys, [-1], name='reshape_target')],
            weights=[tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],  # # 损失的权重，在这里所有的权重都为1，也就是说不同batch和不同时刻的重要程度是一样的
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),  # 然后再把所有的losses加在一起再除以batch_size，这就是我们得到的总cost
                self.batch_size,                           # 之后再对cost进行minimize
                name='average_cost')
            tf.summary.scalar('cost', self.cost)
    def compute_accuracy(self):
        # 一个batch中的输出
        logits = [tf.reshape(self.pred, [-1], name='reshape_pred')]
        print(logits)
        targets = [tf.reshape(self.ys, [-1], name='reshape_target')]




    def msr_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)







if __name__ == "__main__":




    # 首先构造各种路径和相关参数
    data_path = FLAGS.data_path

    train_rate1 = FLAGS.train_rate1
    train_rate2 = FLAGS.train_rate2
    num_preprocess_threads=FLAGS.num_preprocess_threads # 特征预处理时的线程数

    # 我们先确定 RNN 的各种超参数(super-parameters)
    BATCH_START=0
    TIME_STEPS = 2  # backpropagation through time 的 time_steps（误差返回几步，每个序列的长度）
    BATCH_SIZE = 10
    INPUT_SIZE = 15  # 数据输入 size
    OUTPUT_SIZE = 1  # 数据输出 size
    CELL_SIZE = 10  # RNN 的 hidden unit size
    LR = 0.005  # learning rate

    data_dir = os.listdir(FLAGS.data_path)
            # 选择前20个作为模型的主要训练集合，其中对每一个数据集0.8作为训练，0.2作为测试
            # 对后10个作为模型的验证集合
            # data_dir1 = random.sample(data_dir, 10) # 下面就是这个随机得到的
    data_dir1 = ['data1.mat', 'data2.mat', 'data3.mat', 'data4.mat', 'data5.mat', 'data6.mat',
                         'data7.mat', 'data8.mat', 'data9.mat', 'realdata10.mat','data11.mat',
                         'data12.mat', 'data13.mat', 'data14.mat', 'data15.mat', 'data16.mat',
                         'data17.mat', 'data18.mat', 'data19.mat', 'data20.mat']
    data_dir2 = []
    for list in data_dir:
        if list not in data_dir1:
            data_dir2.append(list)
    data_dir1=['data2.mat']
    # 实现序列1的模型训练和测试
    for data in data_dir1:
        train_data, test_data, transform_data=input_data_pca(data_path, data, train_rate1)
        standard_scaler_DIA=transform_data['min_max_scaler_DIA']
        #standard_scaler_SYS=transform_data['min_max_scaler_SYS']
        train_iterion=len(train_data['features'])
        #features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
        pred_dia = []
        with tf.Session() as sess:

            model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
            #sess = tf.Session()

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
            writer.flush()
            # sess.run(tf.initialize_all_variables()) # tf 马上就要废弃这种写法
            # 替换成下面的写法:
            #features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],
                                                                   #train_data['sys_bp'])
            sess.run(tf.global_variables_initializer())

            #plt.ion()  # plt的交互模式启动，不会阻止主程序运行
            #plt.show()


            step = 0
            # 持续迭代

            while step * BATCH_SIZE < train_iterion:
                # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')

                #features_batch, dia_bp_batch, sys_bp_batch=batched_input(train_data['features'], train_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                                                                       #y2)
                # 首先对dia训练模型
                #features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                                                                   #train_data['sys_bp']))
                #sess.run([features_batch, dia_bp_batch])
                if step == 0:
                    # 初始化 data,第一步的时候我们不用定义state，它会使用init_state
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS,INPUT_SIZE)),
                        model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS,OUTPUT_SIZE)),
                    }
                else:
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS,INPUT_SIZE)),
                        model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS,OUTPUT_SIZE)),
                        model.cell_init_state: state  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换
                    }

                # 训练
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                pred_dia.append(pred)
                #print(standard_scaler_DIA.inverse_transform(pred))
                #print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    summary_str = sess.run(merged, feed_dict)
                    writer.add_summary(summary_str, step)



        print(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1))

            #with graph.as_default():
             #   saver = tf.train.Saver()

            # with tf.Session(graph=graph) as sess:
            #     sess.run(tf.global_variables_initializer(), feed_dict={embedding_placeholder: embedding_matrix})
            #     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            #
            #     test_state = sess.run(initial_state)
            #
            #     for ii, (x1, x2, batch_test_ids) in enumerate(
            #             get_test_batches(test_question1_features, test_question2_features, test_ids, batch_size), 1):
            #         feed = {question1_inputs: x1,
            #                 question2_inputs: x2,
            #                 keep_prob: 1,
            #                 initial_state: test_state
            #                 }
            #         test_state, distance = sess.run([question1_final_state, distance], feed_dict=feed)
            #print("Optimization Finished!")

