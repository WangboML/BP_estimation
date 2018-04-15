import os
from get_parser import FLAGS
import tensorflow as tf
from data_input import input_data
from data_input_pca import input_data_pca
from data_batch_input import batched_input
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_batch(train_data_features, train_data_dia_bp, train_data_sys_bp):
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)

    #state_all=[]
    # xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE)
    features_batch = train_data_features[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE]  # 这个是输入数据
    dia_bp_batch = train_data_dia_bp[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE]
    sys_bp_batch = train_data_sys_bp[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE]  # 这个是输出数据

    BATCH_START += TIME_STEPS
    return features_batch, dia_bp_batch, sys_bp_batch
    # 我们的输入数据必须是：(batch, step, input)，所以returned seq, res and xs: shape (batch, step, input)


# def get_batch_test(test_data_features, test_data_dia_bp,test_data_sys_bp,step):
#
#     return test_data_features[step],test_data_dia_bp[step],test_data_sys_bp[step]


class LSTMRNN(object):

    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, LR):
        tf.reset_default_graph()
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.LR = LR

        # self.standard_scaler_DIA=standard_scaler_DIA
        # self.standard_scaler_SYS=standard_scaler_SYS
        with tf.variable_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
            #scope.refuse_variable()


        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
            #scope.refuse_variable()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
            #scope.refuse_variable()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('accuracy'):
            self.compute_accuracy()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.cost)

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
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1, state_is_tuple=True)
        dropout_cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=0.9)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([dropout_cell] * 1)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            # self.cell_init_state = lstm_cell_bw.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


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
            weights=[tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            # # 损失的权重，在这里所有的权重都为1，也就是说不同batch和不同时刻的重要程度是一样的
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),  # 然后再把所有的losses加在一起再除以batch_size，这就是我们得到的总cost
                self.batch_size,  # 之后再对cost进行minimize
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
        #initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        variable=tf.Variable(tf.random_normal(shape=shape,mean=0., stddev=1.),name=name)
        #return tf.get_variable(shape=shape, initializer=initializer, name=name)
        return variable

    def _bias_variable(self, shape, name='biases'):
        #initializer = tf.constant_initializer(0.1)
        #return tf.get_variable(name=name, shape=shape, initializer=initializer)
        bais=tf.Variable(tf.constant(0.1,shape=shape),name=name)
        return bais

def train_dia_model(data_path, data_dir1, train_rate1, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR,train_iterion):

    global model_number
    model_number=0
    coss_min=9
    #state_all = []
    #with tf.variable_scope('train'):
    for ii in range(train_iterion):


        for data in data_dir1:

            #cost_all = []
            model_number=model_number+1
            train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate1)
            standard_scaler_DIA = transform_data['min_max_scaler_DIA']
            # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
            train_iterion = len(train_data['features'])
            # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
            # pred_dia = []


            model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
            # sess = tf.Session()

            #merged = tf.summary.merge_all()
            #writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
            #writer.flush()
            #  # tf 马上就要废弃这种写法
            # 替换成下面的写法:
            # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],
            # train_data['sys_bp'])
            saver = tf.train.Saver(max_to_keep=3)
            loss_min_max=0
            if not os.path.exists('./ckpt_dia_model/'):

                with tf.Session() as sess:
                    # sess=tf.Session()
                    #
                    # plt.ion()  # plt的交互模式启动，不会阻止主程序运行
                    # plt.show()
                    loss=[]

                    sess.run(tf.initialize_all_variables())

                    step = 0
                    # 持续迭代

                    while step * BATCH_SIZE < train_iterion:
                        # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                        # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                        # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                        # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')

                        # features_batch, dia_bp_batch, sys_bp_batch=batched_input(train_data['features'], train_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                        features_batch, dia_bp_batch, _ = get_batch(train_data['features'], train_data['dia_bp'],
                                                                    train_data['sys_bp'])
                        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                        # y2)
                        # 首先对dia训练模型
                        # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                        # train_data['sys_bp']))
                        # sess.run([features_batch, dia_bp_batch])
                        if step == 0:
                            # 初始化 data,第一步的时候我们不用定义state，它会使用init_state
                            feed_dict = {
                                model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                                model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                            }
                        else:
                            feed_dict = {
                                model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                                model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                                model.cell_init_state: state,  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换

                            }

                        # 训练
                        _, cost, state, pred = sess.run(
                            [model.train_op, model.cost, model.cell_final_state, model.pred],
                            feed_dict=feed_dict)
                        loss.append(cost)
                        #cost_all.append(cost)
                        # pred_dia.append(pred)
                        # print(standard_scaler_DIA.inverse_transform(pred))
                        # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                        step += 1

                        # 打印 cost 结果
                        #if step % 5 == 0:
                            #print('cost: ', round(cost, 4))  # 每20步创建一个cost
                            #summary_str = sess.run(merged, feed_dict)
                            #writer.add_summary(summary_str, step)



                    loss_min=sum(loss)/len(loss)
                            # saver = tf.train.Saver([state,])
                    print('cost: %f step: %d ', (round(loss_min,4),model_number))
                    if loss_min<coss_min:
                        coss_min=loss_min
                        state_best=state
                        save_path = saver.save(sess, './ckpt_dia_model/dia-model.ckpt',global_step=model_number)
                    #tf.get_variable_scope().reuse_variables()
            else:
                with tf.Session() as sess:
                    # sess.run(tf.initialize_all_variables())
                    #saver=tf.train.import_meta_graph('./ckpt/dia-model.ckpt-0.meta')
                    model_file=tf.train.latest_checkpoint('./ckpt_dia_model/')
                    saver.restore(sess, model_file)
                    #state=saver.restore(state, model_file)
                    print("model restore"  )

                    loss = []


                    step = 0
                    # 持续迭代
                    global BATCH_START
                    BATCH_START = 0
                    while step * BATCH_SIZE < train_iterion:
                        # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                        # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                        # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                        # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')

                        # features_batch, dia_bp_batch, sys_bp_batch=batched_input(train_data['features'], train_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                        features_batch, dia_bp_batch, _ = get_batch(train_data['features'], train_data['dia_bp'],
                                                                    train_data['sys_bp'])
                        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                        # y2)
                        # 首先对dia训练模型
                        # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                        # train_data['sys_bp']))
                        # sess.run([features_batch, dia_bp_batch])

                        feed_dict = {
                            model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                            model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                            model.cell_init_state: state,  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换

                        }

                        # 训练
                        _, cost, state, pred = sess.run(
                            [model.train_op, model.cost, model.cell_final_state, model.pred],
                            feed_dict=feed_dict)

                        loss.append(cost)
                        #cost_all.append(cost)
                        # pred_dia.append(pred)
                        # print(standard_scaler_DIA.inverse_transform(pred))
                        # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                        step += 1

                        # 打印 cost 结果
                        # if step % 5 == 0:
                        #     #print('cost: ', round(cost, 4))  # 每20步创建一个cost
                        #     summary_str = sess.run(merged, feed_dict)
                        #     writer.add_summary(summary_str, step)
                    loss_min = sum(loss)/len(loss)
                            # saver = tf.train.Saver([state,])
                    print('cost: %f step: %d ', (round(loss_min,4), model_number))
                            # saver = tf.train.Saver([state,])
                    if loss_min<coss_min:
                        coss_min=loss_min
                        state_best = state
                        save_path = saver.save(sess, './ckpt_dia_model/dia-model.ckpt',global_step=model_number)
    state_best_txt_path='./state/state_best.txt'
    state_txt=pickle.dumps(state_best)
    if os.path.exists(state_best_txt_path):
        os.remove(state_best_txt_path)

    f=open(state_best_txt_path,'wb')
    f.write(state_txt)
    f.close()












def dia_train_test(data_path, data_dir2, train_rate2, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR):
    for data in data_dir2:
        # cost_all = []
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
        writer.flush()

        train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate2)
        standard_scaler_DIA = transform_data['min_max_scaler_DIA']

        train_iterion = len(train_data['features'])



        model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
        # sess = tf.Session()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            model_file = tf.train.latest_checkpoint('./ckpt_dia_model/')
            saver.restore(sess, model_file)
            step = 0
            # 持续迭代
            f=open('./state/state_best.txt','rb+')
            state_best_txt=f.read()
            f.close()
            state=pickle.loads(state_best_txt)

            while step * BATCH_SIZE < train_iterion:

                features_batch, dia_bp_batch,_= get_batch(train_data['features'], train_data['dia_bp'],
                                                            train_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])

                feed_dict = {
                    model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                    model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    model.cell_init_state: state,  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换

                }

                # 训练
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                # pred_dia.append(pred)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost


                    # saver = tf.train.Saver([state,])
            save_path = saver.save(sess, './ckpt_dia_train_test/dia-model.ckpt')
        with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            saver.restore(sess, './ckpt_dia_train_test/dia-model.ckpt')
            print("model restore")

            # standard_scaler_DIA = transform_data['min_max_scaler_DIA']
            # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
            # train_iterion = len(test_data['features'])
            # # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
            pred_dia = []
            real_dia = []
            # #
            step = 0
            global BATCH_START
            BATCH_START = 0
            # # # 持续迭代
            # # #BATCH_SIZE=0
            # TIME_STEPS = 1  # backpropagation through time 的 time_steps（误差返回几步，每个序列的长度）
            # BATCH_SIZE = 1
            #

            train_iterion = len(test_data['features'])
            while step * BATCH_SIZE < train_iterion:
                #     # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                #     # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                #     # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                #     # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')
                #
                #     features_batch, dia_bp_batch, sys_bp_batch=batched_input(test_data['features'], test_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, dia_bp_batch, _ = get_batch(test_data['features'], test_data['dia_bp'],
                                                            test_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])
                real_dia.append(dia_bp_batch)
                feed_dict = {
                    model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                    model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    model.cell_init_state: state  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换
                }

                # 训练
                cost, state, pred = sess.run([model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

                pred_dia.append(pred)

                # dia_bp=standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1, 1)).reshape(-1)
                # pred_dia.append(pred)
                # dia.append(dia_bp)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    #summary_str = sess.run(merged, feed_dict)
                    #writer.add_summary(summary_str, step)

            # dia_pred=standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1)
            # print(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1))
            # print(np.array(dia).reshape(-1))
            # plt.plot(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1),'g-*',np.array(dia).reshape(-1),'r-D')
            # #print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
            real_sys = np.array(real_dia).reshape(-1)
            pred_sys = np.array(pred_dia).reshape(-1)
            real_sys = standard_scaler_DIA.inverse_transform(real_sys.reshape(-1, 1))
            pred_sys = standard_scaler_DIA.inverse_transform(pred_sys.reshape(-1, 1))
            # plt.plot(real_dia, 'k-*')

            plt.plot(pred_sys, 'g-*', real_sys, 'r-D')
            # plt.plot(pred_dia, 'g-*')
            plt.show()





def train_sys(data_path, data_dir1, train_rate1, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR):
    for data in data_dir1:

        train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate1)
        standard_scaler_SYS = transform_data['min_max_scaler_SYS']
        # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
        train_iterion = len(train_data['features'])
        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
        # pred_dia = []


        model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
        # sess = tf.Session()

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
        writer.flush()
        #  # tf 马上就要废弃这种写法
        # 替换成下面的写法:
        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],
        # train_data['sys_bp'])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess=tf.Session()
            #
            # plt.ion()  # plt的交互模式启动，不会阻止主程序运行
            # plt.show()
            sess.run(tf.initialize_all_variables())

            step = 0
            # 持续迭代

            while step * BATCH_SIZE < train_iterion:
                # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')

                # features_batch, dia_bp_batch, sys_bp_batch=batched_input(train_data['features'], train_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, _, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],
                                                            train_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])
                if step == 0:
                    # 初始化 data,第一步的时候我们不用定义state，它会使用init_state
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                        model.ys: sys_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    }
                else:
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                        model.ys: sys_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                        model.cell_init_state: state,  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换

                    }

                # 训练
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                # pred_dia.append(pred)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    summary_str = sess.run(merged, feed_dict)
                    writer.add_summary(summary_str, step)

                    # saver = tf.train.Saver([state,])
            save_path = saver.save(sess, './ckpt/dia-model.ckpt')
        with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            saver.restore(sess, './ckpt/dia-model.ckpt')
            print("model restore")

            # standard_scaler_DIA = transform_data['min_max_scaler_DIA']
            # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
            # train_iterion = len(test_data['features'])
            # # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
            pred_sys = []
            real_sys = []
            # #
            step = 0
            global BATCH_START
            BATCH_START = 0
            # # # 持续迭代
            # # #BATCH_SIZE=0
            # TIME_STEPS = 1  # backpropagation through time 的 time_steps（误差返回几步，每个序列的长度）
            # BATCH_SIZE = 1
            #

            train_iterion = len(test_data['features'])
            while step * BATCH_SIZE < train_iterion:
                #     # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                #     # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                #     # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                #     # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')
                #
                #     features_batch, dia_bp_batch, sys_bp_batch=batched_input(test_data['features'], test_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, _, sys_bp_batch = get_batch(test_data['features'], test_data['dia_bp'],
                                                            test_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])
                real_sys.append(sys_bp_batch)
                feed_dict = {
                    model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                    model.ys: sys_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    model.cell_init_state: state  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换
                }

                # 训练
                cost, state, pred = sess.run([model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

                pred_sys.append(pred)

                # dia_bp=standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1, 1)).reshape(-1)
                # pred_dia.append(pred)
                # dia.append(dia_bp)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    #summary_str = sess.run(merged, feed_dict)
                    #writer.add_summary(summary_str, step)

            # dia_pred=standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1)
            # print(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1))
            # print(np.array(dia).reshape(-1))
            # plt.plot(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1),'g-*',np.array(dia).reshape(-1),'r-D')
            # #print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
            real_sys = np.array(real_sys).reshape(-1)
            pred_sys = np.array(pred_sys).reshape(-1)
            real_sys = standard_scaler_SYS.inverse_transform(real_sys.reshape(-1, 1))
            pred_sys = standard_scaler_SYS.inverse_transform(pred_sys.reshape(-1, 1))

            sys_pred_real={"pred":pred_sys,"real":real_sys}
            sys=pickle.dumps(sys_pred_real)
            dia_path='./sysdata/'
            f=open(dia_path+'data30.txt','wb')
            f.write(sys)
            f.close()

            # plt.plot(real_dia, 'k-*')
            plt.plot(pred_sys, 'g-*', real_sys, 'r-D')
            # plt.plot(pred_dia, 'g-*')
            plt.show()

def train_dia(data_path, data_dir1, train_rate2, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR):
    for data in data_dir1:

        train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate2)
        standard_scaler_DIA = transform_data['min_max_scaler_DIA']
        # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
        train_iterion = len(train_data['features'])
        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
        # pred_dia = []


        model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
        # sess = tf.Session()

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
        writer.flush()
        #  # tf 马上就要废弃这种写法
        # 替换成下面的写法:
        # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],
        # train_data['sys_bp'])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess=tf.Session()
            #
            # plt.ion()  # plt的交互模式启动，不会阻止主程序运行
            # plt.show()
            sess.run(tf.initialize_all_variables())

            step = 0
            # 持续迭代

            while step * BATCH_SIZE < train_iterion:
                # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')

                # features_batch, dia_bp_batch, sys_bp_batch=batched_input(train_data['features'], train_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, dia_bp_batch, _ = get_batch(train_data['features'], train_data['dia_bp'],
                                                            train_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])
                if step == 0:
                    # 初始化 data,第一步的时候我们不用定义state，它会使用init_state
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                        model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    }
                else:
                    feed_dict = {
                        model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                        model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                        model.cell_init_state: state,  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换

                    }

                # 训练
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                # pred_dia.append(pred)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    # summary_str = sess.run(merged, feed_dict)
                    # writer.add_summary(summary_str, step)

                    # saver = tf.train.Saver([state,])
            save_path = saver.save(sess, './ckpt_dia/dia-model.ckpt')
        with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            saver.restore(sess, './ckpt_dia/dia-model.ckpt')
            print("model restore")

            # standard_scaler_DIA = transform_data['min_max_scaler_DIA']
            # standard_scaler_SYS=transform_data['min_max_scaler_SYS']
            # train_iterion = len(test_data['features'])
            # # features_batch, dia_bp_batch, sys_bp_batch = get_batch(train_data['features'], train_data['dia_bp'],train_data['sys_bp'])
            pred_dia = []
            real_dia = []
            # #
            step = 0
            global BATCH_START
            BATCH_START = 0
            # # # 持续迭代
            # # #BATCH_SIZE=0
            # TIME_STEPS = 1  # backpropagation through time 的 time_steps（误差返回几步，每个序列的长度）
            # BATCH_SIZE = 1
            #

            train_iterion = len(test_data['features'])
            while step * BATCH_SIZE < train_iterion:
                #     # 提取每一个数据中的下一个batch作为输入,得到的数据shape都是[batch_size*time_steps,input_size]
                #     # x=tf.placeholder(dtype=tf.float32,shape=[None,15], name='x')
                #     # y1=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y1')
                #     # y2=tf.placeholder(dtype=tf.float32,shape=[None,1], name='y2')
                #
                #     features_batch, dia_bp_batch, sys_bp_batch=batched_input(test_data['features'], test_data['dia_bp'], train_data['sys_bp'], BATCH_SIZE, TIME_STEPS,num_preprocess_threads,INPUT_SIZE,OUTPUT_SIZE)
                features_batch, dia_bp_batch, _ = get_batch(test_data['features'], test_data['dia_bp'],
                                                            test_data['sys_bp'])
                # features_batch, dia_bp_batch, sys_bp_batch = get_batch(x, y1,
                # y2)
                # 首先对dia训练模型
                # features_batch, dia_bp_batch, sys_bp_batch=sess.run(get_batch(train_data['features'], train_data['dia_bp'],
                # train_data['sys_bp']))
                # sess.run([features_batch, dia_bp_batch])
                real_dia.append(dia_bp_batch)
                feed_dict = {
                    model.xs: features_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)),
                    model.ys: dia_bp_batch.reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE)),
                    model.cell_init_state: state  # 保持 state 的连续性，因此我们后期要更换，每一次更新完一个batch要更换
                }

                # 训练
                cost, state, pred = sess.run([model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

                pred_dia.append(pred)

                # dia_bp=standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1, 1)).reshape(-1)
                # pred_dia.append(pred)
                # dia.append(dia_bp)
                # print(standard_scaler_DIA.inverse_transform(pred))
                # print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
                step += 1

                # 打印 cost 结果
                if step % 5 == 0:
                    print('cost: ', round(cost, 4))  # 每20步创建一个cost
                    #summary_str = sess.run(merged, feed_dict)
                    #writer.add_summary(summary_str, step)

            # dia_pred=standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1)
            # print(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1))
            # print(np.array(dia).reshape(-1))
            # plt.plot(standard_scaler_DIA.inverse_transform(pred_dia).reshape(-1),'g-*',np.array(dia).reshape(-1),'r-D')
            # #print(standard_scaler_DIA.inverse_transform(dia_bp_batch.reshape(-1,1)))
            real_dia = np.array(real_dia).reshape(-1)
            pred_dia = np.array(pred_dia).reshape(-1)
            real_dia = standard_scaler_DIA.inverse_transform(real_dia.reshape(-1, 1))
            pred_dia = standard_scaler_DIA.inverse_transform(pred_dia.reshape(-1, 1))

            sys_pred_real={"pred":pred_dia,"real":real_dia}
            sys=pickle.dumps(sys_pred_real)
            dia_path='./diadata/'
            f=open(dia_path+'data30.txt','wb')
            f.write(sys)
            f.close()

            # plt.plot(real_dia, 'k-*')
            plt.plot(pred_dia, 'g-*', real_dia, 'r-D')
            # plt.plot(pred_dia, 'g-*')
            plt.show()


if __name__ == "__main__":

    # 首先构造各种路径和相关参数
    data_path = FLAGS.data_path

    train_rate1 = FLAGS.train_rate1
    train_rate2 = FLAGS.train_rate2
    num_preprocess_threads = FLAGS.num_preprocess_threads  # 特征预处理时的线程数

    # 我们先确定 RNN 的各种超参数(super-parameters)
    BATCH_START = 0
    TIME_STEPS = 1  # backpropagation through time 的 time_steps（误差返回几步，每个序列的长度）
    BATCH_SIZE = 1
    INPUT_SIZE = 10  # 数据输入 size
    OUTPUT_SIZE = 1  # 数据输出 size
    CELL_SIZE = 7  # RNN 的 hidden unit size
    LR = 0.008  # learning rate
    train_iterion = 100
    data_dir = os.listdir(FLAGS.data_path)
    # 选择前20个作为模型的主要训练集合，其中对每一个数据集0.8作为训练，0.2作为测试
    # 对后10个作为模型的验证集合
    # data_dir1 = random.sample(data_dir, 10) # 下面就是这个随机得到的
    data_dir1 = ['data1.mat','data2','data3.mat', 'data4.mat', 'data5.mat', 'data6.mat',
                 'data7.mat', 'data8.mat', 'data9.mat', 'data10.mat', 'data11.mat',
                 'data12.mat', 'data13.mat', 'data14.mat', 'data15.mat', 'data16.mat',
                 'data17.mat', 'data18.mat', 'data19.mat', 'data20.mat','realdata1.mat','realdata2',
                 'realdata3.mat', 'realdata4.mat', 'realdata5.mat', 'realdata6.mat',
                 'realdata7.mat', 'realdata8.mat', 'realdata9.mat', 'realdata10.mat']
    #data_dir1 = ['data6.mat']
    data_dir2 = ['realdata10.mat']

    #data_dir1 = ['data6.mat']
    # 实现序列1的模型训练和测试


    #train_dia(data_path, data_dir1, train_rate1, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR,train_iterion)
    #train_sys(data_path, data_dir2, train_rate2, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
    train_dia(data_path, data_dir2, train_rate2, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
    #dia_train_test(data_path, data_dir2, train_rate2, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)