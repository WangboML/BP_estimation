
import tensorflow as tf



def batched_input(features_seq,dia_bp_seq,sys_bp_seq,batch_size,time_steps,num_preprocess_threads,input_size,output_size):
    #
    feature,dia_bp=tf.train.slice_input_producer([features_seq,dia_bp_seq],shuffle=False)
    feature, sys_bp = tf.train.slice_input_producer([features_seq, sys_bp_seq], shuffle=False)
    features_and_dia=[]
    features_and_sys=[]
    for _ in range(num_preprocess_threads):
        # 这里可以用于输入数据的预处理或者特征变换
        feature=feature
        features_and_dia.append([feature,dia_bp])
        features_and_sys.append([feature, sys_bp])


    features_batch,dia_bp_batch=tf.train.batch_join(features_and_dia,batch_size=batch_size*time_steps,capacity=num_preprocess_threads*batch_size*2)
    features_batch,sys_bp_batch = tf.train.batch_join(features_and_sys, batch_size=batch_size*time_steps,capacity=num_preprocess_threads*batch_size *2)
    features_batch=tf.reshape(features_batch,[batch_size,time_steps,input_size])
    dia_bp_batch=tf.reshape(dia_bp_batch,[batch_size,time_steps,output_size])
    sys_bp_batch = tf.reshape(sys_bp_batch, [batch_size, time_steps, output_size])
    return features_batch,dia_bp_batch,sys_bp_batch

