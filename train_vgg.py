# coding=utf-8
# author:zyw
# qq:810308374
# python3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from skimage import io, transform, color, util
import random
import time
import threading



cycle_train_datas_iterations = [10, 20, 50, 100, 200, 500, 1000]                  
cycle_train_datas_classify_accuracy = [.3, .35, .4, .45, .5, .55, .6]
cycle_train_datas_similarity_accuracy = [.65, .68, .71, .74, .76, .78, .81]
cycle_train_datas_A_files = list(map(lambda s:"./HanZiShuShuData/cycle_datas/jgw/A"+str(s), cycle_train_datas_iterations))
cycle_train_datas_B_files = list(map(lambda s:"./HanZiShuShuData/cycle_datas/jw/B"+str(s), cycle_train_datas_iterations))
end_cycle_datas_rank = 8
tf_config = tf.ConfigProto()  
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 分配50%  


input_h = 128
input_w = 128
input_channel = 1
batch_size = 4
num_iterations = 2000000

FLOAT_TYPE = tf.float32

def logger(log_name, info):
    with open(log_name, "a+") as f:
        f.write(info + "\n")
def load_dict_inf(dict_files, path = "./HanZiShuShuData"):
    one_dict = dict()
    for dict_file in dict_files:
        with open(os.path.join(path, dict_file)) as f:
            for line in f:
                items = line.strip().split("\t")
                if items[1] in one_dict:
                    one_dict[items[1]].append(items[0])
                else:
                    one_dict[items[1]] = [ items[0] ]
    return one_dict

def cycle_data_init(A_dict, B_dict, cycle_train_datas_iterations = cycle_train_datas_iterations):
    def cycle_gen(same_labels, A_dict, B_dict):
        A_select_label = random.sample(same_labels, 1)[0]
        while(A_select_label not in A_dict or A_select_label not in B_dict):
            A_select_label = random.sample(same_labels, 1)[0]
        B_img_path = random.sample(B_dict[A_select_label], 1)[0]
        A_img_path = random.sample(A_dict[A_select_label], 1)[0]
        return A_img_path, B_img_path, A_select_label
    cycle_datas_path = os.path.join("./HanZiShuShuData", "cycle_datas")
    same_labels = list(set(A_dict) & set(B_dict))
    if not os.path.exists(cycle_datas_path):
        os.mkdir(cycle_datas_path)
    for num in cycle_train_datas_iterations:
        with open(os.path.join(cycle_datas_path, "jgw/A"+str(num)), "w") as fa, \
                open(os.path.join(cycle_datas_path, "jw/B"+str(num)), "w") as fb:
            for _ in range(num):
                A_img_path, B_img_path, label = cycle_gen(same_labels, A_dict, B_dict)
                fa.write(A_img_path+"\t"+label+"\n")
                fb.write(B_img_path+"\t"+label+"\n")

def read_img(need_img_size, data_path):
    (h, w, c) = need_img_size
    #print('reading the image: %s' % (data_path))
    img = io.imread(data_path)
    if c == 1:
        img = color.rgb2gray(img)
        rows,cols=img.shape

    elif c == 3:
        pass
    else:
        assert False, "image channel is not set to 1 or 3"

    return img
def resize_img(need_img_size, img, add_noise = True):
    (h, w, c) = need_img_size
    if add_noise:
        if random.sample([True,False],1)[0]:
            img = transform.rotate(img, random.randint(1,3) * 90, resize=True)
        if random.sample([True,False],1)[0]:
            img = util.random_noise(img, mode='gaussian')
        if random.sample([True,False],1)[0]:
            img = util.random_noise(img, mode='s&p')
    img = transform.resize(img, (h, w, c))
    #io.imshow(img)
    #plt.show()
    return img

def vgg_net(height, width, channel, target_num, net_name):
    x = (tf.placeholder(FLOAT_TYPE, shape=[None, height, width, channel], name='input')-0.5)*2
    y = tf.placeholder(FLOAT_TYPE, shape=[None, target_num], name='labels_placeholder')
    istraining = tf.placeholder(tf.bool,shape=[],name="istraining")
    lr = tf.Variable(0.0, trainable=True)
    dropout_rate = tf.placeholder(FLOAT_TYPE,shape=[],name="dropout_rate")
    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=FLOAT_TYPE, stddev=0.1)
        return tf.Variable(initial, name=name, trainable=True)
    def conv2d(input, w, name):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')
    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=FLOAT_TYPE, shape=shape)
        return tf.Variable(initial, name=name, trainable=True)
    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, channel, 8], scope + "w")
        output_conv1_1 = tf.nn.elu(conv2d(x, kernel, scope), name=scope+"elu")

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 8, 8], scope + "w")
        output_conv1_2 = tf.nn.elu(conv2d(output_conv1_1, kernel, scope), name=scope+"elu")
    with tf.name_scope('conv1_3') as scope:
        kernel = weight_variable([3, 3, 8, 16], scope + "w")
        output_conv1_3 = tf.nn.elu(conv2d(output_conv1_2, kernel, scope), name=scope+"elu")
    with tf.name_scope('conv1_4') as scope:
        kernel = weight_variable([3, 3, 16, 16], scope + "w")
        output_conv1_4 = tf.nn.elu(conv2d(output_conv1_3, kernel, scope), name=scope+"elu")
    with tf.name_scope('conv1_5') as scope:
        kernel = weight_variable([3, 3, 16, 32], scope + "w")
        output_conv1_5 = tf.nn.elu(tf.layers.batch_normalization(conv2d(output_conv1_4, kernel, scope), training = istraining, name = scope+"bn"), name=scope+"elu")
    pool1 = pool_max(output_conv1_5)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 32, 32], scope + "w")
        output_conv2_1 = tf.nn.elu(conv2d(pool1, kernel, scope), name=scope+"elu")

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 32, 64], scope + "w")
        output_conv2_2 = tf.nn.elu(tf.layers.batch_normalization(conv2d(output_conv2_1, kernel, scope), training = istraining, name = scope+"bn"), name=scope+"elu")
    pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 64, 96], scope + "w")
        output_conv3_1 = tf.nn.elu(conv2d(pool2, kernel, scope), name=scope+"elu")

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 96, 128], scope + "w")
        output_conv3_2 = tf.nn.elu(tf.layers.batch_normalization(conv2d(output_conv3_1, kernel, scope), training = istraining, name = scope+"bn"), name=scope+"elu")
    pool3 = pool_max(output_conv3_2)

    # conv4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 128, 192], scope + "w")
        output_conv4_1 = tf.nn.elu(conv2d(pool3, kernel, scope), name=scope+"elu")

    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 192, 256], scope + "w")
        output_conv4_2 = tf.nn.elu(tf.layers.batch_normalization(conv2d(output_conv4_1, kernel, scope), training = istraining, name = scope+"bn"), name=scope+"elu")
    pool4 = pool_max(output_conv4_2)
    # conv5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 256, 512], scope + "w")
        output_conv5_1 = tf.nn.elu(conv2d(pool4, kernel, scope), name=scope+"elu")

    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512], scope + "w")
        output_conv5_2 = tf.nn.elu(tf.layers.batch_normalization(conv2d(output_conv5_1, kernel, scope), training = istraining, name = scope+"bn"), name=scope+"elu")
    pool5 = pool_max(output_conv5_2)
    #fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        kernel = weight_variable([shape, 4096], scope + "w")
        biases = bias_variable([4096], scope + "b")
        pool5_flat = tf.reshape(pool5, [-1, shape])
        output_fc6 = tf.nn.xw_plus_b(pool5_flat, kernel, biases)
        output_fc6 = tf.nn.elu(tf.layers.batch_normalization(tf.nn.dropout(output_fc6, keep_prob = dropout_rate), training = istraining, name = scope+"bn"), name=scope)
    #fc7
    with tf.name_scope('fc7') as scope:
        kernel = weight_variable([4096, 1024], scope + "w")
        biases = bias_variable([1024], scope + "b")
        output_fc7 = tf.nn.xw_plus_b(output_fc6, kernel, biases)
        output_fc7 = tf.nn.elu(tf.layers.batch_normalization(tf.nn.dropout(output_fc7, keep_prob = dropout_rate), training = istraining, name = scope+"bn"), name=scope)
    #fc8
    with tf.name_scope(net_name + 'fc8') as scope:
        kernel = weight_variable([1024, target_num], scope + "w")
        biases = bias_variable([target_num], scope + "b")
        output_fc8 = tf.nn.xw_plus_b(output_fc7, kernel, biases)
        output_fc8 = tf.layers.batch_normalization(tf.nn.dropout(output_fc8, keep_prob = dropout_rate), training = istraining, name = scope+"bn")
    finaloutput = tf.nn.softmax(output_fc8, name = "output")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_fc8, labels=y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
    new_lr = tf.placeholder(FLOAT_TYPE,shape=[],name="new_learning_rate")
    _lr_update = tf.assign(lr,new_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(cost)
    prediction_labels = tf.argmax(finaloutput, axis=1)
    read_labels = tf.argmax(y, axis=1)

    correct_prediction = tf.equal(prediction_labels, read_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, FLOAT_TYPE))

    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y=y,
        train_step=train_step,
        prediction_labels=prediction_labels,
        correct_prediction=correct_prediction,
        accuracy=accuracy,
        cost=cost,
        new_lr=new_lr,
        _lr_update=_lr_update,
        dropout_rate = dropout_rate,
        istraining = istraining,
    )


def test_network(sess, graph, datas, labels, target_num):
    accuracy, cost = sess.run([graph['accuracy'], graph['cost']], feed_dict={
                    graph['x']: np.reshape(datas, (-1, input_h, input_w, input_channel)),
                    graph['y']: np.reshape(labels, (-1, target_num)),
                    graph['dropout_rate']: 1.0,
                    graph['istraining']: False,
            
                })
    return accuracy, cost
def train_network(sess, graph, datas, labels, target_num):
    accuracy, cost, _ = sess.run([graph['accuracy'], graph['cost'],graph['train_step']], feed_dict={
        graph['x']: np.reshape(datas, (-1, input_h, input_w, input_channel)),
        graph['y']: np.reshape(labels, (-1, target_num)),
        graph['dropout_rate']: 0.6,
        graph['istraining']: True,
    })
    return accuracy, cost

def train_SIM(net, database, retraining = False, _dtype = "vgg_jgw_jw_all", begin_cycle_datas_rank = 1, lr_decay_mul_num = 512):
    print("train %s"%_dtype)
    graph = net(height=input_h, width=input_w, channel=input_channel, target_num=2, net_name = _dtype)
    test_dtype = _dtype + "_test"
    with tf.Session(config=tf_config) as sess:
        
        if retraining:
            model_file=tf.train.latest_checkpoint('%s_ckpt/'%_dtype)
            saver=tf.train.Saver(var_list = tf.global_variables())
            saver.restore(sess,model_file)
        else:
            init = tf.global_variables_initializer()
            saver=tf.train.Saver(var_list = tf.global_variables())
            sess.run(init)
        train_accuracy = 0
        train_cost = 0
        print_iterations_train_time = 0

        cycle_datas_rank = begin_cycle_datas_rank
        if cycle_datas_rank == end_cycle_datas_rank:
            dtype = _dtype + "_train"
            logger("%s.log"%_dtype,"change datas to all_data")
        else:
            dtype = _dtype + "_train_" + str(cycle_train_datas_iterations[cycle_datas_rank-1])
            logger("%s.log"%_dtype,"change datas to "+ "".join([str(i) for i in cycle_train_datas_iterations[0:cycle_datas_rank]]))
        for iteration in range(1, num_iterations+1):
            time_start=time.time()
            lr_decay = 0.5 ** min(np.sqrt(iteration*lr_decay_mul_num/num_iterations), 6)
            lr = sess.run(graph['_lr_update'],feed_dict={graph['new_lr']:lr_decay * 0.5})
            train_datas = []
            train_labels = []
            arandom = random.randint(1,batch_size-1)
            #print("arandom:", arandom)
            for _ in range(arandom):
                train_data, train_label = database.random_gen_data(dtype, True)
                train_datas.append(train_data)
                train_labels.append(train_label)
            for _ in range(batch_size - arandom):
                train_data, train_label = database.random_gen_data(dtype, False)
                train_datas.append(train_data)
                train_labels.append(train_label)
            shuffle_list = list(range(batch_size))
            random.shuffle(shuffle_list)
            train_datas = list(map(lambda indice:train_datas[indice], shuffle_list))
            train_labels = list(map(lambda indice:train_labels[indice], shuffle_list))
            if FLOAT_TYPE == tf.float16:
                train_datas = tf.cast(train_datas, FLOAT_TYPE)
                train_labels = tf.cast(train_labels, FLOAT_TYPE)
            accuracy, cost = train_network(sess, graph, train_datas, train_labels, target_num=2)
            print_iterations_train_time += time.time() - time_start
            train_accuracy += accuracy
            train_cost += cost
            if iteration%1000 is 0:
                train_accuracy = train_accuracy/1000
                logger("%s.log"%_dtype, 'Iteration - {:6d}:lr-{:6.5f},accuracy-{:6.4f},loss_on_train-{:6.5f},lost time-{:6.4f}'.format(iteration, lr, train_accuracy, train_cost/1000, print_iterations_train_time))
                print_iterations_train_time=0
                if cycle_datas_rank != end_cycle_datas_rank:
                    for consult in range(2, end_cycle_datas_rank+1):
                        if cycle_datas_rank < consult and train_accuracy > cycle_train_datas_similarity_accuracy[consult-2]:
                            cycle_datas_rank += 1
                            lr_decay_mul_num /= 1.6
                            if cycle_datas_rank == end_cycle_datas_rank:
                                dtype = _dtype + "_train"
                                logger("%s.log"%_dtype,"change datas to all_data")
                            else:
                                dtype = _dtype + "_train_" + str(cycle_train_datas_iterations[cycle_datas_rank-1])
                                logger("%s.log"%_dtype,"change datas to "+ "".join([str(i) for i in cycle_train_datas_iterations[0:cycle_datas_rank]]))

                train_accuracy = 0
                train_cost = 0
            if iteration%5000 is 0:
                test_accuracy = 0
                test_cost = 0
                for _ in range(100):
                    test_datas = []
                    test_labels = []
                    for _ in range(batch_size):
                        test_data, test_label = database.random_gen_data(test_dtype, True)
                        test_datas.append(test_data)
                        test_labels.append(test_label)
                        test_data, test_label = database.random_gen_data(test_dtype, False)
                        test_datas.append(test_data)
                        test_labels.append(test_label)
                    if FLOAT_TYPE == tf.float16:
                        test_datas = tf.cast(test_datas, FLOAT_TYPE)
                        test_labels = tf.cast(test_labels, FLOAT_TYPE)
                    accuracy, cost = test_network(sess, graph, test_datas, test_labels, target_num=2)
                    test_accuracy += accuracy
                    #print(accuracy)
                    test_cost += cost
                logger("%s.log"%_dtype,'accuracy:{:6.4f},loss_on_test:{:6.4f},'.format(test_accuracy/100, test_cost/100))
            if iteration%100000 is 0:
                saver.save(sess,'%s_ckpt/%s.ckpt'%(_dtype, _dtype),global_step=iteration)


class Database(object):
    def __init__(self, need_img_size):
        self._jw_train_label2img_datas = dict()
        self._jgw_train_label2img_datas = dict()
        self._jw_test_label2img_datas = dict()
        self._jgw_test_label2img_datas = dict()
        
        self._jw_train_cycle_datas = dict()
        self._jgw_train_cycle_datas = dict()
        self.need_img_size = need_img_size
    def load_database(self, marked_dict_paths = None, marked_img_dirs = None):
        if marked_dict_paths != None and marked_img_dirs != None:
            for dict_path, img_dir in zip(marked_dict_paths, marked_img_dirs):
                if "jw" in dict_path:
                    if "test" in dict_path:
                        l2i_dict = self._jw_test_label2img_datas
                    else:
                        l2i_dict = self._jw_train_label2img_datas
                elif "jgw" in dict_path:
                    if "test" in dict_path:
                        l2i_dict = self._jgw_test_label2img_datas
                    else:
                        l2i_dict = self._jgw_train_label2img_datas
                with open(dict_path) as f:
                    for line in f:
                        items = line.strip().split("\t")
                        if items[1] in l2i_dict:
                            l2i_dict[items[1]].append(read_img(self.need_img_size, os.path.join(img_dir, items[0])))
                        else:
                            l2i_dict[items[1]] = [read_img(self.need_img_size, os.path.join(img_dir, items[0]))]
    def load_cycle_database(self, cycle_train_datas_iterations, cycle_data_dict_paths = None, cycle_data_dirs = None):

        self.cycle_train_datas_iterations = cycle_train_datas_iterations
        tmp_cycle_train_datas_iterations = cycle_train_datas_iterations.copy()
        tmp_cycle_train_datas_iterations.reverse()
        for iteration in cycle_train_datas_iterations:
            self._jw_train_cycle_datas[str(iteration)] = dict()
            self._jgw_train_cycle_datas[str(iteration)] = dict()
        for iteration, dict_path, img_dir in zip(cycle_train_datas_iterations*2, cycle_data_dict_paths, cycle_data_dirs):
            if "jw" in dict_path:
                l2i_dicts = self._jw_train_cycle_datas
            elif "jgw" in dict_path:
                l2i_dicts = self._jgw_train_cycle_datas
            with open(dict_path) as f:
                for line in f:
                    items = line.strip().split("\t")
                    
                    for now_iter in tmp_cycle_train_datas_iterations:
                        if now_iter < iteration:
                            break
                        l2i_dict = l2i_dicts[str(now_iter)]
                        if items[1] in l2i_dict:
                            l2i_dict[items[1]].append(read_img(self.need_img_size, os.path.join(img_dir, items[0])))
                        else:
                            l2i_dict[items[1]] = [read_img(self.need_img_size, os.path.join(img_dir, items[0]))]
    def random_gen_data(self, dtype, IS_TRUE_DATA):
        h,w,c = self.need_img_size
        def judge_cycle_data(dtype):
            for iteri in self.cycle_train_datas_iterations:
                if str(iteri) in dtype:
                    return iteri
            return False
        if "all_train" in dtype:
            if "jgw_jw" in dtype:
                A_dict = self._jgw_train_label2img_datas
                B_dict = self._jw_train_label2img_datas
            elif "jgw" in dtype:
                A_dict = self._jgw_train_label2img_datas
                B_dict = self._jgw_train_label2img_datas
            elif "jw" in dtype:
                A_dict = self._jw_train_label2img_datas
                B_dict = self._jw_train_label2img_datas
        elif "cycle_train" in dtype:
            cycle_num = judge_cycle_data(dtype)
            if cycle_num == False:
                jgw_dict = self._jgw_train_label2img_datas
                jw_dict = self._jw_train_label2img_datas
            else:
                jgw_dict = self._jgw_train_cycle_datas[str(cycle_num)]
                jw_dict = self._jw_train_cycle_datas[str(cycle_num)]
            if "jgw_jw" in dtype:
                A_dict = jgw_dict
                B_dict = jw_dict
            elif "jgw" in dtype:
                A_dict = jgw_dict
                B_dict = jgw_dict
            elif "jw" in dtype:
                A_dict = jw_dict
                B_dict = jw_dict
            
        else:
            if "jgw_jw" in dtype:
                A_dict = self._jgw_test_label2img_datas
                B_dict = self._jw_test_label2img_datas
            elif "jgw" in dtype:
                A_dict = self._jgw_test_label2img_datas
                B_dict = self._jgw_test_label2img_datas
            elif "jw" in dtype:
                A_dict = self._jw_test_label2img_datas
                B_dict = self._jw_test_label2img_datas

        same_labels = list(set(A_dict) & set(B_dict))
        if IS_TRUE_DATA:
            A_select_label = random.sample(same_labels, 1)[0]
            while(A_select_label not in A_dict or A_select_label not in B_dict):
                A_select_label = random.sample(same_labels, 1)[0]
            B_data = resize_img((h, w/2, c), random.sample(B_dict[A_select_label], 1)[0])
            label = np.asarray([1,0], np.float32)
        else:
            A_select_label = random.sample(list(set(A_dict)), 1)[0]
            B_no_A_select_label_labels = list(filter(lambda i:i != A_select_label,list(set(B_dict))))
            B_select_label = random.sample(B_no_A_select_label_labels, 1)[0]
            B_data = resize_img((h, w/2, c), random.sample(B_dict[B_select_label], 1)[0])
            label = np.asarray([0,1], np.float32)
        A_data = resize_img((h, w/2, c), random.sample(A_dict[A_select_label], 1)[0])

        data = np.concatenate((A_data, B_data), axis = 1)
        

        return data, label

if __name__ == "__main__":
    dict_paths = ["jgw_label_train", "jgw_label_test", "jw_label_train", "jw_label_test"]
    jgw_img_dir = "./HanZiShuShuData/jgw"
    jw_img_dir = "./HanZiShuShuData/jw"
    marked_dict_paths = list(map(lambda s:"./HanZiShuShuData/"+str(s), dict_paths))
    marked_img_dirs = [jgw_img_dir, jgw_img_dir, jw_img_dir, jw_img_dir]
    #jgw_train_dict = load_dict_inf(["jgw_label_train"], "./HanZiShuShuData")
    #jw_train_dict = load_dict_inf(["jw_label_train"], "./HanZiShuShuData")
    #jgw_test_dict = load_dict_inf(["jgw_label_test"], "./HanZiShuShuData")
    #jw_test_dict = load_dict_inf(["jw_label_test"], "./HanZiShuShuData")
    #cycle_data_init(jgw_train_dict, jw_train_dict)

    #labels = get_AB_classification_labels(jgw_train_dict, jgw_train_dict)
    cycle_data_dict_paths = []
    
    cycle_data_dict_paths.extend(cycle_train_datas_A_files)
    cycle_data_dict_paths.extend(cycle_train_datas_B_files)
    cycle_data_dirs = [jgw_img_dir for _ in range(len(cycle_train_datas_A_files))]
    cycle_data_dirs.extend([jw_img_dir for _ in range(len(cycle_train_datas_B_files))])
    database = Database((input_h,input_w,input_channel))
    database.load_database(marked_dict_paths, marked_img_dirs)
    database.load_cycle_database(cycle_train_datas_iterations, cycle_data_dict_paths, cycle_data_dirs)
    #train_SIM(vgg_net, database, False, "vgg_jgw_cycle", 1, 64)

    coord=tf.train.Coordinator()
    
    threads = [
    #threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jgw_cycle", 1, 256,)),\
    threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jgw_all", end_cycle_datas_rank, 8,)),\
    #threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jw_cycle", 1, 256,)),\
    threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jw_all", end_cycle_datas_rank, 8,)),\
    #threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jgw_jw_cycle", 1, 256,)),\
    threading.Thread(target=train_SIM,args=(vgg_net, database, False, "vgg_jgw_jw_all", end_cycle_datas_rank, 8,))\
    ]
    for t in threads:
        t.start()
    coord.join(threads)

