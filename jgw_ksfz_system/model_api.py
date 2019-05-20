# coding=utf-8
# author:zyw
# qq:810308374
# python3
import tensorflow as tf
import  numpy as np
#from train_vgg import read_img
import os
import random
from skimage import io, transform, color, util

def read_img(need_img_size, data_path, add_noise = False):
    (h, w, c) = need_img_size
    #print('reading the image: %s' % (data_path))
    img = io.imread(data_path)
    if c == 1:
        img = color.rgb2gray(img)
    elif c == 3:
        pass
    else:
        assert False, "image channel is not set to 1 or 3"
    
    if add_noise:
        if random.sample([True,False],1)[0]:
            img = util.random_noise(img, mode='gaussian')
        if random.sample([True,False],1)[0]:
            img = util.random_noise(img, mode='s&p')
        if random.sample([True,False],1)[0]:
            img = transform.rotate(img, random.randint(1,3) * 90, resize=True)
    img = transform.resize(img, (h, w, c))
    return np.asarray(img, np.float32)
class Model_Rec(object):
    def __init__(self, need_img_size, model_type, batch_size = 64,confidence = 0.8):
        
        
        assert model_type in ["JGW_SIM","JW_SIM","JGW_JW_SIM"], "model_type must be one of JGW_SIM、JW_SIM、JGW_JW_SIM"
        self.model_type = model_type
        
        self._jw_label2img_datas = dict()
        self._jgw_label2img_datas = dict()
        self._jw_label2path_datas = dict()
        self._jgw_label2path_datas = dict()
        self._jw_label2img_datas["None"] = []
        self._jgw_label2img_datas["None"] = []
        self._jw_label2path_datas["None"] = []
        self._jgw_label2path_datas["None"] = []
        self.need_img_size = need_img_size
        self.batch_size = batch_size
        self.confidence = confidence

    def load_ckpt_and_var(self, model_file_path, restore_file_name):
        #self._saver = tf.train.Saver(tf.global_variables())
        self._sess = tf.Session()
        self._saver = tf.train.import_meta_graph(model_file_path)
        self._saver.restore(self._sess, restore_file_name)
        self._graph = tf.get_default_graph()

    def load_database(self, marked_dict_paths = None, marked_img_dirs = None, unmarked_img_dirs=None):
        h,w,c = self.need_img_size
        if marked_dict_paths != None and marked_img_dirs != None:
            for dict_path, img_dir in zip(marked_dict_paths, marked_img_dirs):
                if "jw" in dict_path:
                    l2i_dict = self._jw_label2img_datas
                    l2p_dict = self._jw_label2path_datas
                elif "jgw" in dict_path:
                    l2i_dict = self._jgw_label2img_datas
                    l2p_dict = self._jgw_label2path_datas
                with open(dict_path) as f:
                    for line in f:
                        items = line.strip().split("\t")
                        if items[1] in l2i_dict:
                            l2i_dict[items[1]].append(read_img((h, w/2, c), os.path.join(img_dir, items[0])))
                            l2p_dict[items[1]].append(os.path.join(img_dir, items[0]))
                        else:
                            l2i_dict[items[1]] = [read_img((h, w/2, c), os.path.join(img_dir, items[0]))]
                            l2p_dict[items[1]] = [os.path.join(img_dir, items[0])]
        if unmarked_img_dirs!= None and len(unmarked_img_dirs) > 0 and type([]) ==type(unmarked_img_dirs):
            for img_dir in unmarked_img_dirs:
                if "jw" in img_dir:
                    l2i_dict = self._jw_label2img_datas
                    l2p_dict = self._jw_label2path_datas
                elif "jgw" in img_dir:
                    l2i_dict = self._jgw_label2img_datas
                    l2p_dict = self._jgw_label2path_datas
                for dirpath,dirnames,filenames in os.walk(img_dir):
                    for filename in filenames:
                        l2i_dict["None"].append(read_img((h, w/2, c), os.path.join(dirpath,filename)))
                        l2p_dict["None"].append(os.path.join(dirpath,filename))
    def default_recognize(self, half_img_path):
        def _recognize(l2i_dict, l2p_dict, target_img, reverse = False):
            labels = list(set(l2p_dict))
            predict_list = []
            path_list = []
            label_list = []
            whole_imgs = []
            for label in labels:
                for base_img, base_path in zip(l2i_dict[label], l2p_dict[label]):
                    if reverse == False:
                        whole_imgs.append(np.concatenate((base_img, target_img), axis = 1))
                    else:
                        whole_imgs.append(np.concatenate((target_img, base_img), axis = 1))
                    path_list.append(base_path)
                    label_list.append(label)
                    if len(whole_imgs) == self.batch_size:
                        predict_list.extend(self.online_recognize(whole_imgs))
                        whole_imgs.clear()
            if len(whole_imgs) != 0:
                predict_list.extend(self.online_recognize(whole_imgs))
                whole_imgs.clear()
            return predict_list, path_list, label_list
        result = {"predicts":[],"paths":[],"labels":[]}
        h,w,c = self.need_img_size
        target_img = read_img((h, w/2, c), half_img_path)
        if self.model_type == "JGW_SIM":
            predict_list, path_list, label_list = _recognize(self._jgw_label2img_datas, self._jgw_label2path_datas, target_img)
            result["predicts"].extend(predict_list)
            result["paths"].extend(path_list)
            result["labels"].extend(label_list)
        elif self.model_type == "JW_SIM":
            predict_list, path_list, label_list = _recognize(self._jw_label2img_datas, self._jw_label2path_datas, target_img)
            result["predicts"].extend(predict_list)
            result["paths"].extend(path_list)
            result["labels"].extend(label_list)
        else:
            predict_list, path_list, label_list = _recognize(self._jgw_label2img_datas, self._jgw_label2path_datas, target_img)
            result["predicts"].extend(predict_list)
            result["paths"].extend(path_list)
            result["labels"].extend(label_list)
            predict_list, path_list, label_list = _recognize(self._jw_label2img_datas, self._jw_label2path_datas, target_img, True)
            result["predicts"].extend(predict_list)
            result["paths"].extend(path_list)
            result["labels"].extend(label_list)
        return result
    def online_recognize(self, whole_imgs):
        h,w,c = self.need_img_size
        input_op = self._graph.get_tensor_by_name("input:0")
        output_op = self._graph.get_tensor_by_name("output:0")
        dropout_op = self._graph.get_tensor_by_name("dropout_rate:0")
        training_op = self._graph.get_tensor_by_name("istraining:0")
        return self._sess.run(output_op,{input_op:np.reshape(whole_imgs, [-1, h, w, c]), dropout_op:1.0, training_op:False})
    def jgw_jw_dzhs_generate(self, save_filename = "./default_jgw_jw_hstc.txt"):
        jgw_labels = set(self._jgw_label2path_datas)
        jw_labels = set(self._jw_label2path_datas)
        same_labels = jgw_labels & jw_labels
        jgw_different_labels = list(jgw_labels - same_labels)
        jw_different_labels = list(jw_labels - same_labels)
        result = {"unmarked_jgw":{"jgw_paths":[], "jw_paths":[],"predicts":[],"labels":[]}, \
                    "unmarked_jw":{"jw_paths":[], "jgw_paths":[],"predicts":[],"labels":[]}}
        whole_imgs = []
        
        for unmarked_img, unmarked_path in zip(self._jgw_label2img_datas["None"], self._jgw_label2path_datas["None"]):
            
            marked_paths = []
            predict_list = []
            for label in jw_different_labels:
                for marked_img, marked_path in zip(self._jw_label2img_datas[label], self._jw_label2path_datas[label]):
                    result["unmarked_jgw"]["jgw_paths"].append(unmarked_path)
                    marked_paths.append(marked_path)
                    whole_imgs.append(np.concatenate((unmarked_img, marked_img), axis = 1))
                    result["unmarked_jgw"]["labels"].append(label)
                    if len(whole_imgs) == self.batch_size:
                        predict_list.extend(self.online_recognize(whole_imgs))
                        whole_imgs.clear()
            result["unmarked_jgw"]["jw_paths"].extend(marked_paths)
            if len(whole_imgs) != 0:
                predict_list.extend(self.online_recognize(whole_imgs))
                whole_imgs.clear()
            result["unmarked_jgw"]["predicts"].extend(predict_list)
        print("jgw completed")
        for unmarked_img, unmarked_path in zip(self._jw_label2img_datas["None"], self._jw_label2path_datas["None"]):
            
            marked_paths = []
            predict_list = []
            for label in jgw_different_labels:
                for marked_img, marked_path in zip(self._jgw_label2img_datas[label], self._jgw_label2path_datas[label]):
                    result["unmarked_jw"]["jw_paths"].append(unmarked_path)
                    marked_paths.append(marked_path)
                    whole_imgs.append(np.concatenate((unmarked_img, marked_img), axis = 1))
                    result["unmarked_jw"]["labels"].append(label)
                    if len(whole_imgs) == self.batch_size:
                        predict_list.extend(self.online_recognize(whole_imgs))
                        whole_imgs.clear()
            result["unmarked_jw"]["jgw_paths"].extend(marked_paths)
            if len(whole_imgs) != 0:
                predict_list.extend(self.online_recognize(whole_imgs))
                whole_imgs.clear()
            result["unmarked_jw"]["predicts"].extend(predict_list)
        print("jw completed")
        with open(save_filename, "w") as f:
            f.write("jgw:\n")
            for jgw_path, jw_path, predict, label in zip(result["unmarked_jgw"]["jgw_paths"], \
                                                        result["unmarked_jgw"]["jw_paths"], \
                                                        result["unmarked_jgw"]["predicts"], \
                                                        result["unmarked_jgw"]["labels"]):
                if predict[0]>predict[1]:
                    f.write("%s\t%s\t%f\t%s\n"%(jgw_path, jw_path, predict[0], label))
            f.write("jw:\n")
            for jw_path, jgw_path, predict, label in zip(result["unmarked_jw"]["jw_paths"], \
                                                        result["unmarked_jw"]["jgw_paths"], \
                                                        result["unmarked_jw"]["predicts"], \
                                                        result["unmarked_jw"]["labels"]):
                if predict[0]>predict[1]:
                    f.write("%s\t%s\t%f\t%s\n"%(jw_path, jgw_path, predict[0], label))

    def test(self, test_num = 10000):
        def _sample(datas1, datas2, num):
            select1 = random.sample(datas1*num, num)
            select2 = random.sample(datas2*num, num)
            return_datas = []
            for i1, i2 in zip(select1, select2):
                return_datas.append(np.concatenate((i1, i2), axis = 1))
            return return_datas
        def _sample_right_false(labels, other_labels, same_labels, dicts, other_dicts):
            right_imgs = []
            fake_imgs = []
            for _ in range(test_num):
                label = random.sample(same_labels, 1)[0]
                right_imgs.extend(_sample(dicts[label], other_dicts[label], 1))
            self.test_actual_right_num = test_num
            for _ in range(test_num):
                label1 = random.sample(labels, 1)[0]
                label2 = random.sample(set(other_labels) - set([label1]), 1)[0]
                fake_imgs.extend(_sample(dicts[label1], other_dicts[label2], 1))
            self.test_actual_fake_num = test_num
            self.test_actual_num = self.test_actual_right_num + self.test_actual_fake_num
            '''
            #right datas
            via_class_data_num = int(test_num/len(same_labels))+1
            self.test_actual_right_num = via_class_data_num * len(same_labels)
            for label in same_labels:
                right_imgs.extend(_sample(dicts[label], other_dicts[label], via_class_data_num))
            #fake datas
            if via_class_data_num > len(other_labels)-1:#jgw_jw这里可能会多忽略一些
                other_class_num = len(other_labels)-1
                via_class_data_num = int(via_class_data_num/(len(other_labels)-1)) + 1
            else:
                other_class_num = via_class_data_num
                via_class_data_num = 1
            self.test_actual_fake_num = via_class_data_num * len(labels) * other_class_num
            self.test_actual_num = self.test_actual_right_num + self.test_actual_fake_num
            for label in labels:
                select_other_labels = random.sample(other_labels - set([label]), other_class_num)
                for other_label in select_other_labels:
                    fake_imgs.extend(_sample(dicts[label], other_dicts[other_label], via_class_data_num))
            '''
            return right_imgs, fake_imgs
        def _recognize(imgs):
            iteration = int(len(imgs) / self.batch_size)
            predict = []
            for i in range(0,iteration*self.batch_size, self.batch_size):
                predict.extend(self.online_recognize(imgs[i:i+self.batch_size]))
            return predict
        def _accuracy(predicts, label):#label=0 right, label=1 fake
            tmp = np.argmax(predicts, axis=1)
            acc = len(tmp[tmp == label])/len(tmp)
            return acc
        jgw_labels = set(self._jgw_label2path_datas) - set(["None"])
        jw_labels = set(self._jw_label2path_datas) - set(["None"])
        same_labels = jgw_labels & jw_labels

        if self.model_type == "JGW_SIM":
            dicts = self._jgw_label2img_datas
            right_imgs, fake_imgs = _sample_right_false(jgw_labels, jgw_labels, jgw_labels, dicts, dicts)
        elif  self.model_type == "JW_SIM":
            dicts = self._jw_label2img_datas
            right_imgs, fake_imgs = _sample_right_false(jw_labels, jw_labels, jw_labels, dicts, dicts)
        else:
            labels = jgw_labels
            other_labels = jw_labels
            dicts = self._jgw_label2img_datas
            other_dicts = self._jw_label2img_datas
            right_imgs, fake_imgs = _sample_right_false(labels, other_labels, same_labels, dicts, other_dicts)
        print(self.test_actual_right_num, self.test_actual_fake_num, self.test_actual_num)
        right_predict_list = np.array(_recognize(right_imgs))
        print("right predict completed")
        fake_predict_list = np.array(_recognize(fake_imgs))
        print("fake predict completed")
        self.test_right_acc = _accuracy(right_predict_list, 0)
        self.test_fake_acc = _accuracy(fake_predict_list, 1)
        self.test_acc = (self.test_right_acc*self.test_actual_right_num + self.test_fake_acc*self.test_actual_fake_num)/self.test_actual_num
        print("test_right_acc:", self.test_right_acc, "\ntest_fake_acc:", self.test_fake_acc, "\ntest_acc:", self.test_acc)


if __name__ == "__main__":
    
    model = Model_Rec((128,128,1), "JGW_JW_SIM")
    
    model.load_ckpt_and_var("./model/vgg_jgw_jw_cycle.ckpt-2000000.meta","./model/vgg_jgw_jw_cycle.ckpt-2000000")
    model.load_database(["./data/jgw_label_test", "./data/jw_label_test"], ["./data/jgw", "./data/jw"])#,["./data/temp_jgw_unmarked/", "./data/temp_jw_unmarked/"])
    model.test(50000)
    '''
    model = Model_Rec((128,128,1), "JW_SIM")
    model.load_ckpt_and_var("./model/vgg_jw_cycle.ckpt-2000000.meta","./model/vgg_jw_cycle.ckpt-2000000")
    model.load_database(["./data/jw_label_test"], ["./data/jw"])
    model.test(50000)
    
    model = Model_Rec((128,128,1), "JGW_SIM")
    model.load_ckpt_and_var("./model/vgg_jgw_cycle.ckpt-2000000.meta","./model/vgg_jgw_cycle.ckpt-2000000")
    model.load_database(["./data/jgw_label_test"], ["./data/jgw"])
    model.test(50000)
    '''
    #model.jgw_jw_dzhs_generate()
    #model.default_recognize("./ui/tmp.gif")["predicts"]
