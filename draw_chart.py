import matplotlib.pyplot as plt#约定俗成的写法plt
#首先定义两个函数（正弦&余弦）

import numpy as np
import re
import math
vgg_train_remain = [0] + list(range(4,2000,5))

vgg_test_remain = [0] + list(range(1,400,2))
vgg_train_iteration = [1] + list(range(5,2001,5))#201
vgg_test_iteration = [5] + list(range(10,2001,10))#201

res_train_remain = [0] + list(range(4,2000,5))
res_test_remain = [0] + list(range(4,400,5))
res_train_iter = [1] + list(range(5,2001,5))#401
res_test_iter = [5] + list(range(25,2001,25))#81
files = ["vgg_jgw_jw_all.log", "vgg_jgw_jw_cycle.log",\
         "vgg_jgw_all.log", "vgg_jgw_cycle.log", \
         "vgg_jw_all.log", "vgg_jw_cycle.log"]
out_train_test_names = ["JGW_JW_SIM all", "JGW_JW_SIM cycle", "JGW_SIM all", \
                  "JGW_SIM cycle", "JW_SIM all", "JW_SIM cycle"]
out_all_cycle_names = ["JGW_JW_SIM train", "JGW_JW_SIM test", "JGW_SIM train", \
                  "JGW_SIM test", "JW_SIM train", "JW_SIM test"]
def lst_getxy(x, y, order = 4):
    c=np.polyfit(x,y,order)#拟合多项式的系数存储在数组c中
    yy=np.polyval(c,x)#根据多项式求函数值
    f_liner=np.polyval(c,x)

    return x, f_liner
def save_train_test_plot(i, train_acc, test_acc, lr):
    fig= plt.figure()
    plt.title(out_train_test_names[i])
    plt.scatter(vgg_train_iteration, train_acc, label="raw_traindata",marker="|",color="red",s=6)
    x_,y_ = lst_getxy(vgg_train_iteration, train_acc)
    plt.plot(x_,y_,label="fit_traindata",linestyle="-", color="green") # 绘制拟合的曲线图
    
    plt.scatter(vgg_test_iteration, test_acc, label="raw_testdata",marker="*",color="blue",s=16)
    x_,y_ = lst_getxy(vgg_test_iteration, test_acc)
    plt.plot(x_,y_,label="fit_testdata",linestyle="-", color="yellow") # 绘制拟合的曲线图
    plt.plot(vgg_train_iteration, lr, label="lr", linewidth=1)
    plt.legend()
    plt.savefig(out_train_test_names[i], dpi=800)
    
def save_all_cycle_plot(i, x, all_acc, cycle_acc, all_lr, cycle_lr):
    fig= plt.figure()
    plt.title(out_all_cycle_names[i])
    plt.scatter(x, all_acc, label="raw_alldata",marker="|",color="red",s=6)
    x_,y_ = lst_getxy(x, all_acc)
    plt.plot(x_,y_,label="fit_alldata",linestyle="-", color="green") # 绘制拟合的曲线图
    
    plt.scatter(x, cycle_acc, label="raw_cycledata",marker="*",color="blue",s=16)
    x_,y_ = lst_getxy(x, cycle_acc)
    plt.plot(x_,y_,label="fit_cycledata",linestyle="-", color="yellow") # 绘制拟合的曲线图
    #plt.plot(vgg_train_iteration, all_lr, label="all_lr",marker="o", linewidth=1)
    #plt.plot(vgg_train_iteration, cycle_lr, label="cycle_lr",marker="^", linewidth=1)
    plt.legend()
    plt.savefig(out_all_cycle_names[i], dpi=800)
    





pattern = re.compile(r"\d+.\d+")
#"""


#"""
last_train_accs = []
last_test_accs = []
last_lr = []
vgg_test_accs = []
vgg_lr = []

for i, filename in zip(range(len(files)), files):
    now_train_acc = []
    now_test_acc = []
    now_lr = []
    with open(filename) as f:
        for line in f:
            result = pattern.findall(line)
            if line[0] == "I":
                now_train_acc.append(float(result[2]))
                now_lr.append(float(result[1]))
            elif line[0] == "a":
                now_test_acc.append(float(result[0]))
            elif line[0] == "c":
                pass

        now_train_acc = np.array(now_train_acc)[vgg_train_remain]
        now_test_acc = np.array(now_test_acc)[vgg_test_remain]/10
        now_lr = np.array(now_lr)[vgg_train_remain]
        save_train_test_plot(i, now_train_acc, now_test_acc, now_lr)
        if i%2 == 0:
            if i == 0:
               vgg_test_accs = now_test_acc
               vgg_lr = now_lr
            last_train_accs = now_train_acc
            last_test_accs = now_test_acc
            last_lr = now_lr
        else :
            save_all_cycle_plot(i-1, vgg_train_iteration, last_train_accs, now_train_acc, last_lr, now_lr)
            save_all_cycle_plot(i, vgg_test_iteration, last_test_accs, now_test_acc, last_lr, now_lr)







