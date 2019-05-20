# SCOB_ASO
甲骨文-金文相似度计算探究及甲骨文考释辅助系统设计
Study on Similarity Calculation between Oracle Bone with Bronze Inscriptions and Design of Auxiliary System for Oracle Bone Textual Research

train_vgg.py是训练部分代码，主要在vgg模型的基础上做出一些简单调整，未来希望可以尝试用残差网络、googlenet等拟合能力更强的网络来训练，所以已将训练接口写好。训练所需数据应存放在HanZiShuShuData文件夹下。

./jgw_ksfz_system/jgw_ksfz_ui.py是我设计的甲骨文考释辅助系统的GUI。因为在系统运行时，需要同时启用三个tensorflow模型，所以使用多线程来解决一些冲突。此外，未来希望，计算出每个已有图片数据的特征矢量，这样可以大大减少本系统启动及功能运行时，遍历所有数据的资源和时间损耗。

./jgw_ksfz_system/model_api.py包含模型测试以及系统调用的几个方法。

./jgw_ksfz_system/split_char.py主要是图像分割的部分代码。




本文模型结构和测试结果见results文件夹。

本项目数据见百度云盘（来自汉字叔叔——理查德·西尔斯的“汉字与词源”网站，不做商业用途，只用于研究）：
链接: https://pan.baidu.com/s/1aoCal8_fxcy0FIKrBWdU3w 提取码: 55n2
压缩包密码：zyw_luanren

本项目已训练模型见：



未来工作：
1.尝试用残差网络、googlenet等拟合能力更强的网络来训练；
2.优化系统：计算出每个已有图片数据的特征矢量；
3.尝试找出两种训练策略为什么有不同正样本准确率和负样本准确率（或者其实跟使用什么训练策略无关），

e-mail:810308374@qq.com
