# coding=utf-8
# author:zyw
# qq:810308374
# python3
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
def split_chars(path):
    if not os.path.exists("./split_chars_dir"):
        os.mkdir("./split_chars_dir")
    img = io.imread(path)

    height, width, channal = img.shape
    #print(height, width, channal)
    for i in range(height):
        for j in range(width):
            img[i][j] = 0 if max(img[i][j]) else 255
            
    #print(img.shape)
    #io.imshow(img)
    #plt.show()
    data = np.array(img)

    min_val = 7    #设置最小的文字像素高度，防止切分噪音字符

    start_i = -1
    end_i = -1
    rowPairs = []    #存放每行的起止坐标

    #行分割
    for i in range(height):
        if not data[i].all() and start_i < 0: 
            start_i = i

        elif not data[i:i+2].all():
            end_i = i
        if (data[i].all() or i == height-1) and start_i >= 0:
            #print(start_i, end_i)
            if(end_i - start_i >= min_val):
                rowPairs.append((start_i, end_i))
            start_i, end_i = -1, -1

    #列分割
    start_j = -1
    end_j = -1
    min_val_word = 4  #最小文字像素长度
    number = 0        #分割后保存编号

    expand_height_len = 4
    expand_width_len = 6

    #print(rowPairs)
    def expand_num(pos, max_num):
        if pos<0:
            return 0
        elif pos>max_num:
           return max_num
        else:
            return pos
            
    def rejudge_height(start, end, start_j, end_j):
        start_i = -1
        end_i = -1
        for i in range(start, end):
            if not data[i, start_j: end_j].all() and start_i < 0: 
                start_i = i

            elif not data[i:i+2, start_j: end_j].all():
                end_i = i
            if (data[i, start_j: end_j].all() or i == end-1) and start_i >= 0:
                if(end_i - start_i >= min_val):
                    return True
                start_i, end_i = -1, -1
        return False

    for start, end in rowPairs:
        for j in range(width):
            if not data[start: end, j].all() and start_j < 0:
                start_j = j
            elif not data[start: end, j:j+3].all():
                 end_j = j
            elif data[start: end, j].all() and start_j >= 0:
                if end_j - start_j >= min_val_word:
                    if not rejudge_height(start, end, start_j, end_j):
                        start_j, end_j = -1, -1
                        continue
                    tmp = Image.fromarray(\
                    data[expand_num(start-expand_height_len, height):expand_num(end+expand_height_len, height), \
                    expand_num(start_j-expand_width_len, width): expand_num(end_j+expand_width_len, width)])
                    #print(data[start:end, start_j: end_j].shape)
                    tmp.save("./split_chars_dir/" + '%d.gif' % number) 
                    number += 1
                start_j, end_j = -1, -1
    return number

if __name__ == "__main__":
    split_chars('./jgw_sent/B00002.png')
