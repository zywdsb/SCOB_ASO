# coding=utf-8
# author:zyw
# qq:810308374
# python3
import os, shutil
import random
paths = []
if not os.path.exists("./temp_jgw_unmarked/"):
    os.makedirs("./temp_jgw_unmarked/")  
count = 30
with open("./jgw_label_test") as f:
    for line in f:
        stems = line.strip().split("\t")
        paths.append(stems[0])
random.shuffle(paths)
for i, srcfile in zip(range(len(paths)), paths):
    if i<count:
        shutil.copyfile("./jgw/"+srcfile,"./temp_jgw_unmarked/unmarkjgw%d.gif"%i)
    else:
        break

paths = []
random.shuffle(paths)
if not os.path.exists("./temp_jw_unmarked/"):
    os.makedirs("./temp_jw_unmarked/")  
with open("./jw_label_test") as f:
    for line in f:
        stems = line.strip().split("\t")
        paths.append(stems[0])
for i, srcfile in zip(range(len(paths)), paths):
    if i<count:
        shutil.copyfile("./jw/"+srcfile,"./temp_jw_unmarked/unmarkjw%d.gif"%i)
    else:
        break
