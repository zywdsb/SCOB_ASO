# coding=utf-8
# author:zyw
# qq:810308374
# python3
import os,shutil
from skimage import io

image_file_dir = "./image"
jgw_image_dir = "./jgw"
jw_image_dir = "./jw"
other_image_dir = "./other"
file_ext = ".gif"
def movefiles2dir(srcdir, dstdir, files):
    no_exists_files_inds = []
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)
    for afile in files: 
        srcfile = os.path.join(srcdir, afile)
        if not os.path.isfile(srcfile):
            ind = files.index(afile)
            no_exists_files_inds.append(ind)
            continue
        dstfile = os.path.join(dstdir, afile)
        shutil.move(srcfile,dstfile)
    return no_exists_files_inds
def del_no_exists_files_ind(files,labels, no_exists_files_inds):
    no_exists_files_inds.reverse()
    for ind in no_exists_files_inds:
        files.pop(ind)
        labels.pop(ind)
    return files,labels
def mklabelfile(file_name, datas, labels):
    data_indices = list(range(len(datas)))
    train_indices = []
    test_indices = []
    label_kinds = set(labels)
    print(file_name, ":", len(label_kinds))
    for label in label_kinds:
        the_label_data_indices = list(filter(lambda i: labels[i] == label, data_indices))
        train_indices.extend(the_label_data_indices[:int(len(the_label_data_indices)*3/4+1)])
        test_indices.extend(the_label_data_indices[int(len(the_label_data_indices)*3/4+1):])
    with open(file_name, "w") as f:
        for i in data_indices:
            f.write(datas[i] + "\t" + labels[i] + "\n")
    with open(file_name + "_train", "w") as f:
        for i in train_indices:
            f.write(datas[i] + "\t" + labels[i] + "\n")
    with open(file_name + "_test", "w") as f:
        for i in test_indices:
            f.write(datas[i] + "\t" + labels[i] + "\n")
    return label_kinds, train_indices, test_indices
def get_max_wh(image_dir):
    img_dirs = []
    max_w = 0
    max_h = 0
    delete_dirs = []
    for root, dirs, files in os.walk(image_dir):
        for name in files:
          img_dirs.append( name)
    for afile in img_dirs:
        #fp = open(os.path.join(image_dir,afile),'rb')
        try:
            img = io.imread(os.path.join(image_dir,afile))
            (w, h) = img.shape
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        except(OSError, ValueError):
            #fp.close()
            shutil.move(os.path.join(image_dir,afile),"./error_img/" + afile)
            print('Error, Path:',afile)
            delete_dirs.append(afile)
            
    return max_w, max_h, delete_dirs
def process_data():
    jgw_files = []
    jgw_labels = []
    jw_files = []
    jw_labels = []
    other_lang_files = []
    other_lang_labels = []

    max_w, max_h, delete_dirs = get_max_wh(image_file_dir)
    print(max_w, " ", max_h)
    with open("text.txt") as f:

        for line in f:
            items = line.strip().split("\t")
            if items[2] + file_ext in delete_dirs:
                continue
            if "j" in items[2]:
                jgw_labels.append(items[0])
                jgw_files.append(items[2] + file_ext)
            elif "b" in items[2]:
                jw_labels.append(items[0])
                jw_files.append(items[2] + file_ext)
            else:
                other_lang_labels.append(items[0])
                other_lang_files.append(items[2] + file_ext)
    print("jgw label:", len(jgw_files))
    no_exists_files_inds = movefiles2dir(image_file_dir, jgw_image_dir, jgw_files)
    print("no_exists_files_inds:", len(no_exists_files_inds))
    jgw_files, jgw_labels = del_no_exists_files_ind(jgw_files,jgw_labels, no_exists_files_inds)

    print("jw label:", len(jw_files))
    no_exists_files_inds = movefiles2dir(image_file_dir, jw_image_dir, jw_files)
    print("no_exists_files_inds:", len(no_exists_files_inds))
    jw_files, jw_labels = del_no_exists_files_ind(jw_files, jw_labels, no_exists_files_inds)

    print("jgw label:", len(jgw_files))
    no_exists_files_inds = movefiles2dir(image_file_dir, other_image_dir, other_lang_files)
    print("no_exists_files_inds:", len(no_exists_files_inds))
    other_lang_files, other_lang_labels = del_no_exists_files_ind(other_lang_files, other_lang_labels, no_exists_files_inds)


    jgw_label_kinds, jgw_train_indices, jgw_test_indices = mklabelfile("jgw_label", jgw_files, jgw_labels)
    jw_label_kinds, jw_train_indices, jw_test_indices = mklabelfile("jw_label", jw_files, jw_labels)
    #mklabelfile("other_label", other_lang_files, other_lang_labels)

    jgw_max_w, jgw_max_h, _ = get_max_wh(jgw_image_dir)
    jw_max_w, jw_max_h, _ = get_max_wh(jw_image_dir)

    with open("./info", "w") as f:
        f.write("共同种类："+str(len(jgw_label_kinds & jw_label_kinds))+"\n")
        f.write("甲骨文种类："+str(len(jgw_label_kinds))+"\n")
        f.write("甲骨文图片最大宽高："+str(jgw_max_w)+"\t"+str(jgw_max_h)+"\n")
        f.write("金文种类："+str(len(jw_label_kinds))+ "\n")
        f.write("金文图片最大宽高："+str(jw_max_w)+"\t"+str(jw_max_h)+"\n")
    with open("./same_labels", "w") as f:
        for label in (jgw_label_kinds & jw_label_kinds):
            f.write(label+"\n")
process_data()
