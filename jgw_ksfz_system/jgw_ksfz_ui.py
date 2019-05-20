# coding=utf-8
# author:zyw
# qq:810308374
# python3


import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.font as tkFont
from PIL import Image, ImageTk
from io import BytesIO

from model_api import Model_Rec
import numpy as np

import threading
import time
import os
from split_char import split_chars



def tkimg_resized(img, w_box, h_box, keep_ratio=True):
    """对图片进行按比例缩放处理"""
    w, h = img.size

    if keep_ratio:
        if w > h:
            width = w_box
            height = int(h_box * (1.0 * h / w))

        if h >= w:
            height = h_box
            width = int(w_box * (1.0 * w / h))
    else:
        width = w_box
        height = h_box

    img1 = img.resize((width, height), Image.ANTIALIAS)
    tkimg = ImageTk.PhotoImage(img1)
    return tkimg


def image_label(frame, img, width, height, keep_ratio=True):
    """输入图片信息，及尺寸，返回界面组件"""
    if isinstance(img, str):
        _img = Image.open(img)
    else:
        _img = img
    lbl_image = tk.Label(frame, width=width, height=height)

    tk_img = tkimg_resized(_img, width, height, keep_ratio)
    lbl_image.image = tk_img
    lbl_image.config(image=tk_img, text=img)
    return lbl_image


def _font(fname="微软雅黑", size=12, bold=tkFont.NORMAL):
    """设置字体"""
    ft = tkFont.Font(family=fname, size=size, weight=bold)
    return ft

def _ft(size=12, bold=False):
    """极简字体设置函数"""
    if bold:
        return _font(size=size, bold=tkFont.BOLD)
    else:
        return _font(size=size, bold=tkFont.NORMAL)


event = threading.Event()
tasks = dict(JGW_SIM=None, JW_SIM=None, JGW_JW_SIM=None)
working_thread_num = 0
results = {"predicts":[],"paths":[],"labels":[]}
thread_shutdown = False

class model_thread(threading.Thread):
    def __init__(self, size, name, model_path, marked_dict_paths, marked_img_dirs, unmarked_img_dirs=None):
        threading.Thread.__init__(self)
        self.name = name
        self.model = Model_Rec(size, name)
        self.model_path = os.path.abspath(model_path)
        self.marked_dict_paths = None
        self.marked_img_dirs = None
        self.unmarked_img_dirs = None
        if marked_dict_paths != None and marked_img_dirs != None:
            
            self.marked_dict_paths = map(lambda p:os.path.abspath(p), marked_dict_paths)
            self.marked_img_dirs = map(lambda p:os.path.abspath(p), marked_img_dirs)
        if unmarked_img_dirs != None:
            self.unmarked_img_dirs = map(lambda p:os.path.abspath(p), unmarked_img_dirs)
    def load_database(self, marked_dict_paths=None, marked_img_dirs=None, unmarked_img_dirs=None):
        if marked_dict_paths != None and marked_img_dirs != None:
            
            marked_dict_paths = map(lambda p:os.path.abspath(p), marked_dict_paths)
            marked_img_dirs = map(lambda p:os.path.abspath(p), marked_img_dirs)
        if unmarked_img_dirs != None:
            unmarked_img_dirs = map(lambda p:os.path.abspath(p), unmarked_img_dirs)
        self.model.load_database(marked_dict_paths, marked_img_dirs, unmarked_img_dirs)
    def run(self):
        global tasks
        global working_thread_num
        global results
        global thread_shutdown
        self.model.load_ckpt_and_var(self.model_path + ".meta", self.model_path)
        self.model.load_database(self.marked_dict_paths, self.marked_img_dirs, self.unmarked_img_dirs)
        while(True):
            event.wait()
            if thread_shutdown == True:
                break
            if tasks[self.name] == None:
                time.sleep(1)
                continue
            else:
                working_thread_num += 1
                if self.name == "JGW_JW_SIM" and tasks[self.name] == 1:
                    self.jgw_jw_dzhs_generate("./user_jgw_jw_hstc.txt")
                else:
                    tmp = self.model.default_recognize(tasks[self.name])
                    results["predicts"].extend(np.array(tmp["predicts"]) * 0.9 if self.name == "JGW_JW_SIM"\
                                                                                else tmp["predicts"])
                    results["paths"].extend(tmp["paths"])
                    results["labels"].extend(tmp["labels"])
                tasks[self.name] = None
                working_thread_num -= 1
                while working_thread_num != 0:
                    time.sleep(1)
                event.clear()
class Window(tk.Frame):

    def __init__(self, master = None):
        tk.Frame.__init__(self, master)
        self.pack(fill=tk.BOTH,expand=tk.YES)
        self.choosefunc_val = tk.StringVar()
        self.choose_chinese = tk.StringVar()
        self.choose_chinese.set("")
        self.display_cols_num = 6 #列数
        self.display_rows_num = 6 #行数
        self.display_board = None
        self.display_board_items = []
        self.display_board_items_sub = []
        self.display_items = []
        self.display_items_point = 0
        self.body()      # 绘制窗体组件
        self.model_threads = [\
        model_thread((128,128,1), "JGW_SIM", "./model/vgg_jgw_cycle.ckpt-2000000", ["./data/jgw_label"], ["./data/jgw"]), \
        model_thread((128,128,1), "JW_SIM", "./model/vgg_jw_cycle.ckpt-2000000", ["./data/jw_label"], ["./data/jw"]), \
        model_thread((128,128,1), "JGW_JW_SIM", "./model/vgg_jgw_jw_cycle.ckpt-2000000", ["./data/jgw_label", "./data/jw_label"], ["./data/jgw", "./data/jw"]), \
        ]
        for t in self.model_threads:
            t.start()
    # 绘制窗体组件
    def body(self):
        self._title = self.title()
        self._title.pack(side=tk.LEFT, fill=tk.Y)

        self._main = tk.Frame(self, bg="white", width=500)
        self._left = self.welcome()
        self._left.pack(side=tk.LEFT, fill=tk.BOTH)
        self._main.pack(expand=tk.YES, fill=tk.BOTH)
    def _label(self, frame, text, fg="gray", size = 16, bold=False):
        return tk.Label(frame, bg="white", fg=fg, text=text, font=_ft(size, bold))

    def _button(self, frame, text, size, bold=False, width = 14, height = 2, command=None):
        return tk.Button(frame, text=text, command=command, bg="white", fg="brown", width=width, height=height, font=_ft(size, bold))

    def _display(self, EXIST = None):
        def clear_destroy_plot():
            for temp_f in self.display_board_items_sub:
                temp_f.destroy()
            self.display_board_items_sub.clear()
        def renew_display_board_plot():#会清空，并根据self.display_board_items和self.display_items_point重新生成
            clear_destroy_plot()
            if self.display_board != None:
                for i in range(self.display_rows_num):
                    if self.display_items_point >= len(self.display_items):
                        break
                    for j in range(self.display_cols_num):
                        if self.display_items_point < len(self.display_items):
                            now_temp_f = self.display_board_items[i * self.display_cols_num + j]
                            temp_f = tk.Frame(now_temp_f, width=100, height=100, bg='whitesmoke')
                            
                            item = list(self.display_items[self.display_items_point])
                            if item[0] != "this" and item[0] != None:
                                temp_f1 = tk.Frame(temp_f, width=100, height=100, bg='whitesmoke')
                                image_label(temp_f1, item[0], 100, 100, keep_ratio=True).pack(side=tk.TOP, fill=tk.X)
                                tk.Label(temp_f1, bg="red", fg="black", text=item[0].split("/")[-1], font=_ft(6, True)).pack(side=tk.TOP, fill=tk.X)
                                temp_f1.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
                            if item[1] != None:
                                temp_f2 = tk.Frame(temp_f, width=100, height=100, bg='whitesmoke')
                                image_label(temp_f2, item[1], 100, 100, keep_ratio=True).pack(side=tk.TOP, fill=tk.X)
                                tk.Label(temp_f2, bg="blue", fg="black", text=item[1].split("/")[-1], font=_ft(6, True)).pack(side=tk.TOP, fill=tk.X)
                                temp_f2.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
                                temp_f3 = tk.Frame(temp_f2, width=100, height=100, bg='whitesmoke')
                                tk.Label(temp_f3, bg="gray", fg="black", text="%.4f" % float(item[2]), font=_ft(6, True)).pack(side=tk.LEFT, fill=tk.Y)
                                tmp_chinese_label = tk.Label(temp_f3, bg="gray", fg="black", text=str(item[3]), font=_ft(14, True))
                                tmp_chinese_label.pack(side=tk.LEFT, fill=tk.Y)
                                
                                temp_f3.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
                            else:
                                temp_f4 = tk.Frame(temp_f, width=100, height=100, bg='whitesmoke')
                                tk.Label(temp_f4, bg="gray", fg="black", text="None", font=_ft(15, True)).pack(side=tk.TOP, fill=tk.X)
                                temp_f4.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
                            temp_f.pack(expand=tk.YES, fill=tk.BOTH)
                            self.display_items_point += 1
                            
                            self.display_board_items_sub.append(temp_f)
                        else:
                            break
                
        def get_page_items_num():
            return self.display_cols_num * self.display_rows_num
        def get_max_page():
            if int(len(self.display_items)/get_page_items_num()) * get_page_items_num() < len(self.display_items):
                return int(len(self.display_items)/get_page_items_num()) + 1
            else:
                return int(len(self.display_items)/get_page_items_num())
        def get_now_page():
            return int((self.display_items_point-1)/get_page_items_num()) + 1
        def get_page_point(page_num):
            return get_page_items_num() * (page_num - 1)
        def renew_now_page_input(page_num):
            self.display_now_page_input.delete(0.0, tk.END)
            self.display_now_page_input.insert(0.0, str(page_num))
            self.display_items_point = get_page_point(1 if page_num==0 else page_num)
        def goto_fun():
            goto_page_str = self.display_now_page_input.get(0.0,tk.END)
            goto_page_num = int("".join(list(filter(str.isdigit, goto_page_str))))
            if goto_page_num > get_max_page() or goto_page_num < 1:
                print("error: input_page is out of index")
            else:
                renew_now_page_input(goto_page_num)
                renew_display_board_plot()
        def pageup_fun():
            last_page_num = get_now_page() - 1
            if last_page_num > get_max_page() or last_page_num < 1:
                print("log: page is out of index")
            else:
                renew_now_page_input(last_page_num)
                renew_display_board_plot()
        def pagedown_fun():
            last_page_num = get_now_page() + 1
            if last_page_num > get_max_page() or last_page_num < 1:
                print("log: page is out of index")
            else:
                renew_now_page_input(last_page_num)
                renew_display_board_plot()
        if EXIST == True:
            if self.display_board == None:
                self.display_board = tk.Frame(self._main, bg="white", width=500)

                for i in range(self.display_rows_num):
                    temp_f = tk.Frame(self.display_board, bg="black", width=50)
                    for j in range(self.display_cols_num):
                        temp_ff = tk.Frame(temp_f, width=40, height=40,  bg='white')
                        
                        temp_ff.pack(expand=tk.YES, side=tk.LEFT, fill=tk.BOTH)
                        self.display_board_items.append(temp_ff)
                    temp_f.pack(expand=tk.YES, side=tk.TOP, fill=tk.BOTH)
                temp_f = tk.Frame(self.display_board, bg="white", width=500)
                self.display_now_page_input = tk.Text(temp_f, height=1, width=5)
                self.display_max_page_label = tk.Label(temp_f, bg="gray", fg="black", text="/0", font=_ft(18, True))
                self.display_goto_btn = self._button(temp_f, "跳转", 10, True, width = 10, height = 1,command = goto_fun)
                self.display_pageup_btn = self._button(temp_f, "上一页", 10, True, width = 10, height = 1,command = pageup_fun)
                self.display_pagedown_btn = self._button(temp_f, "下一页", 10, True, width = 10, height = 1,command = pagedown_fun)
                self.display_now_page_input.pack(side=tk.LEFT, fill=tk.Y)
                self.display_max_page_label.pack(side=tk.LEFT, fill=tk.Y)
                self.display_goto_btn.pack(side=tk.LEFT, fill=tk.Y)
                self.display_pageup_btn.pack(side=tk.LEFT, fill=tk.Y)
                self.display_pagedown_btn.pack(side=tk.LEFT, fill=tk.Y)
                temp_f.pack(side=tk.TOP, fill=tk.X)
                self.display_board.pack(expand=tk.YES, side=tk.LEFT, fill=tk.BOTH)
            renew_now_page_input(0 if get_max_page()==0 else 1)
            renew_display_board_plot()         
        elif EXIST == False:
            if self.display_board != None:
                self.display_board.destroy()
                self.display_board_items.clear()
                self.display_board = None
        else:      
            self.display_max_page_label.config(text="/%d"%get_max_page())
            renew_now_page_input(0 if get_max_page()==0 else 1)
            renew_display_board_plot()
        
    def welcome(self):
        def _label(frame, text, fg="gray", size = 16, bold=False):
            return tk.Label(frame, bg="white", fg=fg, text=text, font=_ft(size, bold))
        str_ = "甲骨文考释辅助系统有三个功能块组成：\n\n分别为甲骨文-金文对照互释推测、甲骨文-金文混合手写输入检索、已有考释甲骨文识别。\n\n甲骨文-金文对照互释推测功能：\n根据已有甲骨文、金文数据库、以具备独有标注的甲骨文（金文）为线索，给出推测算法认为与之最相似的未标注的金文（甲骨文）\n\n甲骨文-金文混合手写输入检索功能：\n手写输入一个甲骨文或者金文，检索相似的甲骨文和金文；\n\n已有考释甲骨文识别功能：\n分词并识别已考释甲骨文，给出最可能的汉字，并给出其他候选字\n"

        frame = tk.Frame(self._main, bg="white")

        temp_f1 = tk.Frame(frame, bg="whitesmoke", width=400)
        _label(temp_f1, str_).pack(anchor=tk.W, padx=100, pady=100)
        

        temp_f1.pack(expand=tk.YES, fill=tk.BOTH)
        return frame
    def hstc(self):
        def _hstc_read_data(filename = "./default_jgw_jw_hstc.txt"):
            unknow_imgs = []
            know_imgs = []
            predicts = []
            labels = []
            with open(filename) as f:
                for line in f:
                    if line == "jgw:\n" or line == "jw:\n":
                        continue
                    stems = line.strip().split()
                    if stems[3] in labels:
                        continue
                    unknow_imgs.append(stems[0])
                    know_imgs.append(stems[1])
                    predicts.append(stems[2])
                    labels.append(stems[3])
            indices = list(range(len(labels)))
            indices.sort(key=lambda i:predicts[i], reverse=True)
            unknow_imgs = np.array(unknow_imgs)
            know_imgs = np.array(know_imgs)
            predicts = np.array(predicts)
            labels = np.array(labels)
            unknow_imgs = unknow_imgs[indices]
            know_imgs = know_imgs[indices]
            predicts = predicts[indices]
            labels = labels[indices]
            self.display_items.extend(list(zip(unknow_imgs, know_imgs, predicts, labels)))
            self._display()
        def _hstc_generate():
            print(self.hstc_datapath_input.get(0.0,tk.END))
            self.display_cols_num = 6 #列数
            self.display_rows_num = 6 #行数
        str_1 = "甲骨文-金文对照互释推测："
        str_2 = "点击“读取”，读取由默认的未考释甲骨文、\n金文生成的对照互释表；\n\n若使用其他未考释甲骨文、金文数据，\n在下列文本框中输入数据文件路径，\n以英文逗号“,”分割，\n\n并点击“生成”按钮，可能花费较长时间。\n"

        frame = tk.Frame(self._main, bg="white", width=500)

        temp_f1 = tk.Frame(frame, bg="whitesmoke", width=400)
        self._label(temp_f1, str_1, fg="brown", size=20, bold=True).pack(padx=0, pady=10)
        self._label(temp_f1, str_2).pack(padx=0, pady=10)
        self.hstc_datapath_input = tk.Text(temp_f1, height=3, width=50)
        temp_f2 = tk.Frame(temp_f1, bg="whitesmoke", width=400)
        hstc_load_btn = self._button(temp_f2, "读取", 14, True, command = _hstc_read_data)
        hstc_generate_btn = self._button(temp_f2, "生成", 14, True, command = _hstc_generate)
        
        self.hstc_datapath_input.pack(side=tk.TOP, padx=5, pady=20)
        
        hstc_load_btn.pack(side=tk.LEFT, padx=5, pady=20)
        hstc_generate_btn.pack(side=tk.LEFT, padx=5, pady=20)

        temp_f2.pack()
        temp_f1.pack(side=tk.LEFT, fill=tk.Y)
        return frame
    
    def srjs(self):

        def _drawboard_paint(event):
            x1, y1 = (event.x - 10), (event.y - 10)
            x2, y2 = (event.x + 10), (event.y + 10)
            self.drawboard.create_oval(x1, y1, x2, y2, fill='black')
        def _srjs_search():
            global tasks
            global working_thread_num
            global results
            self.display_cols_num = 8 #列数
            self.display_rows_num = 8 #行数
            self.srjs_text.config(text="检索中") 
            self.update()
            start_time = time.time()
            ps = self.drawboard.postscript(colormode="color")
            img = Image.open(BytesIO(ps.encode("utf-8")))
            img.save("tmp.gif")
            paths = []
            scores = []
            labels = []
            tasks["JGW_SIM"] = "tmp.gif"
            tasks["JW_SIM"] = "tmp.gif"
            tasks["JGW_JW_SIM"] = "tmp.gif"
            event.set()
            while tasks["JGW_SIM"]!=None or tasks["JW_SIM"]!=None or tasks["JGW_JW_SIM"]!=None:
                time.sleep(1)
            indices = list(range(len(results["paths"])))
            indices = list(filter(lambda ind:results["predicts"][ind][0]>results["predicts"][ind][1], indices))
            scores.extend([score[0] for score in np.array(results["predicts"])[indices]])
            paths.extend(np.array(results["paths"])[indices])
            labels.extend(np.array(results["labels"])[indices])
            results["predicts"].clear()
            results["paths"].clear()
            results["labels"].clear()
            indices = list(range(len(paths)))

            indices.sort(key=lambda i:scores[i], reverse=True)
            paths = np.array(paths)
            scores = np.array(scores)
            labels = np.array(labels)
            paths = paths[indices]
            scores = scores[indices]
            labels = labels[indices]
            self.display_items.extend(list(zip(["this" for _ in range(len(indices))], paths, scores, labels)))
            self._display()
            #self.srjs_text.config(text="检索完毕,共耗时%.2f秒"%(time.time() - start_time))
            self.srjs_text.config(text="检索完毕")
            self.srjs_text.update()
        str_1 = "甲骨文-金文混合手写输入检索："
        str_2 = "在下方画框中\n手写输入一个甲骨文或者金文，\n点击“检索”按钮\n开始检索相似的甲骨文和金文！\n"

        frame = tk.Frame(self._main, bg="white", width=500)

        temp_f1 = tk.Frame(frame, bg="whitesmoke", width=400)
        self._label(temp_f1, str_1, fg="brown", size=20, bold=True).pack(padx=0, pady=10)
        self._label(temp_f1, str_2).pack(padx=0, pady=10)
        self.drawboard = tk.Canvas(temp_f1, width=400, height=400, background='white')
        self.drawboard.pack()
        self.drawboard.bind('<B1-Motion>', _drawboard_paint)
        self.srjs_btn = self._button(temp_f1, "检索", 14, True, command = _srjs_search)
        self.srjs_text = self._label(temp_f1, "", fg="brown", size=20, bold=True)
        self.srjs_btn.pack(side=tk.TOP, padx=5, pady=20)
        self.srjs_text.pack(side=tk.TOP, padx=5, pady=20)
        temp_f1.pack(side=tk.LEFT, fill=tk.Y)
        return frame
    def jgsb(self):
        def _jgsb_open_jgw():
            self.jgsb_jgw_file_path = tk.filedialog.askopenfilename()
            self.jgsb_jgw = tkimg_resized(Image.open(self.jgsb_jgw_file_path), 360, 360, True)

            self.drawboard.create_image(self._main.winfo_x()+self.drawboard.winfo_x(), self._main.winfo_y()+self.drawboard.winfo_y(), image=self.jgsb_jgw) 
            self._main.update()
            self.drawboard.update()
        def _jgsb_set_first_chinese(no):
            now_first_chinese = self.choose_chinese.get() + str(list(self.display_items[no * self.display_cols_num*self.display_rows_num])[3])
            self.choose_chinese.set(now_first_chinese)
            
            self.jgsb_fisrt_chinese.config(text="识别汉字为:%s"%now_first_chinese)
            self._main.update()
            self.jgsb_fisrt_chinese.update()
        def _jgsb_recogenize():
            global tasks
            global working_thread_num
            global results
            self.display_cols_num = 6 #列数
            self.display_rows_num = 6 #行数
            self.jgsb_text.config(text="识别中")
            self.update()
            start_time = time.time()

            number = split_chars(self.jgsb_jgw_file_path)
            no = 0
            self.choose_chinese.set("")
            for root, dirs, files in os.walk("./split_chars_dir/"):
                files.sort()
                for name in files:
                    paths = []
                    scores = []
                    labels = []
                    if int("".join(list(filter(str.isdigit, name)))) < number:
                        tasks["JGW_SIM"] = "./split_chars_dir/" + name
                        event.set()
                        while tasks["JGW_SIM"]!=None or tasks["JW_SIM"]!=None or tasks["JGW_JW_SIM"]!=None:
                            time.sleep(1)
                        indices = list(range(len(results["paths"])))
                        indices = list(filter(lambda ind:results["predicts"][ind][0]>results["predicts"][ind][1], indices))
                        scores.extend([score[0] for score in np.array(results["predicts"])[indices]])
                        paths.extend(np.array(results["paths"])[indices])
                        labels.extend(np.array(results["labels"])[indices])
                        results["predicts"].clear()
                        results["paths"].clear()
                        results["labels"].clear()
                        indices = list(range(len(paths)))

                        indices.sort(key=lambda i:scores[i], reverse=True)
                        paths = np.array(paths)
                        scores = np.array(scores)
                        labels = np.array(labels)
                        paths = paths[indices]
                        scores = scores[indices]
                        labels = labels[indices]
                        if len(indices) >= self.display_rows_num*self.display_cols_num:
                            
                            display_paths = []
                            display_scores = []
                            display_labels = []
                            for path, score, label in zip(paths, scores, labels):
                                if len(display_paths) == self.display_rows_num*self.display_cols_num:
                                    break
                                if label in display_labels:
                                    continue
                                else:
                                    display_paths.append(path)
                                    display_scores.append(score)
                                    display_labels.append(label)
                            self.display_items.extend(list(zip(["./split_chars_dir/" + name for _ in range(len(display_paths))], display_paths, display_scores, display_labels)))
                        else:
                            
                            self.display_items.extend(list(zip(["./split_chars_dir/" + name for _ in range(len(indices))], paths, scores, labels)))
                            tmp = [None for _ in range(self.display_rows_num*self.display_cols_num - len(indices))]
                            self.display_items.extend(list(zip(tmp, tmp, tmp, tmp)))
                        _jgsb_set_first_chinese(no)
                        no += 1
            self._display()
            #self.jgsb_text.config(text="识别完毕,共耗时%.2f秒"%(time.time() - start_time))
            self.jgsb_text.config(text="识别完毕")
            self.jgsb_text.update()
            
        str_1 = "已有考释甲骨文识别："
        str_2 = "通过“打开文件”按钮\n选择一张多个甲骨文图片，\n点击“识别”按钮开始进行识别！\n每甲骨字提供一页候选字"
        frame = tk.Frame(self._main, bg="white", width=500)

        temp_f1=tk.Frame(frame, bg="whitesmoke", width=400)
        self._label(temp_f1, str_1, fg="brown", size=20, bold=True).pack(padx=0, pady=10)
        self._label(temp_f1, str_2).pack(padx=0, pady=10)
        self.drawboard = tk.Canvas(temp_f1, width=400, height=400, background='white')
        self.drawboard.pack()
        
        temp_f2 = tk.Frame(temp_f1, bg="whitesmoke", width=400)
        tmp_btn = self._button(temp_f2, "打开文件", 14, True, command = _jgsb_open_jgw)
        self.jgsb_btn = self._button(temp_f2, "识别", 14, True, command = _jgsb_recogenize)
        self.jgsb_text = self._label(temp_f1, "", fg="brown", size=14, bold=True)

        tmp_btn.pack(side=tk.LEFT, padx=5, pady=20)
        self.jgsb_btn.pack(side=tk.LEFT, padx=5, pady=20)
        temp_f2.pack(side=tk.TOP, fill=tk.Y)
        self.jgsb_text.pack(side=tk.TOP, padx=5, pady=20)
        self.jgsb_fisrt_chinese = self._label(temp_f1, "识别汉字为:", fg="brown", size=14, bold=True)
        self.jgsb_fisrt_chinese.pack(side=tk.TOP, padx=5, pady=20)
        self.choose_chinese_num = tk.Entry(temp_f1)
        temp_f1.pack(side=tk.LEFT, fill=tk.Y)
        return frame
    def _choose_func(self, _func):
        self.choosefunc_val.set(_func)
        print(self.choosefunc_val.get())
        self._left.destroy()
        self.display_items.clear()
        self.display_items_point = 0
        self._left = _func()
        
        if _func != self.welcome:
            self._display(True)
        else:

            self._display(False)
        self._left.pack(side=tk.LEFT, fill=tk.BOTH)
    def title(self):
        def _button(frame, text, size, bold=False, width = 14, height = 2, command=None):
            return tk.Button(frame, text=text, command=command, bg="white", fg="brown", width=width, height=height, font=_ft(size, bold))
        frame = tk.Frame(self)
        self.bg = tkimg_resized(Image.open("./icos/bg1.jpg"), 240, 1000, False)
        canvas = tk.Canvas(frame, width=800, height=1000, bg='green')
        canvas.create_image(100, 500, image=self.bg) 
        canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.welcome_btn = _button(canvas, "甲骨文\n考释辅助系统\n介绍主页", 20, True, 10, 3, \
                            command = lambda __func = self._choose_func:self._choose_func(self.welcome))
        self.hstc_btn = _button(canvas, "甲骨文-金文\n对照互释推测", 14, False, \
                            command = lambda __func = self._choose_func:self._choose_func(self.hstc))
        self.srjs_btn = _button(canvas, "甲骨文-金文\n混合手写输入检索", 14, False, \
                            command = lambda __func = self._choose_func:self._choose_func(self.srjs))
        self.jgsb_btn = _button(canvas, "已有考释甲骨文识别", 14, False, \
                            command = lambda __func = self._choose_func:self._choose_func(self.jgsb))
        
        self.welcome_btn.pack(side=tk.TOP, padx=0, pady=100)
        self.hstc_btn.pack(side=tk.TOP, padx=5, pady=20)
        self.srjs_btn.pack(side=tk.TOP, padx=5, pady=20)
        self.jgsb_btn.pack(side=tk.TOP, padx=5, pady=20)


        return frame



if __name__ == "__main__":
    master = tk.Tk()
    def onclose():
        global thread_shutdown
        thread_shutdown = True
        event.set()

        master.destroy()
    master.geometry("%dx%d" % (1800, 1000))  # 窗体尺寸
    master.minsize(1200, 800)
    master.title("甲骨文考释辅助系统")                 # 窗体标题
    master.tk.call('wm', 'iconphoto', master._w, tk.PhotoImage(file="./icos/ico.png"))  # 窗体图标
    master.update()
    master.grab_set()
    master.protocol("WM_DELETE_WINDOW", onclose)
    app = Window(master)
    master.mainloop()
