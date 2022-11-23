# 有关画图的lib
import sys
sys.path.insert(0, "/home/wzj/GiteeProjects/faas-scaler")
from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats


save_root = "/home/wzj/GiteeProjects/faas-scaler/results"


class Pyplot_config:
    def __init__(self, figsize=(20,6), fontsize=30):
        # print(f"type(fontsize):{type(fontsize)}")
        self.color_list = ["b","g","r","c","m","y","grey","brown","purple"]
        self.hatch_list = [None, '...', 'x', '***', '|', '-', '/',  '+' , 'O', 'o', 'XXX', '.', '*']
        self.linestyle_list = [None, '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']
        self.dash_list = [(2,5),(4,10),(3,3,2,2),(5,2,20,2),(5,2),(1,1,5,4),(5,8),(2,4,5)]
        self.marker_list = ['o','.','*','+','v']
        self.edge_color_list = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        self.label_size = int(fontsize * 0.8) 
        self.tick_size = int(fontsize * 0.8)
        self.title_size = fontsize
        self.legend_size = int(fontsize * 0.8)
        self.bar_width = 0.1
        self.dpi = 300
        self.figsize = figsize
        self.title = None
        




class Plotter(Pyplot_config):
    def __init__(self, figsize=(20,6), fontsize=30):
        super().__init__(figsize=figsize, fontsize=fontsize)
        pass

    # 用于画折线图, 有几条线就画几个
    def plot_lines(self, x=None, y_list=None, y_label_list=None, x_label="x", y_label="y", title=None, save_root="/home/wzj/GiteeProjects/faas-scaler/results", filename="demo.png", is_show=False, legend_loc="best"):
        # 如果 save_root 没有创建，则创建一个
        os.makedirs(save_root, exist_ok=True)
        # 如果没有指定x的值，就用正整数列进行生成
        if x == None:
            x = [i for i in range(len(y_list[0]))]
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)


        if y_label_list != None:
            for index, y in enumerate(y_list):
                plt.plot(x, y, color=self.color_list[index], linestyle=self.linestyle_list[index], label=y_label_list[index])
        else:
            for index, y in enumerate(y_list):
                plt.plot(x, y, color=self.color_list[index], linestyle=self.linestyle_list[index])

        plt.xlabel(x_label, fontsize=self.label_size)
        plt.ylabel(y_label, fontsize=self.label_size)
        # 判断是否绘制title
        if title != None:  
            plt.title(title, fontdict={'size': self.title_size})
        # 判断是否绘制legend
        if y_label_list != None:
            plt.legend(fontsize=self.legend_size, loc=legend_loc) # 将样例显示出来

        # 调整 tick 的字体大小
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)
        
        plt.tight_layout()
        if is_show:
            plt.show()
        savepath = os.path.join(save_root, filename)
        print(f"图片保存到:{savepath}")
        plt.savefig(savepath)

    # 绘制条形图
    def plot_bars(self, x_label="x", y_label="y", legend_title=None, legend_ncol=1, bbox_to_anchor=None, legend_loc="best", x_data=None, bar_data_list=None, bar_label_list=None, y_min=None, y_max=None, save_root="/home/wzj/GiteeProjects/faas-scaler/results", filename="demo.png", is_hatch=False, is_show=False, ):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = plt.subplot(111)
        # 设置轴的标签字体和大小
        ax.set_xlabel(x_label, fontdict={'size': self.label_size})
        ax.set_ylabel(y_label, fontdict={'size': self.label_size})

        # 分别画柱子
        r_base = np.arange(len(x_data))
        # print(len(x_data))
        # 是否用阴影hatch区别
        if is_hatch:
            hatch_list = self.hatch_list
        else:
            hatch_list = [None for i in range(10)]

        for index, (bar_label, bar_data) in enumerate(zip(bar_label_list, bar_data_list)):
            r = [x + self.bar_width * (index)  for x in r_base]
            ax.bar(r, bar_data, color=self.color_list[index], width=self.bar_width, edgecolor=self.edge_color_list[index], label=bar_label_list[index], hatch=hatch_list[index]) # 创建柱子
            
        # 添加x轴名称
        plt.xticks([r + (len(bar_data_list)-1)/2*self.bar_width for r in range(len(x_data))], x_data, size=self.label_size)
        plt.yticks(size=self.label_size)

        # 创建图例
        legend = plt.legend(fontsize=self.legend_size, title=legend_title, loc=legend_loc, ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)
        legend.get_title().set_fontsize(fontsize=self.legend_size)
        legend._legend_box.align = "left"
        if y_min != None and y_max != None:
            plt.ylim(y_min, y_max)
        plt.tight_layout()

        if is_show:
            plt.show()
        savepath = os.path.join(save_root, filename)
        print(f"图片保存到:{savepath}")
        plt.savefig(savepath)
        # 展示图片
        if is_show:
            plt.show()
        pass

    @staticmethod
    def formatnum(x, pos):
        if x == 0:
            return f'{int(x)}'
        return "%.1f" % x


    @staticmethod
    def cal_cdf(input_data):
        val, cnt = np.unique(input_data, return_counts=True)
        pmf = cnt / len(input_data)
        fs_rv_dist = stats.rv_discrete(values=(val, pmf))
        return {"x":val,"y":fs_rv_dist.cdf(val)}


    # 绘制 cdf 图
    def plot_cdfs(self, x_label="x", y_label="cdf", legend_title=None, legend_ncol=1, bbox_to_anchor=None, legend_loc="best", cdf_data_list=None, cdf_label_list=None, is_marker=False, linewidth=2, alpha=1, save_root="/home/wzj/GiteeProjects/faas-scaler/results", filename="demo.png", is_show=False):
        # 画布
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        # 设置轴的标签字体和大小
        ax.set_xlabel(x_label, fontdict={'size': self.label_size})
        ax.set_ylabel(y_label, fontdict={'size': self.label_size})

        # 调整 tick 的字体大小
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)

        # 让角标变0
        formatter = FuncFormatter(self.formatnum)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        # 计算cdf的值
        for index, (cdf_data, cdf_label) in enumerate(zip(cdf_data_list, cdf_label_list)):
            res = self.cal_cdf(cdf_data)
            x = res["x"]
            y = res["y"]
            # 画图
            if is_marker:
                marker = self.marker_list[index]
            else:
                marker = None
            ax.plot(
                x, y,
                color=self.color_list[index],
                linestyle=self.linestyle_list[index],
                dashes=self.dash_list[index],
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                label=cdf_label,
            )
        # 创建图例
        legend = plt.legend(fontsize=self.legend_size, title=legend_title, loc=legend_loc, ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)
        legend.get_title().set_fontsize(fontsize=self.legend_size)
        legend._legend_box.align = "left"

        plt.tight_layout()

        if is_show:
            plt.show()
        savepath = os.path.join(save_root, filename)
        print(f"图片保存到:{savepath}")
        plt.savefig(savepath)
        # 展示图片
        if is_show:
            plt.show()
        pass
        
    

if __name__ == "__main__":
    my_plotter = Plotter(figsize=(8,6))
    
    y1 = [1,2,3,2,1,2,3,2,1,2]
    y2 = [-1,0,1,0,-1,-1,0,1,0,-1]

    my_plotter.plot_lines(y_list=[y1,y2], y_label_list=["y1","y2"], title="This is a demo!")
