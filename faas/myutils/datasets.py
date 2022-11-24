"""
用于加载dataset,对数据进行提取,过滤,查找等操作。
"""
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import time
from .trace import generate_invocation_in_second
import math
from .trace import is_simple_period, is_idle_rate_filter, is_avg_rate_filter, cal_diff
from tqdm import tqdm
from random import sample
from .plot import Plotter

from typing import TypeVar
T = TypeVar('T')



class Basic_dataset:
    def __init__(self, dataset_root="/home/wzj/GiteeProjects/faas-scaler/datasets/azurefunctions-dataset2019"):
        self.dataset_root = dataset_root
        self.save_name = "azure_dataset.pickle"

    # 将csv文件加载到内存中，用dict来查找
    def cache_dataset(self):
        pass

    # 将 dataset对象保存
    def save_pickle(self):
        savename = self.save_name
        savepath = os.path.join(self.dataset_root, savename)
        start_pickle_t = time.time()
        with open(savepath,'wb') as f:
            pickle.dump(self, f)
        end_pickle_t =  time.time()
        pickle_t = end_pickle_t - start_pickle_t
        print(f"成功将dataset对象保存到:{savepath}!")
        print(f"所用时间:{pickle_t}") 
        pass

    # 将 dataset读取
    def load_pickle(self: T) -> T:
        savename = self.save_name
        savepath = os.path.join(self.dataset_root, savename)
        start_pickle_t = time.time()
        with open(savepath,'rb') as f:
            obj = pickle.load(f)
        end_pickle_t =  time.time()
        pickle_t = end_pickle_t - start_pickle_t
        print(f"成功加载{savename}!")
        print(f"pickle_t:{pickle_t}")
        return obj 

    # 拿到调用时间序列
    def get_invocation_trace(self):
        pass



class Azure_dataset(Basic_dataset):
    def __init__(self, dataset_root="/home/wzj/GiteeProjects/faas-scaler/datasets/azurefunctions-dataset2019"):
        super().__init__()
        self.dataset_root = dataset_root
        self.savename = "azure_dataset.pickle"
        # 用于缓存每一天的dataframe
        self.invocation_df_dict = {}
        self.duration_df_dict = {}
        self.memory_df_dict = {}
        pass


    def load_dataset(self, day):
        # csv的文件命名规则
        invocation_filename = f"invocations_per_function_md.anon.d{str(day).rjust(2,'0')}.csv"
        duration_filename = f"function_durations_percentiles.anon.d{str(day).rjust(2,'0')}.csv"
        memory_filename = f"app_memory_percentiles.anon.d{str(day).rjust(2,'0')}.csv"
        # 加载 invocation
        invocation_filepath = os.path.join(self.dataset_root, invocation_filename)
        invocation_df = pd.read_csv(invocation_filepath)
        # 加载 duration
        duration_filepath = os.path.join(self.dataset_root, duration_filename)
        duration_df = pd.read_csv(duration_filepath)
        # 加载 memory
        if day <= 12:
            memory_filepath = os.path.join(self.dataset_root, memory_filename)
            memory_df = pd.read_csv(memory_filepath)
        else:
            memory_df = None
        return invocation_df, duration_df, memory_df


    # 将csv文件加载到内存中，用dict来查找
    def cache_dataset(self):
        print(f"正在加载数据集!")
        for day in tqdm(range(1,14+1)):
            invocation_df, duration_df, memory_df = self.load_dataset(day)
            self.invocation_df_dict[str(day)] = invocation_df
            self.duration_df_dict[str(day)] = duration_df
            self.memory_df_dict[str(day)] = memory_df


    # 找到某一天全部的hash_function, 返回值是 hash_function_list
    def get_hash_function_list(self, day):
        invocation_df = self.invocation_df_dict[str(day)]
        hash_function_list = invocation_df["HashFunction"].tolist()
        print(type(hash_function_list))
        return hash_function_list
    

    # 找到在每一天都出现的 hash_function_set
    def find_all_hash_function_set(self):
        find_all_hash_function_set = set()
        for day in range(1,14+1):
            if day == 1:
                find_all_hash_function_set = set(self.get_hash_function_list(day))
                continue
            per_day_hash_function_set = set(self.get_hash_function_list(day))
            find_all_hash_function_set = per_day_hash_function_set & find_all_hash_function_set

        return find_all_hash_function_set


    # 拼接某一个hash_function的14天全部trace, 以分钟为单位
    def get_all_invocation_trace(self, hash_function):
        all_invocation_list = []
        for day in range(1,14+1):
            invocation_df_day = self.invocation_df_dict[str(day)]
            df_slice = invocation_df_day.loc[invocation_df_day["HashFunction"] == hash_function]
            invocation_list = []
            # 遍历每一分钟
            for i in range(1,1440+1):
                is_invocation = int(df_slice[str(i)])
                invocation_list.append(is_invocation)
            all_invocation_list.extend(invocation_list)
        return all_invocation_list   


    # 将某一个hash_function按照调用次数，以及泊松分布，推断出并发数的trace
    def get_concurrency_trace(self, hash_function, grain="min", is_draw=False):
        # 先找到 hash_function 的 avg_duration
        for day in range(1,14+1):
            #print(f"day:{day}")
            duration_df = self.duration_df_dict[str(day)]
            try:
                duration = duration_df.loc[duration_df["HashFunction"] == hash_function]["Average"].tolist()[0] / 1000 # 转换秒
                break
            except:
                # print(f"day:{day},duration_df中没找到{hash_function}")
                duration = 1
                continue
            
            print(f"hash_function:{hash_function}的duration:{duration} 秒")

        # 获取分钟级序列
        invocation_in_min = self.get_all_invocation_trace(hash_function)
        # 转化成秒级序列
        invocation_in_second = generate_invocation_in_second(invocations_in_min=invocation_in_min)
        # print(f"invocation_in_min:{invocation_in_min}")
        # print(f"invocation_in_second:{invocation_in_second}")

        # 生成pod的并发度list
        concurrency_list = [0 for i in range(len(invocation_in_second))]
        for index, invocation in enumerate(invocation_in_second):
            if duration < 1:
                span = 1
            else:
                span = math.ceil(duration)
            if invocation >= 1:
                for i in range(index, min(len(invocation_in_second),index+span)):
                    concurrency_list[i] += invocation
        
        if grain == "min":
            # 还原分钟级别的调用序列
            concurrency_max_in_min_list = []
            for i in range(int(len(concurrency_list) / 60)):
                concurrency_max_in_min = max(concurrency_list[i*60:(i+1)*60])
                concurrency_max_in_min_list.append(concurrency_max_in_min)
            res = concurrency_max_in_min_list
            origin = invocation_in_min    
        elif grain == "sec":
            res = concurrency_list
            origin = invocation_in_second
        else:
            pass
        # print(f"hash_function:{hash_function} 的并发度list为:{res}")
        # 画对比图
        if is_draw:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,4), dpi=300, sharey=False)
            plt.title(f"duration:{duration}, hash_function:{hash_function}")
            ax1.plot(res, "r--")
            ax2.plot(origin, "b--")
            plt.tight_layout()
            plt.savefig(f"compare_of_concurrency.png")
        return res
    

    # 画出指定function的调用次数图
    def plot_trace(self,hash_function, data_type="Concurrency", result_root="/home/wzj/GiteeProjects/faas-scaler/results/tmp", filename="tmp.png", is_show=False, title=None, start_day=0, end_day=1, color="b", fontsize=40, figsize=(20,6)):
        os.makedirs(result_root, exist_ok=True)
        # 判断是否画concurrency还是原始数据
        assert data_type in ["Concurrency", "Invocation", "Difference", "Bursty"]
        if data_type == "Invocation":
            plot_data_list = self.get_all_invocation_trace(hash_function)
        elif data_type == "Concurrency":
            plot_data_list = self.get_concurrency_trace(hash_function)
        elif data_type == "Difference":
            concurrency_data_list = self.get_concurrency_trace(hash_function)
            plot_data_list = cal_diff(trace=concurrency_data_list)
        if data_type == "Bursty":
            concurrency_data_list = self.get_concurrency_trace(hash_function)
            diff_data_list = cal_diff(trace=concurrency_data_list)
            abs_diff_data_list = [abs(item) for item in diff_data_list]
            # 找到 P95 分位数
            p95 = np.percentile(abs_diff_data_list, 95)

            plot_data_list = []
            for item in diff_data_list:
                if abs(item) > p95:
                    plot_data_list.append(item)
                else:
                    plot_data_list.append(0)
        
        # 根据 start_day 和 end_day 拿到需要绘制的数据
        invocation_data = plot_data_list[1440*start_day:1440*end_day]
        
        my_plotter = Plotter(figsize=figsize, fontsize=fontsize)
        
        my_plotter.color_list[0] = color
        my_plotter.plot_lines(x=None, y_list=[invocation_data],x_label="Timestamp in Minute", y_label=data_type, save_root=result_root, filename=filename)
        
        filepath = os.path.join(result_root, filename)

        if is_show:
            plt.show()
        
        plt.close()


    # 返回符合条件的hash_function list
    def hash_function_filter(self, random_sample_num=500, iat_threshold=5, idle_rate_threshold=(0, 0.5), avg_rate_threshold=(3, 1000)):
        # is_simple_period 表示是否为简单周期性调用
        # avg_invocation_threshold 表示平均调用的次数需要大于的值
        
        hash_function_list = list(self.find_all_hash_function_set()) # 找到在每一天都出现的 hash_function_list
        print(f"len(hash_function_list):{len(hash_function_list)}")

        # 从中随机抽取
        sampled_hash_function_list = sample(hash_function_list, random_sample_num)

        # filtered_hash_function_list
        filtered_hash_function_list = []

        for hash_function in tqdm(sampled_hash_function_list):
            trace = self.get_all_invocation_trace(hash_function=hash_function)
            is_drop_flag = 0 # 表示是否保留此 hash_function 0:保留
            if iat_threshold != None:
                _is_simple_period = is_simple_period(trace=trace, threshold=iat_threshold)
                if _is_simple_period:
                    is_drop_flag += 1

            if idle_rate_threshold != None:
                _is_idle_rate = is_idle_rate_filter(trace=trace, threshold=idle_rate_threshold)
                if not _is_idle_rate:
                    is_drop_flag += 1

            if avg_rate_threshold != None:
                _is_avg_rate = is_avg_rate_filter(trace=trace, threshold=avg_rate_threshold)
                if not _is_avg_rate:
                    is_drop_flag += 1

            if is_drop_flag == 0:
                filtered_hash_function_list.append(hash_function)
                print(f"hash_function:{hash_function}入选!")

        print(f"符合条件的hash_function有:{len(filtered_hash_function_list)}个")
        print(filtered_hash_function_list)
        return filtered_hash_function_list


    # plot all trace:
    def plot_all_trace(self, random_sample_num=1000, result_root="/home/wzj/GiteeProjects/faas-scaler/results/all_traces"):
        all_hash_function_set = self.find_all_hash_function_set()
        sampled_hash_function_list = sample(all_hash_function_set, min(random_sample_num, len(all_hash_function_set)))
        for hash_function in sampled_hash_function_list:
            print(f"正在绘制{hash_function}!")
            self.plot_trace(hash_function=hash_function, result_root=result_root)
        pass



class Crane_dataset(Basic_dataset):
    def __init__(self, dataset_root="/home/wzj/GiteeProjects/faas-scaler/datasets/crane"):
        self.dataset_root = dataset_root
        self.save_name = "crane_dataset.pickle"
        # 用于缓存每一个csv的dataframe
        self.invocation_df_dict = {}
        pass

    def cache_dataset(self):
        for csv_filename in os.listdir(self.dataset_root):
            print(f"csv_filename:{csv_filename}")
            csv_filepath = os.path.join(self.dataset_root, csv_filename)
            df = pd.read_csv(csv_filepath)
            # csv_filename 作为 hash_function
            self.invocation_df_dict[str(csv_filename)] = df
        print(f"完成加载Dataframe到内存!")

    def get_invocation_trace(self, hash_function="input0.csv"):
        df = self.invocation_df_dict[hash_function]
        data = df["value"].tolist()
        return data

    def get_concurrency_trace(self, hash_function):
        return self.get_invocation_trace(hash_function)


            

def test_crane_dataset():
    ds = Crane_dataset()
    ds.cache_dataset()
    ds.save_pickle()
    ds.load_pickle()
    ds.get_invocation_trace()

def test_azure_dataset():
    ds = Azure_dataset()
    ds.cache_dataset()
    ds.save_pickle()
    ds.load_pickle()
    ds.get_invocation_trace()




if __name__ == "__main__":
    test_azure_dataset()
    # ds = Azure_dataset()
    # ds.cache_dataset()
    # ds.save_pickle()
    # ds = ds.load_pickle()

    # ds.get_concurrency_trace(hash_function="2cd7966fdcf401be25c87c37b1710125305ba8a0af7307703845f10110b1dff4", grain="min", is_draw=False)

    # ds.plot_trace(hash_function="2cd7966fdcf401be25c87c37b1710125305ba8a0af7307703845f10110b1dff4")
    # ds.plot_trace(hash_function="dd1c2b7f7f3330adf96eebf332257cdcc0d4b78c0e797bb3239e9504811b6ad3")
    

    # ds.hash_function_filter(iat_threshold=5, random_sample_num=300, idle_rate_threshold=[0.5, 1], avg_rate_threshold=[0, 10])
    # hash_function_list = ds.hash_function_filter(iat_threshold=5, random_sample_num=300, idle_rate_threshold=(0, 0.5),avg_rate_threshold=(5, 1000))

    # Random
    # hash_function_list = ds.hash_function_filter(iat_threshold=5, random_sample_num=100, idle_rate_threshold=(0, 1), avg_rate_threshold=(0, 1000))
    # print(f"hash_function_list:{hash_function_list}")


    # Rare
    # hash_function_list = ds.hash_function_filter(iat_threshold=5, random_sample_num=200, idle_rate_threshold=(0.5, 1), avg_rate_threshold=(0, 100))
    # print(f"hash_function_list:{hash_function_list}")

    # Freq
    # hash_function_list = ds.hash_function_filter(iat_threshold=5, random_sample_num=1500, idle_rate_threshold=(0, 0.1), avg_rate_threshold=(0, 1000))
    # print(f"hash_function_list:{hash_function_list}")
