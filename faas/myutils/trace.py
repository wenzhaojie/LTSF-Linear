import sys
sys.path.insert(0, "/home/wzj/GiteeProjects/faas-scaler")
from numpy import random
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from src.utils.calstats import cal_avg_rate, cal_idle_rate
import pandas as pd


# 将分钟级调用数据 --> 秒级调用数据 （用泊松分布来生成）
def generate_invocation_in_second(invocations_in_min):
    invocations_in_sec = []
    for each_min in invocations_in_min:
        while True:
            lam = each_min / 60
            res = random.poisson(lam=lam, size=60)
            if sum(res) == each_min:
                break
        invocations_in_sec.extend(res)
    return invocations_in_sec


# 将一组数据的每一个点按照负载水平进行分类
def split_trace_in_levels(trace, do_abs=False, do_plot=False, n_levels=5):
    # 先统计分位数
    q = 80
    
    # 先去除0元素
    drop_zero = []
    for item in trace:
        if item != 0:
            drop_zero.append(item)
    if do_abs:
        abs_drop_zero = [abs(item) for item in drop_zero]
    else:
        abs_drop_zero = drop_zero

    percentile_value_list = []
    for level in range(0, n_levels+1):
        percentile = level * 100 / n_levels
        # print(f"percentile:{percentile}")
        percentile_value = np.percentile(abs_drop_zero, percentile)
        # print(f"percentile_value:{percentile_value}")
        percentile_value_list.append(percentile_value)

    # print(f"percentile_value_list:{percentile_value_list}")

    # 分离高低序列
    split_trace_list = [np.zeros(len(trace)) for i in range(n_levels)] # 这是按照从高到低排列的分位数负载成分
    for i in range(n_levels):
        low_threshold = percentile_value_list[i]
        high_threshold = percentile_value_list[i+1]
        for j in range(len(trace)):
            if do_abs:
                compare = abs(trace[j])
            else:
                compare = trace[j] 
            if compare >= low_threshold and compare < high_threshold:
                split_trace_list[i][j] = trace[j]
    if do_plot:
        # 画示意图
        xlim = (0,480)
        ylim = (1.1*min(trace),1.1*max(trace))
        
        # 颜色列表
        color_list = ["r", "g", "b", "c", "k", "m", "y"]
        for index, split_trace in enumerate(split_trace_list):
            subplot = int(f"{n_levels + 1}1{index+1}")
            # print(f"subplot:{subplot}")
            plt.subplot(subplot)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.plot(split_trace, color_list[index+1])
            plt.title(f"{index+1}th")
        
        subplot = int(f"{n_levels + 1}1{n_levels + 1}")
        plt.subplot(subplot)
        plt.plot(trace, color_list[n_levels + 1])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f"original_trace")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"split_high_low.png")
    else:
        pass
    # print(f"split_trace_list:{split_trace_list}")
    return split_trace_list
    
    
def get_continuous_date(num, interval="day", pattern="yyyy-MM-dd hh:mm:ss"):
    date_base = datetime.fromisoformat("2000-01-01 00:00:00")
    date_base_timestamp = date_base.timestamp()

    date_list = []
    if interval == "day":
        stamp_gap = 60*60*24
    elif interval == "hour":
        stamp_gap = 60*60
    elif interval == "min":
        stamp_gap = 60
    else:
        print(f"interval 必须是 day,hour,min 中的一种")
        stamp_gap = 60
    for index in range(num):
        date = datetime.fromtimestamp(date_base_timestamp + index*stamp_gap)
        date_list.append(date)
    return date_list


# 判断某一个调用序列是否是简单周期性, 简单周期性即 inter arrival time 的上下误差不超过 5min
def is_simple_period(trace, threshold=5):
    last_invocation = -1
    iat_list = []
    for index, invocation_num in enumerate(trace):
        if invocation_num >= 1:
            if last_invocation == -1: 
                last_invocation = index
                continue # 首次记录第二次调用到第一次调用的IAT
            # 有调用
            iat_list.append(index - last_invocation)
            last_invocation = index
        else:
            # 无调用
            continue
    # 判断是否是简单周期性
    delta = max(iat_list) - min(iat_list)
    print(f"序列极值相差为:{delta}")
    if delta < threshold:
        print(f"是简单周期!")
        return True
    else:
        print(f"不是简单周期!")
        return False


# 判断某一个调用序列是否落在平均调用率的阈值区间
def is_avg_rate_filter(trace, threshold=(5,100)):
    low_threshold = threshold[0]
    high_threshold = threshold[1]

    avg_rate = cal_avg_rate(trace=trace)
    # print(f"avg_rate:{avg_rate}")
    if avg_rate >= low_threshold and avg_rate <= high_threshold:
        return True
    else:
        return False

    
# 判断某一个调用序列是否落在了空占比的阈值
def is_idle_rate_filter(trace, threshold=(0,0.5)):
    low_threshold = threshold[0]
    high_threshold = threshold[1]

    idle_rate = cal_idle_rate(trace=trace)
    # print(f"idle_rate:{idle_rate}")
    if idle_rate >= low_threshold and idle_rate <= high_threshold:
        return True
    else:
        return False


# 计算差分序列, 其中第一个差分值为0
def cal_diff(trace):
    df = pd.DataFrame({"data":trace})
    res = df.diff()["data"].values.tolist()
    res[0] = 0
    return res


if __name__ == "__main__":
    # 测试
    # invocations_in_sec = generate_invocation_in_second([1,1,2,0,1])
    # print(invocations_in_sec)
    # print(sum(invocations_in_sec))

    # 测试
    # trace = [1,1,2,0,1,8,4,2,6,3,9,7,5,34,4,7,5,3,62,8,6,14,3,6,78,3,6,7,]
    trace = [1,1,2,0,1,-8,4,-2,6,3,9,7,-5,-34,4,7,5,3,-62,-8,6,14,3,6,78,3,6,7,]
    split_trace_in_levels(trace, do_abs=True)


    simple_period_trace = [0,1,0,1,0,1,0,1]
    none_simple_period_trace = [1,0,0,0,0,0,1,4,8,7,0,0,0,1]
    print(is_simple_period(simple_period_trace))
    print(is_simple_period(none_simple_period_trace))


    print(is_idle_rate_filter(trace=trace, threshold=[0.5, 1]))

    
