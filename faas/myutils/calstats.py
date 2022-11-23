'''
常用的统计函数
'''
import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 求变异系数
def coefficient_of_varience():
    pass


# 返回cdf中指定分位数之前的最后bucket的位置（分钟）
def find_precentile(cdf, percent, head=False):
    """ Returns the last whole bucket (minute) before the percentile """
    for i, value in enumerate(cdf):
        if percent < value:
            if head:
                return max(0, i-1)
            else:
                return min(i+1, len(cdf))
    return len(cdf)


# 计算序列的统计特征
# 平均调用率
def cal_avg_rate(trace):
    avg_rate = sum(trace) / len(trace)
    return avg_rate


# 空占比
def cal_idle_rate(trace):
    idle_counter = 0
    for invocation in trace:
        if invocation == 0:
            idle_counter += 1
    idle_rate = idle_counter / len(trace)
    return idle_rate



