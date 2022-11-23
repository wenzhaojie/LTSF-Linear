import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 求平均平方误差误差
def mse(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    return mse

# 求均方根误差
def rmse(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return rmse

# 求R2
def r2(y_pred, y_test):
    r2 = r2_score(y_test, y_pred)
    return r2

# 求MAE
def mae(y_pred, y_test):
    mae = metrics.mean_absolute_error(y_pred, y_test)
    return mae

# MAPE和SMAPE需要自己实现
def mape(y_pred, y_test):
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    return np.mean(np.abs((y_pred - y_test) / y_test)) * 100

 
def smape(y_pred, y_test):
    return 2.0 * np.mean(np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100


# 误差放大器
class Crane_error:
    def __init__(self):
        pass

    @staticmethod
    def amplify(x):
        res = -math.log(1.0 - x) / math.log(1.25)
        return res

    @staticmethod
    def MAPE(actual, predicted):
        # epsilon
        epsilon = 1e-3
        e = 0 
        assert(len(actual) == len(predicted))
        for (act, pred) in zip(actual, predicted):
            if act < epsilon:
                return "Error"
            if pred < act:
                # If the predicted value is less than the actual one, we amplify the error
                e += Crane_error.amplify((act - pred) / act)
            else:
                e += (pred - act) / act
            e = e / float(len(actual))
        return e

    @staticmethod
    def PredictionError(y_pred, y_test):
        mape = Crane_error.MAPE(y_test, y_pred)
        if mape == "Error":
            # print("1")
            return mae(y_pred, y_test)
        else:
            # print("2")
            return mape
        

# 计算 cold_start_ratio 
def cold_start_ratio(y_pred, y_test, tolerance=0):
    assert len(y_pred) == len(y_test)
    cold_start_count = 0
    for pred, test in zip(y_pred, y_test):
        if test > pred * (1 + tolerance):
            cold_start_count += test - pred
    _cold_start_ratio = cold_start_count / sum(y_test)
    return _cold_start_ratio


# 计算 utilization_ratio 资源利用率
def utilization_ratio(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    total_res = 0
    utilized_res = 0
    for pred, test in zip(y_pred, y_test):
        total_res += max(pred, test)
        utilized_res += min(pred, test)
    _utilization_ratio = utilized_res / total_res
    return _utilization_ratio


# 计算 over_provisioned_ratio 超额分配的调用次数比例
def over_provisioned_ratio(y_pred, y_test, tolerance=0):
    assert len(y_pred) == len(y_test)
    over_provisioned_count = 0
    for pred, test in zip(y_pred, y_test):
        if pred > test * (1 + tolerance):
            over_provisioned_count += pred - test
    _over_provisioned_ratio = over_provisioned_count / sum(y_test)
    return _over_provisioned_ratio



# 计算 cold_start_time_slot_ratio 具有冷启动的时间槽的比例
def cold_start_time_slot_ratio(y_pred, y_test, tolerance=0):
    assert len(y_pred) == len(y_test)
    cold_start_count = 0
    for pred, test in zip(y_pred, y_test):
        if pred * (1 + tolerance) < test:
            cold_start_count += 1
    _cold_start_time_slot_ratio = cold_start_count / len(y_test) 
    return _cold_start_time_slot_ratio



def get_metric_dict(y_pred, y_test):
    # 指标计算
    _rmse = rmse(y_pred=y_pred, y_test=y_test)
    _mse = mse(y_pred=y_pred, y_test=y_test)
    _r2 = r2(y_pred=y_pred, y_test=y_test)
    _cold_start_ratio = cold_start_ratio(y_pred=y_pred, y_test=y_test)
    _utilization_ratio = utilization_ratio(y_pred=y_pred, y_test=y_test)
    _over_provisioned_ratio = over_provisioned_ratio(y_pred=y_pred, y_test=y_test)
    _cold_start_time_slot_ratio = cold_start_time_slot_ratio(y_pred=y_pred, y_test=y_test)

    # crane_error 有时候报错
    try:
        _crane_error = Crane_error().PredictionError(y_pred=y_pred, y_test=y_test)
    except Exception:
        _crane_error = None
        pass

    metrics_dict = {
        "rmse": _rmse,
        "mse": _mse,
        "r2": _r2,
        "crane_error": _crane_error,
        "cold_start_ratio": _cold_start_ratio,
        "utilization_ratio": _utilization_ratio,
        "over_provisioned_ratio": _over_provisioned_ratio,
        "cold_start_time_slot_ratio": _cold_start_time_slot_ratio,
    }

    return metrics_dict


if __name__ == "__main__":
    
    y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
    y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])
    
    # MSE
    print(metrics.mean_squared_error(y_pred, y_test)) # 8.107142857142858
    # RMSE
    print(np.sqrt(metrics.mean_squared_error(y_pred, y_test))) # 2.847304489713536
    # MAE
    print(metrics.mean_absolute_error(y_pred, y_test)) # 1.9285714285714286
    # MAPE
    print(mape(y_pred, y_test)) # 76.07142857142858，即76%
    # SMAPE
    print(smape(y_pred, y_test)) # 57.76942355889724，即58%

    # amplify
    print(Crane_error.amplify(0.1))
    print(Crane_error.amplify(0.5))

    # MAPE_crane
    print(Crane_error.MAPE([0.9999,3.00001,5], [1,3,4]))

    # Crane_error
    print(Crane_error.PredictionError([0.9999,3.00001,5], [1,3,4]))

