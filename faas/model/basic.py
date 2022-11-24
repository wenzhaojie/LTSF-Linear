# models 都需继承以下方法
import sys
sys.path.insert(0, "../..")

import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from faas.myutils.metrics import get_metric_dict
from faas.myutils.plot import Plotter
import math


class Basic_model:
    def __init__(self, name="Basic_model", scaler=StandardScaler()):
        # 初始化一些必要的参数  
        # print(f"正在初始化Basic_model!")
        # print(f"Basic_model scaler:{scaler}")
        self.scaler = scaler
        self.name = name
        self.default_extra_parameters = {
            "seq_len": 1440*3,
            "pred_len": 1440,
            "moving_window": 20,
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": False,
        }
        pass

    def get_scaler(self):
        # print(f"self.scaler:{self.scaler}")
        return self.scaler

    
    def get_name(self):
        print(f"self.name:{self.name}")
    

    def train(self, history, extra_parameters=None):
        # 最开始需要训练  
        pass

    def predict(self, history, predict_window, extra_parameters=None):
        # 使用一段历史序列来预测未来的值
        pass


    def rolling_predict(self, history, predict_window, test=None, use_future=True, extra_parameters=None):
        # 滚动预测
        # history: 用于预测的历史数据
        # predict_window: 预测目标长度
        # test: 用于回测预测效果
        '''
        extra_parameters = {
            seq_len: 每一步预测输入模型的长度
            pred_len: 每一步预测输出模型的长度
        }
        '''
        if extra_parameters == None:
            extra_parameters = self.default_extra_parameters

        try:
            seq_len = extra_parameters["seq_len"]
            pred_len = extra_parameters["pred_len"]
            # print(f"type(use_future):{type(use_future)}")
        except Exception as e:
            print(e)

        # 为了方便,转换成list格式的数据
        # print(f"type(history):{type(history)}")
        history = list(history)
        # print(f"type(history):{type(history)}")

        # 预测的结果需要存放
        predict_list = []
        compute_t_list = []
        pointer = 0

        # 是否使用test数据?
        if use_future == True:
            assert len(test) == predict_window
            history_add = list(test)
        else:
            history_add = predict_list


        while len(predict_list) < predict_window:
            # 获取 input
            if pointer < seq_len:
                history_base = history[-seq_len+pointer:]
                # print(f"len(history_base):{len(history_base)}")
                # print(f"len(history_add[:pointer]):{len(history_add[:pointer])}")
                train = history_base + history_add[:pointer]
            else:
                history_base = []
                train = history_base + history_add[pointer-seq_len:pointer]

            # print(f"len(train):{len(train)}")
            start_at_period = int(pointer) % 1440
            extra_parameters["start_at_period"] = start_at_period

            assert len(train) == seq_len
            predict = self.predict(history=train, predict_window=pred_len, extra_parameters=extra_parameters)
            # print(f"len(predict):{len(predict)}")
            assert len(predict) == pred_len
            predict_list.extend(predict)
            # print(f"len(predict_list):{len(predict_list)}")
            pointer += pred_len


        # 如果使用过去的误差比例来修正
        history_error_correct = extra_parameters["history_error_correct"]
        predict = predict_list
        max_error = 1

        if history_error_correct:
            corrected_predict = []
            for index, (pred, true) in enumerate(zip(predict, test)):

                if index == 0: # 最开始不做修正
                    corrected_value = pred
                    if pred < 1: # 如果预测值小于1，不做修正
                        history_error = 1
                    else:
                        history_error = float(true / pred)
                        if history_error < 1-max_error:
                            history_error = 1-max_error
                        elif history_error > 1+max_error:
                            history_error = 1+max_error
                        else:
                            pass
                    
                else:
                    corrected_value = history_error * pred
                    if pred < 1:
                        history_error = 1
                    else:
                        history_error = float(true / pred)
                        if history_error < 1-max_error:
                            history_error = 1-max_error
                        elif history_error > 1+max_error:
                            history_error = 1+max_error
                        else:
                            pass
                corrected_predict.append(corrected_value)
            predict = np.array(corrected_predict)

        # 最后收尾阶段
        # 有可能预测超出了我们所需要的
        rolling_predict= predict[:predict_window]
        return np.array(rolling_predict)


    def evaluate(self, train, test, extra_parameters=None, plotter=None):
        '''
        用于评估模型的效果
        train: 训练数据
        test: 预测真实的数据
        is_scaler: 是否使用归一化
        use_future: 是否用真实的值来滚动测试性能
        is_round: 是否将预测的连续结果转换成整数
        '''
        use_future = extra_parameters["use_future"]
        is_round = extra_parameters["is_round"]
        is_scaler = extra_parameters["is_scaler"]

        # 先做归一化
        if is_scaler == True:
            combined_trace = np.concatenate((np.array(train), np.array(test)))
            processed_trace = self.scaler.fit_transform(combined_trace.reshape(-1,1)).reshape(-1,)
            train = processed_trace[:len(train)]
            test = processed_trace[len(train):]
        else:
            train = np.array(train)
            test = np.array(test)
        
        # 再训练
        start_t = time.time()
        self.train(history=train, extra_parameters=extra_parameters)
        train_t = time.time() - start_t


        # 滚动预测
        start_t = time.time()
        predict = self.rolling_predict(
            history=train, 
            predict_window=len(test),
            test=test,
            use_future=use_future, 
            extra_parameters=extra_parameters
        )
        predict_t = time.time() - start_t # 计算时间

        
        # 做还原归一化
        if is_scaler == True:

            predict = self.scaler.inverse_transform(predict.reshape(-1,1)).reshape(-1,)
            test = self.scaler.inverse_transform(test.reshape(-1,1)).reshape(-1,)
            # print(f"归一化完成后, predict.shape:{predict.shape}")
            # print(f"归一化完成后, test.shape:{test.shape}")


        # 如果四舍五入
        if is_round:
            # print(f"predict:{predict}")
            round_predict = []
            for num in predict:
                # print(f"num:{num}")
                round_predict.append(max(0,round(num)))
            # predict = [max(0,round(num)) for num in predict]
            predict = round_predict

        print(f"flag")

        # 指标计算
        metrics_dict = get_metric_dict(y_pred=predict, y_test=test)

        # 收集日志
        log_dict = {
            "model":self.name,
            "train_length":len(train),
            "test_length":len(test),
            "predict_t":predict_t,
            "train_t":train_t
        }
        log_dict.update(extra_parameters)
        log_dict.update(metrics_dict)
        print(f"log:{log_dict}")

        
        # 画图
        if plotter != None:
            print(f"开始画图")
            plotter.plot_lines(
                y_list=[predict, test],
                y_label_list=["Predict", "Real"],
                x_label="Timestamp (min)",
                y_label="Concurrency",
                title=plotter.title,
                save_root=plotter.save_root,
                filename=f"Predict vs Real {self.name}",
            )
            # for i in range(24):
            #     plotter.plot_lines(
            #         y_list=[predict[i*60:(i+1)*60], test[i*60:(i+1)*60]],
            #         y_label_list=["Predict", "Real"],
            #         x_label="Timestamp (min)",
            #         y_label="Concurrency",
            #         title=plotter.title,
            #         save_root=plotter.save_root,
            #         filename=f"Evaluation {i}th Hour {self.name}",
            #     )

        return log_dict, predict

        
        





    
