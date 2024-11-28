from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.ndimage import median_filter
import time
import psutil
import os
import tracemalloc
import threading
import copy


# 读取数据
'''公开数据集UK-DALE'''
# path = r'F:\研究生\论文\nilmtk\程序\自适应划窗\数据集\UK-DALE数据集\源数据集\power.csv'
# path1 = r'F:\研究生\论文\nilmtk\程序\自适应划窗\数据集\UK-DALE数据集\源数据集\label.csv'

'''公开数据集REDD'''
# path = r'F:\研究生\Nilmtk\数据集及程序\事件检测算法\自有数据\用于事件检测新\power1102.csv'
# path1 = r'F:\研究生\Nilmtk\数据集及程序\事件检测算法\自有数据\用于事件检测新\label1102.csv'

'''实际采集数据'''
path = r'F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\金卡数据集\金卡数据\金卡测试数据\金卡A相功率.csv'
path1 = r'F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\金卡数据集\金卡数据\金卡测试数据\label.csv'

df1 = pd.read_csv(path1, index_col=[0])
label = df1.values.tolist()
df = pd.read_csv(path)
array_p = df.values[:, 1]
data = array_p[:]
power_data = median_filter(data, size=9)  # 对读取的数据进行滤波

def goodness_of_fit_event(
    pre_event: np.ndarray,
    pos_event: np.ndarray,
    event_threshold: float = 20.0,
):
    """事件的卡方拟合优度检验

    参数：
        pre_event：事件发生前的观测值
        pos_event：事件发生后的观测值
        event_threshold：最小变化检测阈值

    返回：
        tuple[float, float, float]: 卡方统计量，p值，事件发生前的平均值（mu pre-vent），事件发生后的平均值（mu post-event）
    """
    n_measurements = np.array(pos_event).shape[0]  # 获取观测值的数量
    df = np.array(pos_event).shape[0] - 1  # 计算自由度
    p_value = np.ones(n_measurements)  # 初始化p值为1
    statistic = np.zeros(n_measurements)  # 初始化卡方统计量为0
    mu_pre = np.mean(pre_event, axis=0)  # 计算事件发生前的平均值
    mu_pos = np.mean(pos_event, axis=0)  # 计算事件发生后的平均值
    pre_event += 1e-16  # 为了避免除以零，给pre_event添加一个非常小的数
    state_delta = mu_pos - mu_pre  # 计算状态变化量
    # pos_event_norm = np.sum(pre_event) / np.sum(pos_event) * pos_event  # 这行代码被注释掉了，可能是用于标准化pos_event的
    if (np.abs(state_delta) > event_threshold).any():  # 如果状态变化量超过阈值
        statistic = np.sum((pos_event - pre_event) ** 2 / pre_event, axis=0)  # 计算卡方统计量
        p_value = chi2.sf(statistic, df)  # 计算p值
    return statistic, p_value, mu_pre, mu_pos  # 返回卡方统计量，p值，事件发生前的平均值和事件发生后的平均值


def online_events(samples):
    # 在线检测事件
    event_threshold = 80  # 初始化gof阈值
    stat_threshold = 0.05   # 初始化事件判断的p阈值
    stat_idx = 10

    pre_event_samples = np.array(samples[0: stat_idx], dtype=np.float64)  # 滑动窗口前半的功率数据
    pos_event_samples = np.array(samples[stat_idx: len(samples)], dtype=np.float64)   # 滑动窗口后半的功率数据
    stat, gof, mu_pre, mu_pos = goodness_of_fit_event(
        pre_event=pre_event_samples,
        pos_event=pos_event_samples,
        event_threshold=event_threshold,
    )  # 调用函数计算拟合优度等统计量

    # 仅在统计显著时添加统计数据
    if (gof < stat_threshold).any():
        return True  # 返回检测标志和事件对象

def Detection(power_data):
    '''

    Args:
        power_data: 输入csv中读取到的所有功率数据

    Returns:返回所有事件大概位置的标签

    '''
    # 初始化参数
    csv_count = 0  # 数据在csv中的index
    rear = -1  # 初始化一个变量 rear，通常用于标记数组或列表的尾部位置，初始值为-1
    window_size = 20  # 滑动窗口大小
    MAX_SIZE = window_size  # 定义一个变量 MAX_SIZE，其值等于变量 window_size，通常用于定义数组或列表的最大容量
    data_arr = [0] * MAX_SIZE  # 创建一个大小为 MAX_SIZE 的列表，初始值全部为0
    event_index = [] # 初始化事件标签列表
    skip_time = 23   #   用于跳过检测到事件后的读取，初始为23是为了跳过初始读数据的误判，power初始全0，而读取数据时如果稳定功率不为0且大于阈值会被误判

    # 运行检测程序
    while 1:
        data = power_data[csv_count]  # 从队列中获取数据
        csv_count += 1
        # 到数据末尾，结束事件检测
        if csv_count >= len(power_data):
            return event_index
        rear = (rear + 1) % MAX_SIZE  # 当 rear 达到 MAX_SIZE 时，模运算会使其回到 0
        data_arr[rear] = data  # 将 data 赋值给缓冲区的当前尾部位置
        power = data_arr[rear + 1:] + data_arr[:rear + 1]  # 构造功率数据数组
        if skip_time <= 0:
            if online_events(power):
                event_index.append(csv_count-1)
                skip_time = 20
        else:
            skip_time -= 1


'''测试资源消耗'''
'''
# 获取当前脚本的进程ID
current_pid = os.getpid()

# 找到进程对象
for proc in psutil.process_iter(['pid', 'name']):
    if proc.info['pid'] == current_pid:
        process = proc
        process_name = process.name()
        break
else:
    print(f"Process {current_pid} not found.")


# 记录初始资源使用情况
start_time = time.time()
start_cpu = process.cpu_percent(interval=1)
start_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"Monitoring {process_name}...\n")
print(f"Initial CPU Usage: {start_cpu}%")
print(f"Initial Memory Usage: {start_memory:.2f} MB")

# 开始跟踪内存分配
tracemalloc.start()

# 定义一个函数来定期采样CPU使用率
cpu_usage = []

def monitor_cpu():
    while True:
        cpu_usage.append(process.cpu_percent(interval=0.1))
        time.sleep(0.1)

# 启动一个线程来监控CPU使用率
cpu_thread = threading.Thread(target=monitor_cpu)
cpu_thread.start()
'''

start_time = time.time()

# 执行目标代码
all_events_index = Detection(power_data)

end_time = time.time()
# 计算总运行时间
total_time = end_time - start_time
print(f"\nTotal Time Taken: {total_time:} seconds")

'''
# 停止CPU监控线程
cpu_thread.join(timeout=1)

# 记录结束时的资源使用情况
end_time = time.time()
end_cpu = process.cpu_percent(interval=1)
end_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB

# 获取内存分配峰值
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\nFinal CPU Usage: {end_cpu}%")
print(f"Final Memory Usage: {end_memory:.2f} MB")
print(f"Peak Memory Usage: {peak / (1024 * 1024):.2f} MB")
print(f"Peak CPU Usage: {max(cpu_usage):.2f}%")

# 计算总运行时间
total_time = end_time - start_time
print(f"\nTotal Time Taken: {total_time:.2f} seconds")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
'''

print("检测事件数",len(all_events_index))
print("检测完成")



# 获取检验标准，前后标签不同时说明事件有发生
lis = []
for j in range(len(label[0])):
    for i in range(1, len(label)):
        if label[i - 1][j] != label[i][j]:
            lis.append(i)

'''
# 储存所有正确事件的标签
save_all_data_index_path = r"F:\\研究生\\论文\\nilmtk\\程序\\自适应划窗\\数据集\\UK-DALE数据集\\处理结果集\\all_data_index.csv"
with open(save_all_data_index_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([item] for item in lis)  # 写入数据

all_data_temp = []
for z in range(len(lis)):
    all_data_temp.append(data[lis[z]-10:lis[z]+30])
# 储存所有正确事件的数据
save_all_data_path = r"F:\\研究生\\论文\\nilmtk\\程序\\自适应划窗\\数据集\\UK-DALE数据集\\处理结果集\\all_data.csv"
with open(save_all_data_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_data_temp)  # 写入数据

'''


copy_event = copy.deepcopy(all_events_index)


# 用于储存判断错误的事件数据，该数据是识别算法得出的
error_events_data = []
# 用于储存错误事件的标签
error_events_data_index = []
# 用于储存没检测出的事件的数据
lose_events_data = []
# 所有事件的标签信息复制
lis_copy = copy.deepcopy(lis)


# 判断检测的事件是否正确，这个判断方法应该有一定误差，具体有没有误差要看具体数据
TP = 0
temp_index = 0
for i in lis:
    count=0
    # 用于调试程序
    # if i == 21397:
    #     a = 0
    for n in range(len(copy_event)):
        count += 1
        if copy_event[n] - 5 <= i <= copy_event[n] + 5:
            TP += 1
            # 得到错误事件的总和信息
            copy_event.pop(n)
            # 得到丢失事件的总和标签信息
            lis_copy.pop(temp_index)
            break
        if count == len(copy_event):
            temp_index += 1


# 错误事件的数据列表每个列表数据包含起始点到结束点再加20个数据
for k in range(len(copy_event)):
    temp_data = data[copy_event[k]-20:copy_event[k]+20]
    error_events_data.append(temp_data)
    error_events_data_index.append([copy_event[k]])

# 丢失事件的真实总和数据
for x in lis_copy:
    lose_events_data.append(data[x-10:x+30])

'''
save_error_data_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\error_data.csv"
save_lose_data_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\lose_data.csv"
save_error_data_index_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\error_data_index.csv"
save_lose_data_index_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\lose_data_index.csv"
with open(save_error_data_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(error_events_data)  # 写入数据
with open(save_lose_data_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lose_events_data)  # 写入数据
with open(save_error_data_index_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(error_events_data_index)  # 写入数据
with open(save_lose_data_index_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([item] for item in lis_copy)  # 写入数据
'''


Re = TP/len(lis)
Pr = TP/(len(all_events_index))
F1 = 2*(Re*Pr/(Re+Pr))
print('事件总数：', len(lis))
print('检测正确的事件数：', TP)
print('精确率：', Pr)
print('召回率：', Re)
print('F1:', F1)