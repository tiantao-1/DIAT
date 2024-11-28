import copy
import csv
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
import time
import psutil
import os
import tracemalloc
import threading


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


# 采样函数
def Samples(data, lookback, start_index, stop_index, step):
    # 初始化窗口，载入数据，取一个窗口长度的数据
    i = start_index
    while 1:
        # 生成索引数组
        rows = np.arange(i, i + lookback)
        # 生成列表存储索引值
        index_list = np.zeros((lookback))
        # 如果窗口超出数据范围，则停止
        if i + len(rows) > stop_index:
            break
        # 窗口向后移动
        i += step
        # 建立一个空数组
        samples = np.zeros((len(rows)))
        # 截取窗口处的数据，载入到samples数组中
        for index, row in enumerate(rows):
            index_list[index] = row
            samples[index] = data[row]
        # 迭代器返回索引列表 采样数据窗
        yield index_list, samples

# 标准差
def sigma(data):
    # 求总体均值
    x_bar = sum(data)/len(data)
    # 求方差
    s_2 = []
    for i in data:
        s_2.append((i - x_bar)**2)
    sigma = (sum(s_2)/len(s_2))**0.5
    return round(sigma, 2)


# 检测函数
def WD(T_max,data,mi_mean,h, u):
    # 初始化 k_cusum  g_plus  g_minus mi_det
    k_cusum, g_plus, g_minus, dt_up, dt_down = 0, 0, 0, 0, 0
    while k_cusum < T_max:

        # 计算检测均值
        temp = data[k_cusum]
        # 若均值正偏移量超过beta
        if temp >= mi_mean + u:
            # 正偏移
            g_plus = g_plus + temp - mi_mean
            dt_up += 1
        # 若均值负偏移量超过beta
        elif temp < mi_mean - u:
            # 负偏移
            g_minus = g_minus + mi_mean - temp
            dt_down += 1
        # 若正偏移超过h
        if g_plus > h:
            # 返回事件窗口位置
            return [dt_up, 1, g_plus, k_cusum+1]
        # 若负偏移超过h
        elif g_minus > h:
            return [dt_down, 0, -g_minus, k_cusum+1]
        k_cusum += 1
    return 0



def Detection(power_data):

    event_index = []  # 初始化事件标签列表
    skip_time = 23   #   用于跳过检测到事件后的读取，初始为23是为了跳过初始读数据的误判，power初始全0，而读取数据时如果稳定功率不为0且大于阈值会被误判
    csv_count = 0  # 数据在csv中的index
    rear = -1  # 初始化一个变量 rear，通常用于标记数组或列表的尾部位置，初始值为-1
    mean_size = 6  # 均值窗口大小
    data_size = 20  # 滑动窗口数据大小
    window_size = data_size + mean_size  # 滑动窗口大小
    MAX_SIZE = window_size  # 定义一个变量 MAX_SIZE，其值等于变量 window_size，通常用于定义数组或列表的最大容量
    data_arr = [0] * MAX_SIZE  # 创建一个大小为 MAX_SIZE 的列表，初始值全部为0

    # 标准差对比量
    sigma_pre = 0
    # 开始检测信号
    count_det = 0
    count_min = 4  # 标准差检测最小次数

    # 初始化变量
    time1 = 0
    h = 0
    r = 0
    delta_peak_sta = 0
    increment = 0
    sigma_cur = 0
    q = 1
    mi_mean = 0
    win_det = []
    # 初始化最大允许检测延迟时间T_max
    T_max = 20
    # 窗口大小
    lookback = 6 + T_max


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
            if h == 0:
                # 检测窗口
                win_det = power[int(lookback - T_max):]
                # 标准差检测
                sigma_cur = sigma(win_det)

                if sigma_cur > 10:
                    if sigma_cur > sigma_pre:
                        count_det += 1
                    elif sigma_cur <= sigma_pre:
                        if count_det >= count_min:
                            # 取评估窗口
                            win_mean = power[:int(lookback - T_max)]
                            # 更新mi_mean
                            mi_mean = sum(win_mean) / len(win_mean)
                            static_1 = sum(win_det[-2:]) / 2  # 窗口最后的稳态值
                            static_2 = sum(win_det[:2]) / 2  # 窗口最前的稳态值

                            if static_2 > static_1:
                                r = 1
                                delta_peak_sta = 0
                                h = (T_max / r ** 0.5 * sigma_pre + 3 * abs(static_1 - mi_mean)) * q
                            else:
                                r = (max(win_det) - mi_mean) / (static_1 - mi_mean)
                                delta_peak_sta = (max(win_det) - mi_mean) - (static_1 - mi_mean)

                                if r > 0:
                                    h = (T_max / r ** 0.5 * sigma_pre + 3 * abs(static_1 - mi_mean)) * q
                            increment = static_1 - mi_mean
                            time1 = 5
                        elif count_det < count_min and sigma_cur <= sigma_pre:
                            # 信号重置
                            count_det = 0
                            h = 0
            if time1 > 0 and h != 0:
                time1 -= 1
                result = WD(len(win_det), win_det, mi_mean, h, 1)  # 检测事件
                if result == 0:
                    pass
                else:
                    # 储存事件
                    event_index.append(csv_count - result[0] - 1)

                    # 信号重置
                    count_det = 0
                    h = 0
                    time1 = 0
                    skip_time = 5
                win_det.append(power[-1])
                if time1 == 0:
                    h = 0
                count_det = 0
            sigma_pre = sigma_cur
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
        if copy_event[n] - 10 <= i <= copy_event[n] + 10:
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