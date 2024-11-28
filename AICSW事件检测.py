# 导入头文件
import copy
import csv
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
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

# path = r'F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\金卡数据集\金卡数据\金卡测试数据\shili.csv'


df1 = pd.read_csv(path1, index_col=[0])
label = df1.values.tolist()
df = pd.read_csv(path)
array_p = df.values[:, 1]
data = array_p[:]
data = np.array(data, dtype=float)  # 转换为浮点数
power_data = median_filter(data, size=9)   # 对读取的数据进行滤波


'''
filter_data_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\filter_data.csv"
with open(filter_data_path, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows([item] for item in power_data)  # 写入数据
'''


def PointsSlidingWindow(sample_data, change_point_power):
    '''
    计算巡检窗口的积分值的绝对值和标签

    :param sample_data: 滑窗的事件数据
    :param change_point_power: 滑窗中变点的功率值
    :return: 返回为列表
            list[0]为所有事件的巡检窗口积分绝对值
            list[1]为所有事件的巡检窗口积分绝对值对应的标签
            list[2]为滑窗误判标志
    '''

    R_S = 0.3   # 对于公开数据集该比例取0.5，私有数据集取0.3
    points_sliding_misjudge_flag = 0    #   滑窗误判标志
    points_sliding_window_threshold = change_point_power*4*R_S #   根据变点功率情况作为第二次判断
    points_value_absolute = []  # 用来记录所有巡检窗口积分值的绝对值
    points_value_index = []  # 用来记录points_value_forward_square向前的对应的巡检窗口标签
    i = 0  # 用于while循环
    while 1:
        points_sliding_window_data = sample_data[i:i + 4]  # 积分滑动窗口的数据
        points_sliding_window_points_value = sum(points_sliding_window_data) - points_sliding_window_data[0] * len(
            points_sliding_window_data)  # 用来记录积分值
        points_value_absolute.append(abs(points_sliding_window_points_value))  # 保存积分值的绝对值
        points_value_index.append(i)  # 保存标签
        i += 1
        if i+4 == len(sample_data)+1:
            break


    #   判断是否是滑动窗口误判了
    if max(points_value_absolute) <= points_sliding_window_threshold:
        points_sliding_misjudge_flag = 1
        # print("滑动窗口误判!!!!!!")



    # print("**************************************************")
    # print("points_value_absolute", points_value_absolute)
    # print("points_value_index", points_value_index)
    # print("**************************************************")


    events_threshold = max(points_value_absolute)*0.4   #   事件判断阈值，用于判断事件数和定位
    incident_value_absolute = []    #   所有事件的巡检窗口积分绝对值
    incident_value_index = []   #   对应的标签

    # 记录所有事件
    for k in range(len(points_value_absolute)):
        if points_value_absolute[k] >= events_threshold:
            incident_value_absolute.append(points_value_absolute[k])
            incident_value_index.append(points_value_index[k])

    return [incident_value_absolute, incident_value_index, points_sliding_misjudge_flag]

def EventLocate(csv_count, incident_value_index):
    '''
    进行事件的定位

    :param csv_count: 当前单个功率数据在csv文件中的标签
    :param incident_value_index: 巡检窗口所有事件的积分绝对值对应的标签
    :return: 返回当前事件起始点的标签值，可能有多个
    '''

    temporary_event_start_index = []    #   用于拆分多个事件的标签
    index_list = [] #   事件起始对应在incident_value_index里的标签
    current_event_start_index = []  #   事件的起始标签
    for i in range(len(incident_value_index)-1):
        if incident_value_index[i]+1 != incident_value_index[i+1]:
            index_list.append(i+1)
    if len(index_list) > 0:
        temporary_event_start_index.append(incident_value_index[0]+2)
        for i in index_list:
            temporary_event_start_index.append(incident_value_index[i]+2)
    else:
        temporary_event_start_index.append(incident_value_index[0]+2)
    for i in temporary_event_start_index:
        current_event_start_index.append(csv_count-20+i)

    return current_event_start_index



def Detection(power_data, minimum_power):
    '''
    进行事件检测

    :param power_data: 输入csv中读取到的所有功率数据
    :param minimum_power: 输入当前csv的功率数据中电器最小的功率值
    :return: 返回所有事件起始位置的标签、所有事件巡检窗口的积分绝对值、所有事件巡检窗口的积分绝对值的对应标签
    '''

    '''数据初始化'''
    mean_size = 3  # 均值窗口大小
    data_size = 20  # 滑动窗口数据大小
    window_size = data_size + mean_size  # 滑动窗口大小
    change_point_power = 0  #   变点的功率情况
    csv_count = 0  # 数据在csv中的index
    rear = -1  # 初始化一个变量 rear，通常用于标记数组或列表的尾部位置，初始值为-1
    MAX_SIZE = window_size  # 定义一个变量 MAX_SIZE，其值等于变量 window_size，通常用于定义数组或列表的最大容量
    data_arr = [0] * MAX_SIZE  # 创建一个大小为 MAX_SIZE 的列表，初始值全部为0
    threshold_ratio = 0.8   #   阈值比例，用于降低事件判断标准
    skip_time = 23   # 用于跳过检测到事件后的读取，初始为23是为了跳过初始读数据的误判，power初始全0，而读取数据时如果稳定功率不为0且大于阈值会被误判
    event_count = 0 #   用于事件计数


    all_events_index = []   #   储存所有事件起始位置的标签
    all_events_incident_value_absolute = [] #   储存储存所有事件巡检窗口的积分绝对值
    all_events_incident_value_index = []    #   储存储存所有事件巡检窗口的积分绝对值的对应标签


    while 1:
        data = power_data[csv_count]  # 从队列中获取数据
        csv_count += 1
        # 到数据末尾结束循环
        if csv_count == len(power_data):
            return all_events_index, all_events_incident_value_absolute, all_events_incident_value_index
        rear = (rear+1) % MAX_SIZE  # 当 rear 达到 MAX_SIZE 时，模运算会使其回到 0
        data_arr[rear] = data  # 将 data 赋值给缓冲区的当前尾部位置
        power = data_arr[rear + 1:] + data_arr[:rear + 1]   # 构造功率数据数组
        if skip_time <= 0:
            mean = sum(power[:mean_size])/mean_size # 计算均值
            sample_data = power[mean_size:] # 提取样本数据
            #   进行滑动窗口对应的积分值的阈值的自适应
            change_point_power = max(abs(max(sample_data)-mean), abs(min(sample_data)-mean))
            if change_point_power < minimum_power*0.7:
                change_point_power = minimum_power
            window_threshold = change_point_power*10*threshold_ratio    # 根据变点功率更新滑窗阈值，乘10为了控制变点位置靠近滑动窗口中间位置
            points_value = sum(sample_data) - mean * len(sample_data)   # 计算数据窗口对应的积分值

            # print("++++++++++++++++++++++++++++++++++++++++++++++")
            # print("功率数据", power)
            # print("变点功率", change_point_power)
            # print("滑窗阈值", window_threshold)
            # print("滑窗积分值", points_value)
            # print("++++++++++++++++++++++++++++++++++++++++++++++")

            '''事件判断'''
            if abs(points_value) >= window_threshold:
                slide_result = PointsSlidingWindow(sample_data, change_point_power)
                incident_value_absolute = slide_result[0]   # 所有事件的巡检窗口积分绝对值
                incident_value_index = slide_result[1]  # 巡检窗口积分绝对值对应的标签
                points_sliding_misjudge_flag = slide_result[2]  #   滑窗误判标志
                #   滑窗误判
                if points_sliding_misjudge_flag == 1:
                    skip_time = 18
                    continue
                current_event_start_index = EventLocate(csv_count, incident_value_index)    #   当前事件起始位置，可能有多个事件
                event_count = event_count+len(current_event_start_index)


                # print("-----------------------------------------")
                # print("检测到第{:d}个事件".format(event_count))
                # print("对应事件的起始点标签：", current_event_start_index)
                # print("对应事件的巡检窗口积分绝对值：", incident_value_absolute)
                # print("对应事件的巡检窗口积分绝对值对应的标签：", incident_value_index)
                # print("-----------------------------------------")

                for i in current_event_start_index:
                    all_events_index.append(i)
                all_events_incident_value_absolute.append(incident_value_absolute)
                all_events_incident_value_index.append(incident_value_index)

                skip_time = 18
        else:
            skip_time -= 1





'''
    UK-DALE最低功率41
    REDD最低功率65
    私有数据集金卡270
'''
minimum_power = 270 # 电器数据中的最小稳态功率Mp

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
all_events_index, all_events_incident_value_absolute, all_events_incident_value_index = Detection(power_data, minimum_power)

# print("all_events_index", all_events_index)

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
save_all_data_index_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\all_data_index.csv"
with open(save_all_data_index_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([item] for item in lis)  # 写入数据

all_data_temp = []
for z in range(len(lis)):
    all_data_temp.append(data[lis[z]-10:lis[z]+30])
# 储存所有正确事件的数据
save_all_data_path = r"F:\研究生\论文\事件检测论文\基于AICSW事件检测算法的电器能效分析\24年9月版\程序\输出结果\all_data.csv"
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
    for n in range(len(copy_event)):
        count += 1
        if copy_event[n] - 3 <= i <= copy_event[n] + 3:
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
    temp_data = data[copy_event[k]-10:copy_event[k]+10]
    error_events_data.append(temp_data)
    error_events_data_index.append([copy_event[k]])

# 丢失事件的真实总和数据
for x in lis_copy:
    lose_events_data.append(data[x-10:x+10])

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