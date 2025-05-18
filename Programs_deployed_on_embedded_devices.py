import pandas as pd
import serial.tools.list_ports
import time
import pymysql
import numpy as np
from dtw import *
from threading import Thread
from multiprocessing import Event, freeze_support
from queue import Queue
import serial
import binascii
import datetime
from datetime import timedelta
from scipy.signal import medfilt
import threading


# -------------定时发送------------------
# 定义一个函数，用于检查时间并在整分钟时发送信号
def check_time_and_send_signal():
    while True:
        now = datetime.datetime.now()  # 获取当前时间
        # 计算下一次整分钟的时间
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        # 计算从现在到下一次整分钟的等待时间（秒）
        wait_time = (next_minute - now).total_seconds()
        # 使用 threading.Timer 在等待时间后调用 send_signal 函数
        threading.Timer(wait_time, send_signal).start()
        # 等待到下一次整分钟
        time.sleep(wait_time)


# 定义一个函数，用于发送信号（在这里是打印当前时间）
def send_signal():
    global count_for_minute
    global count_for_minute_flag
    # current_time = datetime.now().replace(second=0, microsecond=0)  # 获取当前整分钟的时间
    # # 格式化时间字符串，精确到2位数的毫秒
    # formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-6] + '00'
    with count_lock:
        count_for_minute += 1
        count_for_minute_flag = 1
    # print(f"Signal sent at {current_time}Signal sent at {formatted_time}")  # 打印信号发送的时间




# -------------json转十六进制------------------
def json_to_hex(json_string):
    # 将字符串转换为字节序列
    byte_sequence = json_string.encode('utf-8')
    # 使用hexlify函数将字节序列转换为十六进制
    hex_representation = binascii.hexlify(byte_sequence)

    return hex_representation

# -------------用来改变json里的值，发送碳排用------------------
def json_send_carbon_emission(id_counter, carbon_emission, carbon_emission_total):
    # 获取当前时间
    current_time = datetime.datetime.now().replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S.%f')[:-6] + '50'

    # 用generate_json_string生成的字符串替换原始字符串中的占位符
    json_string = r'{\"Carbon_emission\":%s,\"Carbon_emission_total\":%s,\"Photovoltaic_emission_reduction\":%s}' % (carbon_emission, carbon_emission_total, 0)

    # 构建最终的字符串
    final_string = f'{{"Event":{{"Profile":["Time","Index","Event","Description"],"Records":[["{current_time}",{id_counter},2000,"{json_string}"]]}}}}'

    return final_string



# -------------用来改变json里的值，发送功率用------------------
def json_send_power(id_counter, power):
    # 获取当前时间
    current_time = datetime.datetime.now().replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S.%f')[:-6] + '00'

    # 用generate_json_string生成的字符串替换原始字符串中的占位符
    json_string = r'{\"PowerA\":%s,\"PowerB\":%s,\"PowerC\":%s}' % (power[0], power[1], power[2])

    # 构建最终的字符串
    final_string = f'{{"Event":{{"Profile":["Time","Index","Event","Description"],"Records":[["{current_time}",{id_counter},2002,"{json_string}"]]}}}}'

    return final_string

# -------------用来改变json里的值，发送事件用------------------
def json_send_event(id_counter, state, event_number, G=0, all_carbon_emissions = 0):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]

    # 0为灯，1为锂电池，2为热水壶
    if event_number == 0:
        # 用generate_json_string生成的字符串替换原始字符串中的占位符
        json_string = r'{\"Light\":%s}' % (state)

    if event_number == 1:
        # 用generate_json_string生成的字符串替换原始字符串中的占位符
        json_string = r'{\"Lithium_battery\":%s}' % (state)

    if event_number == 2 and state == 1:
        # 用generate_json_string生成的字符串替换原始字符串中的占位符
        json_string = r'{\"Electric_Kettle\":%s}' % (state)
    if event_number == 2 and state == 0:
        # 用generate_json_string生成的字符串替换原始字符串中的占位符
        json_string = r'{\"Electric_Kettle\":%s,\"Event_carbon_emission\":%s,\"Cumulative_carbon_emission\":%s}' % (state, G, all_carbon_emissions)

    # 构建最终的字符串
    final_string = f'{{"Event":{{"Profile":["Time","Index","Event","Description"],"Records":[["{current_time}",{id_counter},2001,"{json_string}"]]}}}}'

    return final_string



# -------------ttl串口通信------------------
 # original_string格式类似于下面的
# original_string = '{"Event":{"Profile":["Time","Index","Event","Description"],"Records":[["2024-08-13 14:20:00.00",25,2000,"{\\"lidianchi\\":0,\\"reshuihu\\":0}"]]}}'
# original_string = '{"Event":{"Profile":["Time","Index","Event","Description"],"Records":[["2024-08-13 14:20:00.00",25,2000,"{\\"Carbon_emission\\":0,\\"Power\\":0}"]]}}'
def ttl_chuankou(original_string):

    # 串口配置
    ser = serial.Serial(
        port='/dev/ttyAMA0',  # 串口设备，根据实际情况修改
        # port='COM6',  # 串口设备，根据实际情况修改
        baudrate=115200,  # 波特率
        timeout=1  # 超时时间，单位秒
    )
    # 打开串口
    if not ser.is_open:
        ser.open()
    try:
        # 发送数据
        hex_string = json_to_hex(original_string)
        hex_string = hex_string.decode('utf-8')
        crc16 = calc_crc16(hex_string)
        hex_string = hex_string + crc16
        byte_data = binascii.unhexlify(hex_string)
        # print(byte_data)
        ser.write(byte_data)

        # 读取串口数据
        received_data = ser.readline()
        if received_data:
            print("Received:", received_data)

    except KeyboardInterrupt:
        pass

    # 关闭串口
    ser.close()



# --------------------------------串口通信和数据采集模块----------------------------------
def get_elec(ser):
    # har 为谐波设置命令
    # 办公室读所有数据
    w = '030300400032C429'

    # 十六进制 字符串 转int
    int_w = int(w, 16)
    # int 转 字节
    b_w = int_w.to_bytes(len(w) // 2 , byteorder='big', signed=False)
    # print(b_w)
    ser.write(b_w)
    # 串口接收字节数，据情况修改
    # 读前半
    receive_data = ser.read(105)
    # print("Data:", receive_data)
    data = receive_data.hex().upper()
    # print("Data:", data)
    return data


# -------------crc校验------------------
def calc_crc16(string):  # Modbus-16 CRC校验
    data = bytearray.fromhex(string)
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for i in range(8):
            if(crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    result = ((crc & 0xff) << 8) + (crc >> 8)
    result = '0' * (4 - len(hex(result)[2:])) + hex(result)[2:]
    return result


# ---------数据类型转换---------
# 16转10
def hex_to_decimal(data):
    if type(data) == str:
        dec_one = int(data, 16)
        # 下面这句转换有问题
        # dec_one = int.from_bytes(bytes.fromhex(data), byteorder="little", signed=True)
        return dec_one
    else:
        # print('类型错误，应为list或str')
        return 0


def calc(data, ser):
    data_value = data[:-4]
    crc_value = calc_crc16(data_value).upper()
    add = 68
    '''多相'''
    if crc_value == data[-4:]:

        valA = hex_to_decimal(data[6+add:14+add])/10.0   # A相功率,除以10.0进行换算
        valB = hex_to_decimal(data[14+add:22+add])/10.0  # B相功率
        valC = hex_to_decimal(data[22+add:30+add])/10.0 # C相功率
        val_all = hex_to_decimal(data[30+add:38+add])/10.0 # 总功率
        val_Wh = hex_to_decimal(data[38+add:46+add])/1000  # 当前电能（类似于电表读数）
        sourse_hex = data   # 原十六进制
        val = [valA,valB,valC,val_all,val_Wh,sourse_hex]
        # print("val:", val)
        return val
    else:
        # print("crc_value",crc_value)
        # print("data_last_4",data[-4:])
        # valA = hex_to_decimal(data[6 + add:14 + add]) / 10.0  # A相功率,除以10.0进行换算
        # print("valA:", valA)
        ser.flushInput()
        return 'ser data error'






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





def ColData(q_col2de, q_col2de_2, q_ser, q_col2base, q_col2base_carbon_emission):
    """
    从串口接收数据，处理后将结果放入指定的队列中。

    参数:
    q_col2de (queue): 用于存放处理后数据的队列。
    q_ser (queue): 用于存放串口数据的队列。
    q_col2base (queue): 第三个用于存放处理后数据的队列。
    """
    global count_for_minute
    global count_for_minute_flag
    # 用来标记电能是否要读取，0是没到15分钟，1是到了15分钟
    dian_neng_flag = 0
    # 用于储存电能
    dianneng = []
    # 光伏发电比例,即光伏发电量占总用电量的比例（0到1之间）
    Photovoltaic_power_generation_ratio = 0
    # 电力二氧化碳排放因子（kgCO2/kWh）,浙江省2021年的为0.5422
    EF = 0.5422
    # 用于给4G的事件计数
    id_counter_power = 1
    id_counter_carbon = 1

    ser = ''  # 初始化串口变量
    while 1:  # 无限循环，直到遇到break语句
        # 检查串口数据队列是否为空
        if not q_ser.empty():
            # 如果队列不为空，从队列中取出数据
            ser = q_ser.get()

        # 如果取出的数据不为空
        if ser != '':
            with count_lock:
                if count_for_minute < 15:
                    # 调用 get_elec 函数处理串口数据
                    elec = get_elec(ser)
                    # 调用 calc 函数计算数据
                    data = calc(elec, ser)

                    # 如果计算结果为 'error'，打印错误并退出循环
                    if data == 'ser data error':
                        print('ser data error')
                        continue
                    # 将处理后的数据放入三个不同的队列中
                    q_col2de.put(data)
                    q_col2de_2.put(data)
                    # q_col2base.put(data)
                    if count_for_minute_flag == 1:
                        count_for_minute_flag = 0
                        # 0是A相（插座）,2是C相（日光灯）
                        power = [data[0], data[1], data[2]]
                        # 向TTL发送功率数据
                        power_string = json_send_power(id_counter_power, power)
                        id_counter_power += 1
                        # print("power_string", power_string)
                        ttl_chuankou(power_string)

            with count_lock:
                if count_for_minute >= 15:
                    count_for_minute = 0
                    dianneng.append(data[4])
                    # print("dianneng",dianneng)
                    
                    # 碳排计算
                    if len(dianneng) >= 2:
                        # E为15分钟用电电能
                        E = (dianneng[-1] - dianneng[-2])
                        # E_totel为当前熔断器电能
                        E_totel = dianneng[-1]
                        carbon_emission = (1 - Photovoltaic_power_generation_ratio) * E * EF
                        carbon_emission_total = (1 - Photovoltaic_power_generation_ratio) * E_totel * EF
                        # q_col2base_carbon_emission.put(carbon_emission)

                        # 向TTL发送碳排数据
                        carbon_emission_string = json_send_carbon_emission(id_counter_carbon, carbon_emission, carbon_emission_total)
                        id_counter_carbon += 1
                        # print("carbon_emission_string", carbon_emission_string)
                        ttl_chuankou(carbon_emission_string)

                # 清洗dianneng的数据
                if len(dianneng) >= 10000:
                    dianneng = dianneng[-3:]

        # 休眠0.2秒，减少CPU占用
        time.sleep(0.2)


def Put2base(q_col2base, q_flag, e_base, q_res2base, q_res2base_2, q_col2base_carbon_emission):
    """
    将数据插入数据库。

    参数:
    q_col2base (queue): 包含功率数据的队列。
    q_flag (queue): 控制标志队列。
    e_base (event): 事件对象，用于等待信号。
    q_res2base (queue): 包含分类结果的队列。
    """
    e_base.wait()  # 等待事件信号
    # 连接到数据库
    db = pymysql.connect(host='localhost',
                         user="pcroot",
                         password="123456",
                         db='mysql',
                         port=3306,
                         autocommit=True)
    # 数据库操作，获取游标
    cursor = db.cursor()
    num_power = 0  # 初始化功率数据计数器
    num_dianneng_c = 0  # 初始化功率数据计数器
    num_dianneng_a = 0  # 初始化功率数据计数器
    print('数据库已连接')  # 打印连接信息

    while 1:  # 无限循环，直到程序终止
        if not q_flag.empty():  # 检查控制标志队列是否为空
            # 如果队列不为空，执行表的创建或删除操作
            # cursor.execute("DROP TABLE IF EXISTS PowerData_bangongshi_0905")  # 如果存在表则重新创建
            # 使用 execute() 方法执行 SQL 查询
            creatTab = """CREATE TABLE PowerData_bangongshi_0905(
                        DataID INT,
                        PowerA DOUBLE,
                        PowerB DOUBLE,
                        PowerC DOUBLE,
                        PowerAll DOUBLE,
                        DianNeng DOUBLE,
                        CarbonEmission DOUBLE,
                        TimeStamp VARCHAR(50),
                        Sourse_hex VARCHAR(255)
            )"""
            # 执行数据库语句
            cursor.execute("SHOW TABLES LIKE 'PowerData_bangongshi_0905'")
            result0 = cursor.fetchone()
            if not result0:
                cursor.execute(creatTab)

            # cursor.execute("DROP TABLE IF EXISTS C_Event_Result")  # 如果存在表则重新创建
            # 使用 execute() 方法执行 SQL 查询
            creatTab = """CREATE TABLE C_Event_Result(
                                    EventID INT,
                                    ED_DTW char(20),
                                    CD_DTW char(20),
                                    DTW char(20),
                                    Vote char(20),
                                    TimeStamp VARCHAR(50)
                                    )"""
            # 执行数据库语句
            cursor.execute("SHOW TABLES LIKE 'C_Event_Result'")
            result0 = cursor.fetchone()
            if not result0:
                cursor.execute(creatTab)

            # cursor.execute("DROP TABLE IF EXISTS A_Event_Result")  # 如果存在表则重新创建
            # 使用 execute() 方法执行 SQL 查询
            creatTab = """CREATE TABLE A_Event_Result(
                                                EventID INT,
                                                ED_DTW char(20),
                                                CD_DTW char(20),
                                                DTW char(20),
                                                Vote char(20),
                                                TimeStamp VARCHAR(50)
                                                )"""
            # 执行数据库语句
            cursor.execute("SHOW TABLES LIKE 'A_Event_Result'")
            result0 = cursor.fetchone()
            if not result0:
                cursor.execute(creatTab)

            print('创建完成')  # 打印创建完成信息
            q_flag.get()  # 从控制标志队列中取出数据

        if not q_col2base.empty():  # 检查功率数据队列是否为空
            data = q_col2base.get()  # 取出数据
            if not q_col2base_carbon_emission.empty():
                data_carbon_emission = q_col2base_carbon_emission.get()
            else:
                data_carbon_emission = 0
            now = datetime.datetime.now()
            timestamp = now.strftime('%m-%d-%H:%M:%S') + f".{int(now.microsecond / 10000):02}"
            sql = "INSERT INTO PowerData_bangongshi_0905 (DataID, PowerA, PowerB, PowerC, PowerAll, DianNeng, CarbonEmission, TimeStamp, Sourse_hex) VALUES('%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%s', '%s')" % (num_power, float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data_carbon_emission), timestamp, data[5])
            cursor.execute(sql)  # 执行插入操作
            num_power += 1  # 更新功率数据计数器


        if not q_res2base.empty():  # 检查分类结果队列是否为空
            result = q_res2base.get()  # 取出分类结果
            timestamp = now.strftime('%m-%d-%H:%M:%S') + f".{int(now.microsecond / 10000):02}"
            sql = "INSERT INTO C_Event_Result (EventID, ED_DTW, CD_DTW, DTW, Vote, TimeStamp)" \
                  "VALUES(%d, '%s', '%s', '%s', '%s', '%s')" \
                  % (num_dianneng_c, result[0], result[1], result[2], result[3], timestamp)
            cursor.execute(sql)  # 执行插入操作
            num_dianneng_c += 1

        if not q_res2base_2.empty():  # 检查分类结果队列是否为空
            result = q_res2base_2.get()  # 取出分类结果
            timestamp = now.strftime('%m-%d-%H:%M:%S') + f".{int(now.microsecond / 10000):02}"
            sql = "INSERT INTO A_Event_Result (EventID, ED_DTW, CD_DTW, DTW, Vote, TimeStamp)" \
                  "VALUES(%d, '%s', '%s', '%s', '%s', '%s')" \
                  % (num_dianneng_a, result[0], result[1], result[2], result[3], timestamp)
            cursor.execute(sql)  # 执行插入操作
            num_dianneng_a += 1

        time.sleep(0.2)  # 休眠0.2秒，减少CPU占用


'''用于C项日光灯监测'''
# 定义一个字典 equip_list，用于存储设备编号和设备名称的映射关系
equip_list_1 = {
    0: '日光灯',  # 设备编号0对应的设备是白炽灯
    1: '小锂电池',   # 设备编号1对应的设备是吹风机
    2: '热水壶',     # 设备编号2对应的设备是风扇
    # 3: '电脑',    # 设备编号3对应的设备是热水壶
    # 4: '暖风机',     # 设备编号2对应的设备是风扇
    # 5: '笔记本',    # 设备编号3对应的设备是热水壶
    # 6: '电风扇',    # 设备编号3对应的设备是热水壶
    3: '异常'
}


'''用于A项锂电池和热水壶监测'''
# 定义一个字典 equip_list，用于存储设备编号和设备名称的映射关系
equip_list_2 = {
    # 0: '日光灯',  # 设备编号0对应的设备是白炽灯
    0: '示波器',   # 设备编号1对应的设备是吹风机
    1: '锂电池',     # 设备编号2对应的设备是风扇
    2: '热水壶',    # 设备编号3对应的设备是热水壶
    3: '小锂电池',     # 设备编号2对应的设备是风扇
    # 5: '未知电器',    # 设备编号3对应的设备是热水壶
    # 6: '电风扇',    # 设备编号3对应的设备是热水壶
    4: '异常'
}


# 定义一个字典 state_list，用于存储设备状态编号和状态名称的映射关系
state_list = {
    0: '关',  # 状态编号0对应的状态是“关”
    1: '开'   # 状态编号1对应的状态是“开”
}



# 使用lambda表达式定义一个函数manhattan_distance，用于计算曼哈顿距离
# 这个函数接受两个参数x和y，代表两个点的坐标
# 曼哈顿距离是所有坐标差的绝对值之和，这里使用numpy的abs函数来计算绝对值
manhattan_distance = lambda x, y: np.abs(x - y)


def DTW(power, templist, phase_flag):
    """
    使用DTW算法对功率数据进行分类。

    参数:
    power (list): 待分类的功率数据列表。
    templist (list of lists): 模板列表，包含多个功率数据模板。
    phase_flag(int): A相还是C相的标志位，0代表A相，1代表C相

    返回:
    tuple: 包含设备编号和状态编号的元组。
    """
    # 计算power数组前两个元素的平均值和后两个元素的平均值的最小值
    min_data = min(sum(power[:2]) / 2, sum(power[-2:]) / 2)
    # 将power数组中的每个元素减去min_data，进行数据标准化
    pending = np.array(power) - min_data
    # 初始化一个列表，用于存储每个模板与待分类数据的DTW距离
    dis_list = []
    # 遍历模板列表
    for j in range(len(templist)):
        temp = templist[j]
        # 计算模板temp和待分类数据pending的DTW距离
        # 使用manhattan_distance作为距离度量函数
        ds, a, b, c = dtw(temp, pending, dist=manhattan_distance) # 等同于下面两句代码，非金卡的距离度量函数
        # DTW_result = dtw(temp, pending, dist_method=manhattan_distance)
        # ds = DTW_result.distance
        # 将计算得到的DTW距离添加到dis_list中
        dis_list.append(ds)
    # 使用numpy的argsort函数找到dis_list中最小值的索引
    min_index = np.argsort(dis_list)[0]

    '''多时间序列计算设备编号'''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 10
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index // 5 % 2
    '''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 2
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index % 2
    '''

    # phase_flag为0代表A相,1代表C相
    if phase_flag == 0:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('a_DTW算法匹配结果：', equip_list_2[equip], state_list[state])
    if phase_flag == 1:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('c_DTW算法匹配结果：', equip_list_1[equip], state_list[state])

    # 返回设备编号和状态编号
    return equip, state

# 定义欧几里得函数
euclidean_distance = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
# 用欧几里得函数作为距离函数
def ED_DTW(power, templist, phase_flag):
    """
    使用DTW算法对功率数据进行分类。

    参数:
    power (list): 待分类的功率数据列表。
    templist (list of lists): 模板列表，包含多个功率数据模板。
    phase_flag(int): A相还是C相的标志位，0代表A相，1代表C相

    返回:
    tuple: 包含设备编号和状态编号的元组。
    """
    # 计算power数组前两个元素的平均值和后两个元素的平均值的最小值
    min_data = min(sum(power[:2]) / 2, sum(power[-2:]) / 2)
    # 将power数组中的每个元素减去min_data，进行数据标准化
    pending = np.array(power) - min_data
    # 初始化一个列表，用于存储每个模板与待分类数据的DTW距离
    dis_list = []
    # 遍历模板列表
    for j in range(len(templist)):
        temp = templist[j]
        # 计算模板temp和待分类数据pending的DTW距离
        # 使用manhattan_distance作为距离度量函数
        ds, a, b, c = dtw(temp, pending, dist=euclidean_distance)
        # DTW_result = dtw(temp, pending, dist_method=euclidean_distance)
        # ds = DTW_result.distance
        # 将计算得到的DTW距离添加到dis_list中
        dis_list.append(ds)

    # print("dis_list", dis_list)
    # 使用numpy的argsort函数找到dis_list中最小值的索引
    min_index = np.argsort(dis_list)[0]

    '''多时间序列计算设备编号'''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 10
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index // 5 % 2
    '''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 2
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index % 2
    '''
    # phase_flag为0代表A相,1代表C相
    if phase_flag == 0:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('a_ED_DTW算法匹配结果：', equip_list_2[equip], state_list[state])
    if phase_flag == 1:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('c_ED_DTW算法匹配结果：', equip_list_1[equip], state_list[state])

    # 返回设备编号和状态编号
    return equip, state

# 定义切比雪夫函数
chebyshev_distance = lambda x, y: np.max(np.abs(x - y))

# 用切比雪夫函数作为距离函数
def CD_DTW(power, templist, phase_flag):
    """
    使用DTW算法对功率数据进行分类。

    参数:
    power (list): 待分类的功率数据列表。
    templist (list of lists): 模板列表，包含多个功率数据模板。
    phase_flag(int): A相还是C相的标志位，0代表A相，1代表C相

    返回:
    tuple: 包含设备编号和状态编号的元组。
    """
    # 计算power数组前两个元素的平均值和后两个元素的平均值的最小值
    min_data = min(sum(power[:2]) / 2, sum(power[-2:]) / 2)
    # 将power数组中的每个元素减去min_data，进行数据标准化
    pending = np.array(power) - min_data
    # 初始化一个列表，用于存储每个模板与待分类数据的DTW距离
    dis_list = []
    # 遍历模板列表
    for j in range(len(templist)):
        temp = templist[j]
        # 计算模板temp和待分类数据pending的DTW距离
        # 使用manhattan_distance作为距离度量函数
        ds, a, b, c = dtw(temp, pending, dist=chebyshev_distance)
        # DTW_result = dtw(temp, pending, dist_method=chebyshev_distance)
        # ds = DTW_result.distance
        # 将计算得到的DTW距离添加到dis_list中
        dis_list.append(ds)

    # print("dis_list", dis_list)
    # 使用numpy的argsort函数找到dis_list中最小值的索引
    min_index = np.argsort(dis_list)[0]

    '''多时间序列计算设备编号'''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 10
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index // 5 % 2
    '''
    # 计算设备编号，假设每个设备有两个状态，因此每两个索引对应一个设备
    equip = min_index // 2
    # 计算状态编号，使用模运算获取索引的第二个数字
    state = min_index % 2
    '''
    # phase_flag为0代表A相,1代表C相
    if phase_flag == 0:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('a_CD_DTW算法匹配结果：', equip_list_2[equip], state_list[state])
    if phase_flag == 1:
        # 打印匹配结果，使用equip_list和state_list字典将编号转换为名称
        print('c_CD_DTW算法匹配结果：', equip_list_1[equip], state_list[state])

    # 返回设备编号和状态编号
    return equip, state

def Detection(q_col2de, e_det, q_res2base, minimum_power):
    '''
    进行事件检测

    :param power_data: 输入csv中读取到的所有功率数据
    :param minimum_power: 输入当前csv的功率数据中电器最小的功率值
    :return: 返回所有事件起始位置的标签、所有事件巡检窗口的积分绝对值、所有事件巡检窗口的积分绝对值的对应标签
    '''

    e_det.wait()  # 等待检测信号
    print('正在初始化。。。。。。')

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

    # 读取DTW模板数据
    templist = pd.read_csv('/home/pi/Desktop/DTW_Light.csv', index_col=0).values.tolist()

    # 发送事件计数
    id_counter = 1

    on_flag = 0
    on_count = 0
    all_carbon_emissions = 0


    while 1:  # 无限循环，持续检测
        if not q_col2de.empty():
            data = q_col2de.get()  # 从队列中获取数据

            rear = (rear + 1) % MAX_SIZE  # 当 rear 达到 MAX_SIZE 时，模运算会使其回到 0
            data_arr[rear] = data[2]  # 将 data 赋值给缓冲区的当前尾部位置
            power = data_arr[rear + 1:] + data_arr[:rear + 1]  # 构造功率数据数组
            # 进行中值滤波
            # power = medfilt(power, kernel_size=9)
            # print("缓存data:", data_arr)
            # print("缓存power:", power)

            try:
                if skip_time <= 0:
                    mean = sum(power[:mean_size]) / mean_size  # 计算均值
                    sample_data = power[mean_size:]  # 提取样本数据
                    #   进行滑动窗口对应的积分值的阈值的自适应
                    change_point_power = max(abs(max(sample_data) - mean), abs(min(sample_data) - mean))
                    if change_point_power < minimum_power * 0.7:
                        change_point_power = minimum_power
                    window_threshold = change_point_power * 10 * threshold_ratio  # 根据变点功率更新滑窗阈值，乘10为了控制变点位置靠近滑动窗口中间位置
                    points_value = sum(sample_data) - mean * len(sample_data)  # 计算数据窗口对应的积分值

                    # print("++++++++++++++++++++++++++++++++++++++++++++++")
                    # print("功率数据", power)
                    # print("变点功率", change_point_power)
                    # print("滑窗阈值", window_threshold)
                    # print("滑窗积分值", points_value)
                    # print("++++++++++++++++++++++++++++++++++++++++++++++")

                    '''事件判断'''
                    if abs(points_value) >= window_threshold:
                        slide_result = PointsSlidingWindow(sample_data, change_point_power)
                        incident_value_absolute = slide_result[0]  # 所有事件的巡检窗口积分绝对值
                        incident_value_index = slide_result[1]  # 巡检窗口积分绝对值对应的标签
                        points_sliding_misjudge_flag = slide_result[2]  # 滑窗误判标志
                        #   滑窗误判
                        if points_sliding_misjudge_flag == 1:
                            skip_time = 18
                            continue
                        current_event_start_index = EventLocate(csv_count, incident_value_index)  # 当前事件起始位置，可能有多个事件
                        event_count = event_count + len(current_event_start_index)


                        # 投票
                        vote_list = [[0, 0, 0, 0], [0, 0, 0, 0]]
                        equip = -1
                        state = -1
                        # 参数1代表测的是灯
                        ED_DTW_result = ED_DTW(sample_data, templist, 1)
                        CD_DTW_result = CD_DTW(sample_data, templist, 1)
                        DTW_result = DTW(sample_data, templist, 1)

                        vote_list[ED_DTW_result[1]][ED_DTW_result[0]] += 1
                        vote_list[CD_DTW_result[1]][CD_DTW_result[0]] += 1
                        vote_list[DTW_result[1]][DTW_result[0]] += 1


                        for i in range(2):
                            for j in range(len(vote_list[0])):
                                if vote_list[i][j] >= 2:
                                    equip = j
                                    state = i
                                    break
                            
                        if equip != -1:
                            vote_result = equip_list_1[equip] + state_list[state]
                            # resule_list = [equip_list_1[ED_DTW_result[0]]+state_list[ED_DTW_result[1]],equip_list_1[CD_DTW_result[0]]+state_list[CD_DTW_result[1]],equip_list_1[DTW_result[0]]+state_list[DTW_result[1]],vote_result]
                            # q_res2base.put(resule_list)
                            print('投票结果：', vote_result)

                            if equip == 0:
                                event_string = json_send_event(id_counter, state, 0)
                                id_counter += 1
                                # print("event_string", event_string)
                                ttl_chuankou(event_string)
                        else:
                            vote_result = '异常事件！'
                            print(vote_result)

                        skip_time = 18
                else:
                    skip_time -= 1
            except IndexError:
                pass
        time.sleep(0.1)  # 休眠0.1秒，减少CPU占用


def Detection2(q_col2de_2, e_det_2, q_res2base_2, minimum_power_2):
    '''
    进行事件检测

    :param power_data: 输入csv中读取到的所有功率数据
    :param minimum_power_2: 输入当前csv的功率数据中电器最小的功率值
    :return: 返回所有事件起始位置的标签、所有事件巡检窗口的积分绝对值、所有事件巡检窗口的积分绝对值的对应标签
    '''

    e_det_2.wait()  # 等待检测信号
    print('正在初始化。。。。。。')

    '''数据初始化'''
    mean_size = 3  # 均值窗口大小
    data_size = 20  # 滑动窗口数据大小
    window_size = data_size + mean_size  # 滑动窗口大小
    change_point_power = 0  # 变点的功率情况
    csv_count = 0  # 数据在csv中的index
    rear = -1  # 初始化一个变量 rear，通常用于标记数组或列表的尾部位置，初始值为-1
    MAX_SIZE = window_size  # 定义一个变量 MAX_SIZE，其值等于变量 window_size，通常用于定义数组或列表的最大容量
    data_arr = [0] * MAX_SIZE  # 创建一个大小为 MAX_SIZE 的列表，初始值全部为0
    threshold_ratio = 0.8  # 阈值比例，用于降低事件判断标准
    skip_time = 23  # 用于跳过检测到事件后的读取，初始为23是为了跳过初始读数据的误判，power初始全0，而读取数据时如果稳定功率不为0且大于阈值会被误判
    event_count = 0  # 用于事件计数

    all_events_index = []  # 储存所有事件起始位置的标签
    all_events_incident_value_absolute = []  # 储存储存所有事件巡检窗口的积分绝对值
    all_events_incident_value_index = []  # 储存储存所有事件巡检窗口的积分绝对值的对应标签

    # 读取DTW模板数据
    templist = pd.read_csv('/home/pi/Desktop/DTW_Lidianchi.csv', index_col=0).values.tolist()

    # 发送事件计数
    id_counter = 1

    on_flag = 0
    on_count = 0
    all_carbon_emissions = 0

    reshuihu_power = 1400
    lidianchi_power = 265

    while 1:  # 无限循环，持续检测
        if not q_col2de_2.empty():
            data = q_col2de_2.get()  # 从队列中获取数据
            rear = (rear + 1) % MAX_SIZE  # 当 rear 达到 MAX_SIZE 时，模运算会使其回到 0
            data_arr[rear] = data[0]  # 将 data 赋值给缓冲区的当前尾部位置
            power = data_arr[rear + 1:] + data_arr[:rear + 1]  # 构造功率数据数组
            # 进行中值滤波
            # power = medfilt(power, kernel_size=9)
            # print("缓存data:", data_arr)
            # print("缓存power:", power)

            try:
                if skip_time <= 0:
                    mean = sum(power[:mean_size]) / mean_size  # 计算均值
                    sample_data = power[mean_size:]  # 提取样本数据
                    #   进行滑动窗口对应的积分值的阈值的自适应
                    change_point_power = max(abs(max(sample_data) - mean), abs(min(sample_data) - mean))
                    if change_point_power < minimum_power_2 * 0.7:
                        change_point_power = minimum_power_2
                    window_threshold = change_point_power * 10 * threshold_ratio  # 根据变点功率更新滑窗阈值，乘10为了控制变点位置靠近滑动窗口中间位置
                    points_value = sum(sample_data) - mean * len(sample_data)  # 计算数据窗口对应的积分值

                    # print("++++++++++++++++++++++++++++++++++++++++++++++")
                    # print("功率数据", power)
                    # print("变点功率", change_point_power)
                    # print("滑窗阈值", window_threshold)
                    # print("滑窗积分值", points_value)
                    # print("++++++++++++++++++++++++++++++++++++++++++++++")

                    '''事件判断'''
                    if abs(points_value) >= window_threshold:
                        slide_result = PointsSlidingWindow(sample_data, change_point_power)
                        incident_value_absolute = slide_result[0]  # 所有事件的巡检窗口积分绝对值
                        incident_value_index = slide_result[1]  # 巡检窗口积分绝对值对应的标签
                        points_sliding_misjudge_flag = slide_result[2]  # 滑窗误判标志
                        #   滑窗误判
                        if points_sliding_misjudge_flag == 1:
                            skip_time = 18
                            continue
                        current_event_start_index = EventLocate(csv_count, incident_value_index)  # 当前事件起始位置，可能有多个事件
                        event_count = event_count + len(current_event_start_index)

                        # 投票
                        vote_list = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
                        equip = -1
                        state = -1
                        # 参数1代表测的是灯
                        ED_DTW_result = ED_DTW(sample_data, templist, 0)
                        CD_DTW_result = CD_DTW(sample_data, templist, 0)
                        DTW_result = DTW(sample_data, templist, 0)

                        vote_list[ED_DTW_result[1]][ED_DTW_result[0]] += 1
                        vote_list[CD_DTW_result[1]][CD_DTW_result[0]] += 1
                        vote_list[DTW_result[1]][DTW_result[0]] += 1

                        for i in range(2):
                            for j in range(len(vote_list[0])):
                                if vote_list[i][j] >= 2:
                                    equip = j
                                    state = i
                                    break

                        if equip != -1:
                            vote_result = equip_list_2[equip] + state_list[state]
                            # resule_list = [equip_list_1[ED_DTW_result[0]]+state_list[ED_DTW_result[1]],equip_list_1[CD_DTW_result[0]]+state_list[CD_DTW_result[1]],equip_list_1[DTW_result[0]]+state_list[DTW_result[1]],vote_result]
                            # q_res2base_2.put(resule_list)
                            print('投票结果：', vote_result)

                            if equip == 1:
                                event_string = json_send_event(id_counter, state, 1)
                                id_counter += 1
                                # print("event_string", event_string)
                                ttl_chuankou(event_string)

                            if equip == 2:
                                if state == 1:
                                    event_string = json_send_event(id_counter, state, 2)
                                    id_counter += 1
                                    # print("event_string", event_string)
                                    ttl_chuankou(event_string)
                                    on_count += 1
                                    if on_count > 1:
                                        on_count = 1
                                    start_time = time.time()
                                if state == 0 and on_count == 1:
                                    end_time = time.time()
                                    running_time = (end_time - start_time) / 3600
                                    on_count = 0
                                    G = running_time * reshuihu_power / 1000 * 0.5422
                                    all_carbon_emissions += G

                                    # 向TTL发送碳排数据
                                    event_string = json_send_event(id_counter, state, 2, G, all_carbon_emissions)
                                    id_counter += 1
                                    # print("event_string", event_string)
                                    ttl_chuankou(event_string)
                        else:
                            vote_result = '异常事件！'
                            print(vote_result)

                        skip_time = 18
                else:
                    skip_time -= 1
            except IndexError:
                pass
        time.sleep(0.1)  # 休眠0.1秒，减少CPU占用




def Input(q_ser, q_flag, e_base, e_det, e_det_2):
    """
    串口数据接收函数，同时提供图形界面显示实时数据。

    参数:
    q_ser (queue): 串口对象队列。
    q_flag (queue): 控制标志队列。
    e_base (event): 基础事件对象。
    e_det (event): 检测事件对象。
    """
    # 串口对象
    print("连接数据库")
    e_base.set()
    time.sleep(0.5)
    print("重建数据库")
    q_flag.put(1)


    ser = serial.Serial()  # 创建串口对象
    ser.baudrate = 19200  # 设置波特率
    ser.parity = 'E'
    port_list = [i[0] for i in list(serial.tools.list_ports.comports())]  # 获取串口设备列表
    print('当前检测到的端口：')
    print(port_list)
    # 默认选择第一个可用的串口
    if port_list:
        ser.port = port_list[0]
    else:
        print("没有检测到可用的串口")

    ser.timeout = 1  # 设置串口超时时间
    ser.open()  # 打开串口
    print('端口打开', ser.is_open)  # 检验串口是否成功打开
    q_ser.put(ser)  # 将串口对象放入队列
    e_det.set()  # 设置检测事件
    e_det_2.set()


# 如果是主程序，执行以下代码
if __name__ == "__main__":

    freeze_support()

    q_col2base_carbon_emission = Queue() # 碳排放数据队列

    # 创建多个队列，用于线程间通信
    q_col2de = Queue()  # 功率数据处理队列
    q_ser = Queue()  # 串口对象队列
    q_flag = Queue()  # 控制标志队列
    q_col2base = Queue()  # 功率数据存储队列
    q_res2base = Queue()  # 结果数据存储队列

    # 创建事件对象，用于线程间的同步
    e_base = Event()  # 基础事件对象
    e_det = Event()  # 检测事件对象

    # 定时计数
    count_for_minute = 0
    count_for_minute_flag = 0
    count_lock = threading.Lock()

    q_col2de_2 = Queue()  # 功率数据处理队列
    e_det_2 = Event()  # 检测事件对象
    q_res2base_2 = Queue() # 结果数据存储队列

    minimum_power = 260
    minimum_power_2 = 270

    # 创建多个线程，分别执行不同的任务
    th1 = Thread(target=ColData, args=(q_col2de, q_col2de_2, q_ser, q_col2base, q_col2base_carbon_emission))  # 数据收集线程
    th2 = Thread(target=Detection, args=(q_col2de, e_det, q_res2base, minimum_power))  # 检测线程
    # th3 = Thread(target=Put2base, args=(q_col2base, q_flag, e_base, q_res2base, q_res2base_2, q_col2base_carbon_emission))  # 数据存储线程
    th4 = Thread(target=check_time_and_send_signal)  # 定时发送线程
    th5 = Thread(target=Detection2, args=(q_col2de_2, e_det_2, q_res2base_2, minimum_power_2))  # 检测线程2

    # 设置线程为守护线程，这样当主程序结束时，线程也会随之结束
    th1.daemon = True
    th2.daemon = True
    # th3.daemon = True
    th4.daemon = True
    th5.daemon = True


    # 启动线程
    th1.start()
    th2.start()
    # th3.start()
    th4.start()
    th5.start()

    # 调用Input函数，开始接收串口数据并进行实时显示
    Input(q_ser, q_flag, e_base, e_det, e_det_2)

    while True:
        time.sleep(0.01)
