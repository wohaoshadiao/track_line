# -*- coding: utf-8 -*-
# @Time : 2021/7/25 12:18
# @Author : blue-eyes
# @Computer : HUAWEI MATEBOOK14
# @FileName: serial_lib.py
# @Software: PyCharm
from time import sleep
import serial


def send_order(ser_input, the_order, number, number1, number2):
    ser_input.write(the_order.encode("gbk"))
    ser_input.write([int(number)])
    ser_input.write([int(number1)])
    ser_input.write([int(number2)])
    print(the_order, int(number), int(number1), int(number2))


def send_t265_order(ser_put, y, x, yaw, is_print=1):
    # ser_put.write([1])
    # ser_put.write([int(y)])
    # # print(int(y))
    # ser_put.write([int(x)])
    # ser_put.write([int(yaw / 2)])
    # ser_put.write([85])
    # if is_print:
    #     print("T265", int(y), int(x), int(yaw / 2))
    ser_put.write([1])
    ser_put.write([2])
    ser_put.write([int(y)])
    ser_put.write([int(x)])
    ser_put.write([int(yaw / 2)])
    ser_put.write([3])


def send_con_order(ser_put, the_order, number, number1, number2):
    ser_put.write([239])
    ser_put.write(the_order.encode("gbk"))
    ser_put.write([int(number)])
    ser_put.write([int(number1)])
    ser_put.write([int(number2)])
    ser_put.write([55])
    print("order", the_order, number, number1, number2)


def send_order_all(ser_put, y, x, yaw, the_order, number, number1, number2, is_print=0):
    # ser_put.write([254])
    # ser_put.write([int(y)])
    # ser_put.write([int(x)])
    # ser_put.write([int(yaw / 2)])
    # ser_put.write(the_order.encode("gbk"))
    # ser_put.write([int(number)])
    # ser_put.write([int(number1)])
    # # ser_put.write([int(number2)])
    # ser_put.write([85])
    ser_put.write([1])
    ser_put.write([2])
    ser_put.write([int(y)])
    ser_put.write([int(x)])
    ser_put.write([int(yaw / 2)])
    ser_put.write(the_order.encode("gbk"))
    ser_put.write([int(number)])
    ser_put.write([int(number1)])
    ser_put.write([int(number2)])
    ser_put.write([3])



    # ser_put.write([int(y)])
    # ser_put.write([int(x)])
    # ser_put.write([int(y)])
    # ser_put.write([int(x)])
    if is_print:
        print("T265", int(y), int(x), int(yaw / 2), the_order, number, number1, number2)


if __name__ == '__main__':
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    while 1:
        ser.write("M".encode("gbk"))
