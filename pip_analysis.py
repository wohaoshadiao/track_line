# -*- coding: utf-8 -*-
# @Time : 2021/7/22 1:59
# @Author : blue-eyes
# @Computer : HUAWEI MATEBOOK14
# @FileName: hsv_get.py
# @Software: PyCharm

import cv2
import numpy as np
import time
import math

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              'max_black': {'Lower': np.array([0, 20, 0]), 'Upper': np.array([180, 255, 110])},
              'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 45])},
              'black_max': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 130])},
              'orange': {'Lower': np.array([11, 43, 46]), 'Upper': np.array([25, 255, 255])},
              'yellow': {'Lower': np.array([18, 43, 46]), 'Upper': np.array([40, 255, 255])},
              'all': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([255, 255, 255])},
              'field_zcx1': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([122, 127, 188])},
              'field_zn1': {'Lower': np.array([97, 2, 119]), 'Upper': np.array([158, 60, 223])},
              'field_zn2': {'Lower': np.array([115, 12, 140]), 'Upper': np.array([152, 65, 245])},
              }


class CamPic:
    def __init__(self, need_analysis="shape", pic_shape=None, thresholding=None, need_return=None):
        if need_return is None:
            need_return = ["center"]
        if pic_shape is None:
            pic_shape = [640, 480]
        if thresholding is None:
            thresholding = ["hsv"]
        self.thresholding = thresholding
        self.pic_shape = pic_shape
        self.need_return = need_return
        self.need_analysis = need_analysis

    def analysis_pic(self, in_image, input_parameter=None):
        if input_parameter is None:
            input_parameter = [["red"], [12]]
        if self.thresholding == "hsv":
            threshold_pic = pic_hsv_color(in_image, input_parameter[0][0])
        elif self.thresholding == "threshold":
            threshold_pic = pic_threshold_color(in_image)

        # if self.need_analysis == "shape":
        #     cx, cy = find_shape()


def find_the_point(img):
    return [0, 0]


def yolo_false():
    return False


def pic_hsv_color(in_image, color, kernel=np.ones([3, 3], np.uint8), white_large=None):
    """
    :param kernel:
    :param white_large: about the hope is bigger or not
    :param in_image: BGR
    :param color: the_need color
    :return:
    """
    # 色彩空间转换
    hsv = cv2.cvtColor(in_image, cv2.COLOR_BGR2HSV)
    # 设定阈值提取指定色彩
    inRange_hsv = cv2.inRange(hsv, color_dist[color]['Lower'],
                              color_dist[color]['Upper'])
    inRange_hsv = change_white(inRange_hsv, kernel, white_large)
    return inRange_hsv


def pic_threshold_color(in_image, min_gray=10, max_gray=120, kernel=np.ones([3, 3], np.uint8), white_large=None):
    """
    OpenCV中的mask掩膜原理：
    掩模一般是小于等于源图像的单通道矩阵，掩模中的值分为两种0和非0。
    当mask掩膜中的值不为0，则将源图像拷贝到目标图像，当mask中的值为0，则不进行拷贝，目标图像保持不变。
    以 dst=cv2.bitwise_and(src1, src2, mask) 为例，先进行src1和src2的 "与" 运算，所得结果再与mask进行掩膜运算(mask为非0的则拷贝到dst中)。

    :param min_gray: 最小数值
    :param max_gray: 最大数值
    :param kernel: 默认就可以了
    :return:  thresh
    """
    # 255-> 白色
    # 0 -> 黑色
    # GRAY是8位灰度图
    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    _, thresh_1 = cv2.threshold(gray, max_gray, 255, cv2.THRESH_BINARY_INV)
    # 这个参数涉及到去掉影子，数字越小影子干掉的也就越多
    # 数字越小 -> 白色越少
    _, thresh_2 = cv2.threshold(gray, min_gray, 255, cv2.THRESH_BINARY)
    # 这个参数涉及到去掉曝光，数字越大干掉的曝光就越多
    # 数字越大 -> 白色越多
    # 与操作，白色区域是保留，黑色区域是剔除
    thresh = cv2.bitwise_and(thresh_1, thresh_2, init_canvas(640, 480, (0, 0, 0)))
    # 对不同二值化方式图像进行与操作
    thresh = form_operation(thresh, kernel, white_large)
    return thresh


def form_operation(in_image, kernel=np.ones([3, 3], np.uint8), white_large=None):
    """
    形态学运算 cv.erode(),cv.dilate(), cv.morphologyEx()
    :param white_large:腐蚀次数
    :param kernel:形态学滤波器(3×3核)
    开运算 = (erode->dilate)消除小黑点 闭运算(dilate->erode) 消除小黑洞
    """
    # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # erosion = cv.erode(img,kernel,iterations = 1)  -->  dilation = cv.dilate(img,kernel,iterations = 1) == opening

    if white_large is None:
        print("___not dilate___")
        return in_image
    # iterations是腐蚀的次数,一般为1
    for large_white_size in white_large:
        if large_white_size > 0:
            in_image = cv2.dilate(in_image, kernel, iterations=large_white_size)
        elif large_white_size < 0:
            in_image = cv2.erode(in_image, kernel, iterations=abs(large_white_size))
    return in_image

#
# def D435_dis_inrange(img, dis_min, dis_max, input_depth_scale):
#     """
#     :param img: 需要输入的图片，应该为灰度的深度图片
#     :param dis_min: 最小的距离
#     :param dis_max: 最大的距离
#     :param input_depth_scale: 这个是D435的一个相机的参数
#     :return: 白色为在范围之内的，黑色为不在范围之内的
#     """
#     pic_dis_min = dis_min / input_depth_scale
#     pic_dis_max = dis_max / input_depth_scale
#
#     mask = cv2.inRange(img, pic_dis_min, pic_dis_max)
#
#     return mask


def D435_color_dis(pipeline, align, depth_scale, color, dis_min, dis_max):
    global count
    """
    :param pipeline:  D435 的 pipeline
    :param align: D435 的一个参数，可以直接在上面获取
    :param depth_scale: D435 的关于最后计算距离的一个参数
    :param color:  所想测量的颜色的距离
    :param dis_min: 距离阈值的下线
    :param dis_max:  距离阈值的上限
    :return:  center_x -> 图片中X的位置（0 - 100）
              center_y  -> 图片中的位置（0 - 100）
              mean_dis -> 要找的位置的平均值
    """
    try:
        center_x, center_y, mean_dis = 0, 0, 0
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if aligned_depth_frame and color_frame:
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # cv2.rectangle(depth_image, (0, 0), (250, 350), (0, 0, 255), 2)
            # cv2.rectangle(color_image, (0, 0), (250, 350), (0, 0, 255), 2)
            mask_image = D435_dis_inrange(depth_image, dis_min, dis_max, depth_scale)
            color_mask = pic_hsv_color(color_image, color)
            imgResult = cv2.bitwise_and(color_mask, color_mask, mask=mask_image)

            color_need = depth_image[imgResult != 0]
            if len(color_need):
                center_x, center_y = thresh_center(imgResult)
                mean_dis = np.mean(color_need) * depth_scale
                # print(mean_dis)

            # cv2.imshow("mask", mask_image)
            # # cv2.imshow("depth", depth_image)
            #
            # cv2.putText(color_image, str(mean_dis*100), (0, 110), cv2.FONT_HERSHEY_COMPLEX, 5, (250, 0, 0), 5)
            cv2.imshow("color", color_image)
            cv2.imshow("color_mask", color_mask)
            cv2.imwrite("img/" + str(time.time())[0:-5] + ".jpg", color_image)
            # cv2.imshow("result", imgResult)
            cv2.waitKey(10)

        return center_x, center_y, mean_dis
    except:
        print("D435 error")


def init_canvas(width, height, color=(255, 255, 255)):
    """
    这个函数只有在创建通道的时候是需要使用的
    正常的时候是用不到的
    :param width: 图片宽度
    :param height: 图片高度
    :param color: 图片所需要的颜色的通道颜色
    :return:
    """
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


def thresh_center(in_image, the_max=0.1, the_width=640, the_height=480):
    """

    :param in_image: 输入的必须为二值格式
    :param the_max:
    :param the_width:
    :param the_height:
    :return: 返回的是相对位置 -> ([0-100], [0-100])
    """
    # print(np.sum(in_image) / 255 / 64 / 48)
    # 这个数值其实就是一个关于像素点的数值 -> 判断里面白色的数值是多少，如果过于稀少则不进行判断
    if np.sum(in_image) / 2.55 / the_width / the_height > the_max:
        M =cv2.moments(in_image) # 以字典形式返回图像的矩
        if M["m00"] != 0:
            cx = int((M["m10"] / M["m00"]))
            cy = int((M["m01"] / M["m00"]))
            return int(cx * 100 / the_width), int(cy * 100 / the_height)
    return 0, 0


def find_shape(in_image, shape=4, perimeter_limit=None, ar_limit=None, is_draw=0):
    """
    工训H的参数(外围正方形) -> 4 [300,3000] [0.95, 1.05]
    工训H的参数(内部的H的参数) -> 12 [300, 1000] []
    :param is_draw:
    :param in_image: 必须为二值化的图片
    :param shape:
    :param perimeter_limit:周长限制
    :param ar_limit:
    :return:
    """
    if ar_limit is None:
        ar_limit = [0, 100000]
    if perimeter_limit is None:
        perimeter_limit = [0, 100000]
    cx = 0
    cy = 0
    # 源图像,轮廓检索模式,轮廓逼近方法
    contours, hierarchy = cv2.findContours(in_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # cv2.drawContours(img_show, contours, i, colormap[hierarchy[0, i, 3] + 1], 5)
        # 计算弧长(第二个参数用来指定图像是否闭合(True))
        perimeter = cv2.arcLength(contour, True)
        if perimeter_limit[0] < perimeter < perimeter_limit[1]:
            M = cv2.moments(contour)
            # 多边形拟合曲线（第二个参数表明轮廓到近似轮廓的精度,通常由0.1,0.01）
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == shape:
                if M["m00"] != 0:
                    # (x, y, w, h) = cv2.boundingRect(approx)
                    # ar = w / float(h)
                    # if ar_limit[0] <= ar <= ar_limit[1]:
                    cx = int((M["m10"] / M["m00"]))
                    cy = int((M["m01"] / M["m00"]))
                    if is_draw:
                        cv2.drawContours(in_image, contour, -1, (255, 0, 0), 5)
                        cv2.circle(in_image, (cx, cy), 2, (255, 255, 0), 2)
                    # break
    # return int(cx * 100 / 640), int(cy * 100 / 480)
    return int(cx), int(cy)


def find_shape_test(img_draw, in_image, shape=4, perimeter_limit=None, ar_limit=None, is_draw=0):
    """
    工训H的参数(外围正方形) -> 4 [300,3000] [0.95, 1.05]
    工训H的参数(内部的H的参数) -> 12 [300, 1000] []
    :param is_draw:
    :param in_image: 必须为二值化的图片
    :param shape:
    :param perimeter_limit:
    :param ar_limit:
    :return:
    """
    if ar_limit is None:
        ar_limit = [0, 100000]
    if perimeter_limit is None:
        perimeter_limit = [0, 100000]
    cx = 0
    cy = 0

    contours, hierarchy = cv2.findContours(in_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = cv2.drawContours(img_draw, contours, -1, (0, 255, 0), 5)
    for contour in contours:
        # cv2.drawContours(img_show, contours, i, colormap[hierarchy[0, i, 3] + 1], 5)
        perimeter = cv2.arcLength(contour, True)
        if perimeter_limit[0] < perimeter < perimeter_limit[1]:
            M = cv2.moments(contour)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == shape:
                if M["m00"] != 0:
                    # (x, y, w, h) = cv2.boundingRect(approx)
                    # ar = w / float(h)
                    # if ar_limit[0] <= ar <= ar_limit[1]:
                    cx = int((M["m10"] / M["m00"]))
                    cy = int((M["m01"] / M["m00"]))
                    if is_draw:
                        cv2.drawContours(in_image, contour, -1, (255, 0, 0), 5)
                        # 用圆来拟合
                        cv2.circle(in_image, (cx, cy), 2, (255, 255, 0), 2)
                    # break
    # return int(cx * 100 / 640), int(cy * 100 / 480)
    return int(cx), int(cy)


def find_circle(in_image, max_translation=16, white_large=0, if_show=0):
    """
    HSV the find will be the white
    gray the find will be the black
    use the perimeter and area to judge a circle
    :param if_show: if show the img
    :param white_large: it can large the max and min
    :param max_translation:the circle max, if it large,the find will be more
    :param in_image: the_input_img -> TH
    :return: the center of circle(width), the center of circle(height)
    """

    out_x = []
    out_y = []

    gray_img = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    # II参数表明轮廓输出形式(tree,balcane) III表明轮廓逼近的方式
    # 返回轮廓的列表信息,每一元素代表一个边沿信息
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    run_time = 0
    for contour in contours:
        # arclength表示轮廓的周长,contour表示轮廓,True表示是否闭合q
        perimeter = cv2.arcLength(contour, True)
        # 多边形逼近函数:I为输入的点集,II为原始与拟合之间的精度,III闭合与否
        if perimeter > 200:
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            print(approx)
            if len(approx) != 4:
                # 求图像的矩,可拓展成不变矩
                M = cv2.moments(contour)
                run_time = run_time + 1
                # cv2.putText(in_image, "1", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                if M["m00"] != 0:
                    # cv2.putText(in_image, "2", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    # cv2.putText(in_image, str(perimeter * perimeter / M["m00"]), (110, run_time*50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    #             (0, 0, 250), 2)
                    if 10 < perimeter * perimeter / M["m00"] < max_translation:
                        # cv2.putText(in_image, "3", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        cx = int((M["m10"] / M["m00"]))
                        cy = int((M["m01"] / M["m00"]))
                        out_x.append(cx)
                        out_y.append(cy)
                        # print(contour[0][0])
                        cv2.drawContours(in_image, [contour], 0, (0, 0, 250), 2)
                        cv2.imshow('contours', in_image)
                        # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    if len(out_x) == 0:
        out_x.append(0)
    if len(out_y) == 0:
        out_y.append(0)
    # return int(np.mean(out_x) * 100 / 640), int(np.mean(out_y) * 100 / 480)
    # 求均值
    return int(np.mean(out_x)), int(np.mean(out_y))


def find_hough_circle(in_image):
    """
    param1参数表示Canny边缘检测的高阈值，低阈值会被自动置为高阈值的一半。
    param2参数表示圆心检测的累加阈值，参数值越小，可以检测越多的假圆圈，但返回的是与较大累加器值对应的圆圈。
    minRadius参数表示检测到的圆的最小半径。
    maxRadius参数表示检测到的圆的最大半径。
    """

    # 工训的参数
    circle_parameter = [120, 20, 50, 48, 95, 180]
    all_gray_low = circle_parameter[0]
    all_miniDist = circle_parameter[1]
    all_param1 = circle_parameter[2]
    all_param2 = circle_parameter[3]
    all_minRadius = circle_parameter[4]
    all_maxRadius = circle_parameter[5]

    circles = cv2.HoughCircles(in_image,cv2.HOUGH_GRADIENT_ALT, 1.5, all_miniDist,
                               param1=all_param1, param2=all_param2, minRadius=all_minRadius, maxRadius=all_maxRadius)
    circles_core_x = []
    circles_core_y = []
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
        for i in circles[0, :]:
            cv2.circle(in_image, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv2.circle(in_image, (i[0], i[1]), 2, (255, 0, 0), 2)
            circles_core_x.append(i[0])
            circles_core_y.append(i[1])
    if len(circles_core_x) == 0:
        circles_core_x.append(0)
    if len(circles_core_y) == 0:
        circles_core_y.append(0)

    # print(int(np.mean(circles_core_x) * 100 / img_width), int(np.mean(circles_core_y) * 100 / img_height))
    return int(np.mean(circles_core_x)), int(np.mean(circles_core_y))


def find_circle_all(in_image):
    cx, cy = find_circle(in_image)
    if cx == 0 and cy == 0:
        cx, cy = find_hough_circle(in_image)
    else:
        cv2.circle(in_image, (cx, cy), 2, (0, 0, 255), 1)
        return cx, cy

    if cx != 0 and cy != 0:
        cv2.circle(in_image, (cx, cy), 2, (0, 0, 255), 1)
        return cx, cy

    print("no find circle")
    return 0, 0


def find_chessboard(img, size):
    # 返回输出检测到的角点数组
    ret, corners = cv2.findChessboardCorners(img, size, None)
    print(corners)
    print(corners.shape)
    if ret:
        for pt in corners:
            point = pt[0]
            # print(point)
            cv2.circle(img, center=(int(point[0]), int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('cannot find chessboard points')


def Corner_detection(img):
    """
     • img - 数据类型为 float32 的输入图像
　　 • blockSize - 角点检测中要考虑的领域大小
　　 • ksize - Sobel 求导中使用的窗口大小
　 　• k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # 提升角点的清晰度
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    print(dst)


def find_hough_line(thresh, is_all_return=0):
    """

    cv2.HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 )
    dst:    输出图像. 它应该是个灰度图 (但事实上是个二值化图)
    lines:  储存着检测到的直线的参数对 (x_{start}, y_{start}, x_{end}, y_{end}) 的容器
    rho :   参数极径 r 以像素值为单位的分辨率. 我们使用 1 像素.
    theta:  参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
    threshold:    设置阈值： 一条直线所需最少的的曲线交点。超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
    minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
    maxLineGap:   能被认为在一条直线上的两点的最大距离。
    return:返回的是含有一条直线的起始点和终点坐标[x1,y1,x2,y2]

    x1，x2 默认的返回是在图像的左侧的值作为x1来返回。
    """
    # gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # thresh = colorThresh(in_image, 140, -5, 5)
    # cv2.imshow("gray", thresh)
    thresh = cv2.Canny(thresh, 50, 120, apertureSize=5)         # -去噪 - 梯度计算 - 非极大值抑制 -迟滞阈值法  :param 计算得到的边缘图像
    # cv2.imshow("gray", thresh)
    # 概率霍夫变换返回线的两个端点
    lines = cv2.HoughLinesP(thresh, 3, np.pi / 180, 100, minLineLength=40, maxLineGap=50)

    # 仅作为备份图像使用,以及下面画直线时,避免直接在原图上操作
    # result = in_image.copy()

    list_lines_angle = []
    list_lines_pos = []

    list_all_line = []

    if lines is not None:
        # 使用索引来截取字符
        line1 = lines[:, 0, :]
        for x1, y1, x2, y2 in line1[::]:
            if (y1 - y2) == 0:
                # np.pi使用常数来表示圆周率
                h = np.pi / 2
            else:
                # h = math.atan((y1 - y2) / (x1 - x2))
                # 返回斜率对应的弧度值
                h = math.atan((x1 - x2) / (y1 - y2))
                # if -45 < h < 45:
                # cv2.line(in_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 关于直线的标注，以后需要看到直线可以直接写
                # cv2.circle(in_image, (x1, y1), 2, (255, 0, 0), 2)
                if is_all_return == 0:
                    # 将弧度制转化为角度制  rad/pi == angle/180
                    list_lines_angle.append(int(math.degrees(h)) * -1)
                    list_lines_pos.append([(x1 + x2) / 2, (y1 + y2) / 2])
                else:
                    list_all_line.append([[x1, y1], [x2, y2], int(math.degrees(h)) * -1])
    # need_lines = []
    # if list_lines:
    #     for i in list_lines:
    #         if -45 < i < 45:
    #             need_lines.append(int(i))
    # print(need_lines)
    # the_return = 100
    # if list_lines:
    #     the_return = int(np.average(list_lines) * -1)
    # print(the_return)
    if is_all_return == 0:
        the_angle = 100
        the_pos_x = 100
        the_pos_y = 100
        # print(list_lines_pos)
        # print(list_lines_angle)
        if list_lines_angle:
            # 计算平均值
            the_angle = int(np.average(list_lines_angle))
            the_pos_x, the_pos_y = np.average(list_lines_pos, axis=0)
        # 竖像素为y轴，对速度范围的限制
        return the_angle, int(the_pos_y * 100 / 1280), int(the_pos_x * 100 / 720)
    else:
        return list_all_line


def find_A(img):
    threshold_pic = pic_threshold_color(img)
    contours, hierarchy = cv2.findContours(threshold_pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if 300 < perimeter < 700:
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == 8:
                M = cv2.moments(contour)
                cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
                if M["m00"] != 0:
                    cx = int((M["m10"] / M["m00"]))
                    cy = int((M["m01"] / M["m00"]))
                    # print("A", perimeter, cx, cy)
                    return cx, cy
        # if 100 < perimeter < 200:
        #     approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        #     cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
        #     print(perimeter)
        #     if len(approx) == 3:
        #         M = cv2.moments(contour)
        #         print(perimeter)
        #
        #         if M["m00"] != 0:
        #             cx = int((M["m10"] / M["m00"]))
        #             cy = int((M["m01"] / M["m00"]))
        # return cx, cy
    return 0, 0


def find_back(img):
    threshold_pic = pic_threshold_color(img, white_large=[5, -5, 5])
    # cv2.imshow("thresh", threshold_pic)
    contours, hierarchy = cv2.findContours(threshold_pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cx = 0
    cy = 0
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # print(type(contours))
    draw_list = []
    # print(len(contours))
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # cv2.drawContours(img, contours, -1, (0, 255, 255), 3)

        if 800 < perimeter < 1200:
            cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            print(len(approx))
            if len(approx) == 16:
                cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int((M["m10"] / M["m00"]))
                    cy = int((M["m01"] / M["m00"]))
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), 5)
                    print("find ten")
                    return [cx, cy]

        elif 1100 < perimeter < 1600:
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            cv2.drawContours(img, contour, -1, (0, 0, 255), 3)
            if len(approx) > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int((M["m10"] / M["m00"]))
                    cy = int((M["m01"] / M["m00"]))
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), 5)
                    print("find circle")
                    return [cx, cy]
    return cx, cy


def find_circle_back(frame):
    find_hough_circle(frame)


def is_need_light(frame):
    # img_cut = frame[160:560, 440:840]
    img_cut = frame[355:555, 491:691]

    thresh = pic_hsv_color(img_cut, "field_zcx1")
    # cv2.imshow("cut", thresh)
    # cv2.waitKey(5)
    pic_avg = np.average(thresh)
    print(pic_avg)

    if pic_avg > 100:
        return True
    return False


# 关于摄像头的
# if __name__ == '__main__':
#     frameWidth = 640
#     frameHeight = 480
#     cap = cv2.VideoCapture(1)
#     # cap.set(3, frameWidth)
#     # cap.set(4, frameHeight)
#     # cap.set(10, 150)
#     while True:
#         success, img = cap.read()
#         find_chessboard_h(img)
#         cv2.imshow("Result", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# if __name__ == '__main__':
#     img_show_f = cv2.imread("img_1.png")
#
#     pos_A_x, pos_A_y = find_A(img_show_f)
#
#     if pos_A_x == 0 and pos_A_y == 0:
#         print("no find A")
#     else:
#         cv2.circle(img_show_f, (pos_A_x, pos_A_y), 5, (255, 255, 0), 5)
#
#     cv2.imshow("img ", img_show_f)
#     cv2.waitKey(0)


# if __name__ == '__main__':
#
#
#     img_show_f = cv2.imread("f16.png")
#
#     gray = pic_threshold_color(img_show_f, white_large=[5])
#
#     print(find_hough_circle(gray))
#
#     cv2.imshow("img ", gray)
#     cv2.waitKey(0)




def false_hough_circle(img):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=100, param2=30, minRadius=100, maxRadius=200)

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
    # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow("deceted_cirlce", cimg)
