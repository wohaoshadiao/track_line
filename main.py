import pip_analysis as pro
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        # if not cap.isOpened():
        #     print('___open pic failed___')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_thr = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        # 自适应阈值(better)
        thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        # frame_pro = CamPic()
        # list_all_line = pro.find_hough_line(thresh_img, is_all_return=0)
        list = pro.find_hough_line(thresh_img)
        # try:
        cv2.imshow('videos', frame)
        cv2.imshow('thresh', thresh_img)
        print(list)
        # che = pro.find_chessboard(thresh_img, size, None)
            # thresh1,2值越小,边缘信息越多
        # thresh = cv2.Canny(thresh_img, 50, 120, apertureSize=5)
        # cv2.imshow('canny_img', thresh)
        # print(list_all_line)
        # 使用waitKey(0)则只会显示第一帧视频,表示需要按下任意键继续
        if cv2.waitKey(1) == ord('q'):
            break

        # except KeyboardInterrupt:
        #     print('___Interupt stopped')
