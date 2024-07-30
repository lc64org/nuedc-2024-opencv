import cv2 #opencv读取的格式是BGR
import numpy as np
import cv2
import datetime


white_D= np.array([15, 15, 200])
white_U= np.array([180, 55, 255])
black_D= np.array([0, 0, 0])
black_U= np.array([180, 255, 100])
green_D= np.array([40, 70, 70])
green_U= np.array([80, 255, 255])

capture=cv2.VideoCapture(1)#开启相机

while(True):
    ret, frame = capture.read()
    # 获取一帧
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 将彩色图像'img'转换为‘HSV'

    mask_B= cv2.inRange(hsv_img, black_D, black_U)
    mask_W= cv2.inRange(hsv_img, white_D,white_U)
    mask_G= cv2.inRange(hsv_img, green_D, green_U)
    # 创建掩码，提取位于颜色范围内的像素
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_B, mask_W), mask_G)
    #合并掩码

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 在二值图像'threshold'中查找轮廓
    # RETR_TREE: 检索所有轮廓，并重建嵌套轮廓的完整层次结构
    # CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，仅保留它们的终点

    i = 0

    # 遍历图像中找到的每个轮廓
    for contour in contours:
        # 忽略第一个轮廓，因为findContours函数会将整个图像检测为一个轮廓
        if i == 0:
            i = 1
            continue

        # 使用cv2.approxPolyDP()函数对轮廓进行多边形逼近
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # 使用drawContours()函数绘制轮廓
        cv2.drawContours(frame, [contour], 0, (0, 0, 255), 5)

        # 计算形状的中心点
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        # 在每个形状的中心点放置形状名称
        if len(approx) == 3:
            cv2.putText(frame , 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(frame , 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(frame, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(frame , 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.putText(frame ,'Circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if()



    cv2.imshow('shape', frame)
    cv2.imshow('Color Blocks', mask)
    cv2.waitKey(1)




