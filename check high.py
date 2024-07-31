import cv2
import numpy as np

capture = cv2.VideoCapture(1)  # 开启相机

while True:

    # 读取图像
    ret, image = capture.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯滤波器平滑图像以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每一个轮廓
    for contour in contours:
        # 多边形逼近
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 画出边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 打印物体长度
            print(f"物体的长度（宽度）为: {w} 像素")
            print(f"物体的高度为: {h} 像素")

    # 显示结果
    cv2.imshow('Contours', image)
    cv2.waitKey(1)

