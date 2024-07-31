import cv2
import numpy as np

cv2.VideoCapture(1)
capture=cv2.VideoCapture(1)#开启相机





# 回调函数必须要写
def empty(i):
    # 提取滑动条的数值 共6个
    hue_min = cv2.getTrackbarPos("Hue Min", "sb")
    hue_max = cv2.getTrackbarPos("Hue Max", "sb")
    sat_min = cv2.getTrackbarPos("Sat Min", "sb")
    sat_max = cv2.getTrackbarPos("Sat Max", "sb")
    val_min = cv2.getTrackbarPos("Val Min", "sb")
    val_max = cv2.getTrackbarPos("Val Max", "sb")

    # 颜色空间阈值
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    # 根据颜色空间阈值生成掩膜
    imgMASK = cv2.inRange(imgHSV, lower, upper)

    cv2.imshow("img", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", imgMASK)


# 参数(‘窗口标题’,默认参数)
cv2.namedWindow("sb")
# 设置窗口大小
cv2.resizeWindow("sb", 640, 240)

# 第一个参数时滑动条的名字，
# 第二个参数是滑动条被放置的窗口的名字，
# 第三个参数是滑动条默认值，
# 第四个参数时滑动条的最大值，
# 第五个参数时回调函数，每次滑动都会调用回调函数。
cv2.createTrackbar("Hue Min", "sb", 0, 179, empty)
cv2.createTrackbar("Hue Max", "sb", 179, 179, empty)
cv2.createTrackbar("Sat Min", "sb", 0, 255, empty)
cv2.createTrackbar("Sat Max", "sb", 255, 255, empty)
cv2.createTrackbar("Val Min", "sb", 0, 255, empty)
cv2.createTrackbar("Val Max", "sb", 255, 255, empty)


while(True):
    ret, img = capture.read()
    # 转HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 提取滑动条的数值 共6个
    hue_min = cv2.getTrackbarPos("Hue Min", "sb")
    hue_max = cv2.getTrackbarPos("Hue Max", "sb")
    sat_min = cv2.getTrackbarPos("Sat Min", "sb")
    sat_max = cv2.getTrackbarPos("Sat Max", "sb")
    val_min = cv2.getTrackbarPos("Val Min", "sb")
    val_max = cv2.getTrackbarPos("Val Max", "sb")

    # 颜色空间阈值
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    # 根据颜色空间阈值生成掩膜
    imgMASK = cv2.inRange(imgHSV, lower, upper)

    cv2.imshow("img", img)
    #cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", imgMASK)


    # 按任意键关闭
    cv2.waitKey(1)
