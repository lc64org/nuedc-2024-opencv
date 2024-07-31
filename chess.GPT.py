import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, image = cap.read()
    if not ret:
        break  # 如果读取图像失败，退出循环

    circle_image = image.copy()
    gray = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)

    # 检测圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=20)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 绘制圆和中心点
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 4)
            cv2.circle(circle_image, (x, y), 2, (0, 0, 255), 3)

            height, width, channels = image.shape
            # 扩大一点点区域来检查颜色
            offset = 5  # 偏移量
            x_min = max(0, x - offset)
            y_min = max(0, y - offset)
            x_max = min(width, x + offset)
            y_max = min(height, y + offset)

            # 检查区域的像素颜色
            region = circle_image[y_min:y_max, x_min:x_max]
            mean_color = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))

            # 判断黑白色
            if mean_color > 127:  # 127 是灰度值的中点
                color = "White"
            else:
                color = "Black"

            # 显示结果
            cv2.putText(circle_image, f"{color}", (x - 10, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 应用边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到轮廓
    if contours:
        # 假设最大轮廓是棋盘
        rect = max(contours, key=cv2.contourArea)

        # 找到最小旋转矩形
        rot_rect = cv2.minAreaRect(rect)
        box = cv2.boxPoints(rot_rect)
        box = np.intp(box)

        # 绘制最小旋转矩形
        output_image = image.copy()
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

        # 将最小旋转矩形分割成3x3网格
        rows = 3
        cols = 3
        width = int(rot_rect[1][0])
        height = int(rot_rect[1][1])
        cell_width = width // cols
        cell_height = height // rows

        # 获取旋转矩形的中心和角度
        center = tuple(map(int, rot_rect[0]))
        angle = rot_rect[2]

        # 修正角度
        if angle < -45:
            angle += 90

        # 获取旋转矩形的仿射变换矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋转图像
        rotated_image = cv2.warpAffine(circle_image, rotation_matrix, (image.shape[1], image.shape[0]),
                                       flags=cv2.INTER_CUBIC)

        # 获取旋转矩形的包围矩形
        box = np.intp(box)
        x, y, w, h = cv2.boundingRect(box)

        # 确保裁剪区域有效
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print("Invalid cropping area.")
            cropped_image = np.zeros((1, 1, 3), dtype=np.uint8)  # 生成一个空白图像
        else:
            # 裁剪旋转后的矩形区域
            cropped_image = rotated_image[y:y + h, x:x + w]

            # 绘制网格
            for i in range(1, rows):
                cv2.line(cropped_image, (0, i * cell_height), (w, i * cell_height), (255, 0, 0), 2)
            for j in range(1, cols):
                cv2.line(cropped_image, (j * cell_width, 0), (j * cell_width, h), (255, 0, 0), 2)

        # 显示结果
        cv2.imshow("Rotated Image", output_image)
        cv2.imshow('Grid', cropped_image)

    # 显示棋盘和棋子
    cv2.imshow('chess', circle_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
