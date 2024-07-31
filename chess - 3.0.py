import cv2
import numpy as np


def coordinates_transformation(rotation_matrix, coordinate_x, coordinate_y):
    # 将旋转矩阵从2x3转换为3x3
    rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])

    # 将二维坐标转换为齐次坐标（加上一个1）
    homogeneous_coordinate = np.array([coordinate_x, coordinate_y, 1])

    # 计算旋转后的坐标
    rotated_coordinate = np.dot(rotation_matrix_3x3, homogeneous_coordinate)

    # 提取旋转后的x和y坐标
    rotated_x = rotated_coordinate[0]
    rotated_y = rotated_coordinate[1]

    return rotated_x, rotated_y


def draw_circle(image, coordinates, color):
    for coordinate in coordinates:
        if len(coordinate) == 3:
            x, y, radius = coordinate
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (int(x), int(y)), radius, color, 2)
                cv2.circle(image, (int(x), int(y)), 2, color, 3)


def keep_point_order(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    return points[sort_idx]


def chess(image):
    circle_image = image.copy()
    # 检测圆
    circles = cv2.HoughCircles(cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                               param1=50, param2=30, minRadius=10, maxRadius=18)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 扩大一点点区域来检查颜色
            height, width, channels = image.shape
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
                white_coordinates.append((x, y, r))
            else:
                black_coordinates.append((x, y, r))


def chess_board(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓，限制面积
    min_area = 10000  # 最小面积
    max_area = 50000  # 最大面积
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    if filtered_contours:
        # 假设最大轮廓是矩形
        rect = max(filtered_contours, key=cv2.contourArea)

        # 找到最小旋转矩形
        rot_rect = cv2.minAreaRect(rect)
        box = cv2.boxPoints(rot_rect)
        box = np.intp(box)

        # 确保点的顺序一致（顺时针或逆时针）
        box = keep_point_order(box)
        # 给四个角上色
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]  # 红、黄、蓝、绿
        for i in range(4):
            cv2.circle(image, tuple(box[i]), 5, colors[i], -1)  # 为每个点上色

        # 绘制最小旋转矩形
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

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
        print(angle)
        if angle > 50:
            angle = angle - 90
            #width, height = height, width

        cv2.putText(image, f"{angle}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 获取旋转矩形的仿射变换矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 旋转图像
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        # 获取旋转矩形的包围矩形
        x, y, w, h = cv2.boundingRect(cv2.transform(np.array([box]), rotation_matrix))
        # 裁剪旋转后的矩形区域
        cropped_image = rotated_image[y:y + h, x:x + w]

        # 旋转坐标
        white_rot_coordinates = []
        black_rot_coordinates = []
        for coordinate in white_coordinates[:]:
            wx, wy = coordinates_transformation(rotation_matrix, coordinate[0], coordinate[1])
            wx = wx - x
            wy = wy - y
            if 0 <= int(wx) < cropped_image.shape[1] and 0 <= int(wy) < cropped_image.shape[0]:
                white_coordinates.remove(coordinate)  # 删除在棋盘格内的棋子坐标
                white_chess_coordinates.append(coordinate)  # 添加旋转前在棋盘格内的棋子坐标
                white_rot_coordinates.append((wx, wy))  # 添加旋转后在棋盘格内的棋子坐标
                cv2.circle(cropped_image, (int(wx), int(wy)), coordinate[2], (255, 255, 255), -1)

        for coordinate in black_coordinates[:]:
            wx, wy = coordinates_transformation(rotation_matrix, coordinate[0], coordinate[1])
            wx = wx - x
            wy = wy - y
            if 0 <= int(wx) < cropped_image.shape[1] and 0 <= int(wy) < cropped_image.shape[0]:
                black_coordinates.remove(coordinate)  # 删除在棋盘格内的棋子坐标
                black_chess_coordinates.append(coordinate)  # 添加旋转前在棋盘格内的棋子坐标
                black_rot_coordinates.append((wx, wy))  # 添加旋转后在棋盘格内的棋子坐标
                cv2.circle(cropped_image, (int(wx), int(wy)), coordinate[2], (0, 0, 0), -1)

        # 绘制网格
        for i in range(1, rows):
            cv2.line(cropped_image, (0, i * cell_height), (w, i * cell_height), (255, 0, 0), 2)
        for j in range(1, cols):
            cv2.line(cropped_image, (j * cell_width, 0), (j * cell_width, h), (255, 0, 0), 2)

        # 初始化3x3矩阵
        grid = np.zeros((rows, cols), dtype=int)

        # 确定每个棋子所在的格子
        for wx, wy in white_rot_coordinates:
            row = int(wy // cell_height)
            col = int(wx // cell_width)
            grid[row, col] = 1  # 1表示白棋

        for bx, by in black_rot_coordinates:
            row = int(by // cell_height)
            col = int(bx // cell_width)
            grid[row, col] = 2  # 2表示黑棋

        # 打印棋盘矩阵
        print("Chess board grid:")
        print(grid)

        cv2.imshow('Grid', cropped_image)

    # 画在棋盘内的圆
    draw_circle(image, white_chess_coordinates, (0, 255, 0))
    draw_circle(image, black_chess_coordinates, (0, 255, 0))
    # 画不在棋盘内的圆
    draw_circle(image, white_coordinates, (0, 0, 255))
    draw_circle(image, black_coordinates, (0, 0, 255))

    # 显示结果
    cv2.imshow("Rotated Image", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    # 读取图像
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
    else:
        while True:
            white_coordinates = []
            black_coordinates = []

            white_chess_coordinates = []
            black_chess_coordinates = []

            ret, img = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            image = img.copy()
            chess(image)
            chess_board(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
