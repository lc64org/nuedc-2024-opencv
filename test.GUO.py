import cv2
import numpy as np


# 定义棋子颜色的 HSV 范围
white_D = np.array([15, 20, 183])
white_U = np.array([180, 55, 222])
black_D = np.array([0, 0, 0])
black_U = np.array([180, 255, 92])

# 定义绿色背景的 HSV 范围
green_D = np.array([110, 40, 40])
green_U = np.array([85, 255, 255])


def get_grid_coordinates():
    """ 返回九宫格的坐标位置 """
    grid_coordinates = []
    square_size = 90 // 3
    for row in range(3):
        for col in range(3):
            top_left = (col * square_size, row * square_size)
            bottom_right = ((col + 1) * square_size, (row + 1) * square_size)
            grid_coordinates.append((top_left, bottom_right))
    return grid_coordinates


def detect_chess_pieces(hsv_img, grid_coordinates, image, matrix):
    """ 检测棋子颜色并判断出哪个格子中有棋子 """
    for i, (top_left, bottom_right) in enumerate(grid_coordinates):
        grid_section = hsv_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 创建黑色、白色和绿色的掩码
        mask_black = cv2.inRange(grid_section, black_D, black_U)
        mask_white = cv2.inRange(grid_section, white_D, white_U)
        mask_green = cv2.inRange(grid_section, green_D, green_U)

        # 计算黑色、白色和绿色的像素数
        black_pixels = cv2.countNonZero(mask_black)
        white_pixels = cv2.countNonZero(mask_white)
        green_pixels = cv2.countNonZero(mask_green)

        # 判断格子是否是绿色背景
        if green_pixels / (grid_section.size // 3) > (black_pixels + white_pixels) * 0.8:  # 设定绿色像素的阈值
            print(f"格子 {i + 1} 是空的")
        else:
            if black_pixels > white_pixels and black_pixels > 100:
                print(f"格子 {i + 1} 有一个黑色棋子")
                # 识别黑色棋子中心
                moments = cv2.moments(mask_black)
                if moments['m00'] > 0:
                    center_x = int(moments['m10'] / moments['m00']) + top_left[0]
                    center_y = int(moments['m01'] / moments['m00']) + top_left[1]
                    cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)
                    print(f"黑色棋子位置: ({center_x}, {center_y})")
            elif white_pixels > black_pixels and white_pixels > 100:
                print(f"格子 {i + 1} 有一个白色棋子")
                # 识别白色棋子中心
                moments = cv2.moments(mask_white)
                if moments['m00'] > 0:
                    center_x = int(moments['m10'] / moments['m00']) + top_left[0]
                    center_y = int(moments['m01'] / moments['m00']) + top_left[1]
                    cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)
                    print(f"白色棋子位置: ({center_x}, {center_y})")
            else:
                print(f"格子 {i + 1} 是空的")

        # 计算每个格子的中心点
        center_x_grid = (top_left[0] + bottom_right[0]) // 2
        center_y_grid = (top_left[1] + bottom_right[1]) // 2

        # 使用逆变换将中心点从变换后图像映射回原始图像
        center_point_warped = np.array([[center_x_grid, center_y_grid]], dtype='float32')
        center_point_warped = np.array([center_point_warped])
        try:
            center_point_original = cv2.perspectiveTransform(center_point_warped, np.linalg.inv(matrix))
            center_x_original = int(center_point_original[0][0][0])
            center_y_original = int(center_point_original[0][0][1])

            # 在原始图像上标注格子的编号
            text = f"{i + 1}"
            cv2.putText(image, text, (center_x_original - 10, center_y_original + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

            # 打印格子编号和中心坐标
            print(f"格子 {i + 1}: 中心坐标 ({center_x_original}, {center_y_original})")
        except cv2.error as e:
            print(f"Error in perspective transform: {e}")


def correct_rotation_and_transform(pts):
    """ 根据点的顺序校正棋盘并进行透视变换 """
    # 使用最小面积矩形包络找到可能的棋盘四个顶点
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)  # 将 box 转换为浮点数数组

    # 对点进行排序以形成左上、右上、右下、左下的顺序
    box = sorted(box, key=lambda x: (x[0], x[1]))

    # 将左上和右上点按y排序，左下和右下点按y排序
    if box[0][1] > box[1][1]:
        box[0], box[1] = box[1], box[0]
    if box[2][1] < box[3][1]:
        box[2], box[3] = box[3], box[2]

    # 确保顶点顺序为左上、右上、右下、左下
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    pts2 = np.float32([[0, 0], [90, 0], [90, 90], [0, 90]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 检查矩阵是否奇异
    if np.linalg.matrix_rank(matrix) < 3:
        print("检测到奇异矩阵，无法计算透视变换")
        return None

    return matrix


def process_frame(frame):
    """ 处理每一帧以检测棋盘和棋子 """
    # 转换为灰度图像并进行边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊来减少噪声影响
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # 检测轮廓并找到面积最大的轮廓，假设是棋盘
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        for contour in contours:
            # 计算轮廓的周长并进行多边形逼近
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                # 获取棋盘的顶点，并进行透视变换
                matrix = correct_rotation_and_transform(approx.reshape(4, 2))

                if matrix is None:
                    print("透视变换矩阵无效，跳过处理")
                    continue

                warped = cv2.warpPerspective(frame, matrix, (90, 90))

                # 转换为 HSV 图像
                hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

                # 获取九宫格的坐标位置
                grid_coordinates = get_grid_coordinates()

                # 检测棋子
                detect_chess_pieces(hsv_warped, grid_coordinates, frame, matrix)


                # 显示棋盘图像
                cv2.imshow('Warped Chessboard', warped)
                cv2.waitKey(1)
                break
            else:
                print("无法找到棋盘")
    else:
        print("无法找到轮廓")


def main():
    capture = cv2.VideoCapture(1)  # 开启相机

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        circle_image = frame.copy()
        gray = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)

        # 检测圆
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=50, param2=30, minRadius=10, maxRadius=20)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (a, b, r) in circles:
                # 绘制圆和中心点
                cv2.circle(circle_image, (a, b), r, (0, 255, 0), 4)
                cv2.circle(circle_image, (a, b), 2, (0, 0, 255), 3)

                height, width, channels = frame.shape
                # 扩大一点点区域来检查颜色
                offset = 5  # 偏移量
                x_min = max(0, a - offset)
                y_min = max(0, b - offset)
                x_max = min(width, a + offset)
                y_max = min(height, b + offset)

                # 检查区域的像素颜色
                region = circle_image[y_min:y_max, x_min:x_max]
                mean_color = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))

                # 判断黑白色
                if mean_color > 127:  # 127 是灰度值的中点
                    color = "White"
                else:
                    color = "Black"

                # 显示结果
                cv2.putText(circle_image, f"{color}", (a - 10, b - r - 10),
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
            output_image = frame.copy()
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
            rotated_image = cv2.warpAffine(circle_image, rotation_matrix, (frame .shape[1], frame.shape[0]),
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
            cv2.imshow('Original Frame1', circle_image)

        process_frame(frame)

        # 显示原始帧
        cv2.imshow('Original Frame2', frame)


    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
