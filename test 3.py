import cv2
import numpy as np

# 定义棋子颜色的 HSV 范围
white_D = np.array([15, 15, 200])
white_U = np.array([180, 55, 255])
black_D = np.array([0, 0, 0])
black_U = np.array([180, 255, 100])

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

def detect_chess_pieces(hsv_img, grid_coordinates, image):
    """ 检测棋子颜色并判断出哪个格子中有棋子 """
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2

    for i, (top_left, bottom_right) in enumerate(grid_coordinates):
        grid_section = hsv_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 创建黑色和白色的掩码
        mask_black = cv2.inRange(grid_section, black_D, black_U)
        mask_white = cv2.inRange(grid_section, white_D, white_U)

        # 计算黑色和白色的像素数
        black_pixels = cv2.countNonZero(mask_black)
        white_pixels = cv2.countNonZero(mask_white)

        # 在每个格子中心显示编号
        center_x_grid = (top_left[0] + bottom_right[0]) // 2
        center_y_grid = (top_left[1] + bottom_right[1]) // 2

        # 计算相对坐标
        relative_x = center_x_grid - center_x
        relative_y = center_y_grid - center_y

        cv2.putText(image, f"{i+1}", (center_x_grid - 10, center_y_grid + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(image, f"({relative_x}, {relative_y})", (center_x_grid - 30, center_y_grid + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        print((center_x_grid - 30, center_y_grid + 30))

        # 判断哪个颜色的像素数更多
        if black_pixels > white_pixels and black_pixels > 100:  # 阈值决定是否有棋子
            print(f"格子 {i+1} 有一个黑色棋子")
        elif white_pixels > black_pixels and white_pixels > 100:
            print(f"格子 {i+1} 有一个白色棋子")
        else:
            print(f"格子 {i+1} 是空的")

def process_frame(frame):
    """ 处理每一帧以检测棋盘和棋子 """
    # 转换为灰度图像并进行边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 检测轮廓并找到面积最大的轮廓，假设是棋盘
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        board_contour = contours[0]

        # 获取棋盘的四个顶点
        epsilon = 0.1 * cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, epsilon, True)

        if len(approx) == 4:
            # 获取棋盘的顶点，并进行透视变换
            pts1 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            pts2 = np.float32([[0, 0], [90, 0], [90, 90], [0, 90]])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(frame, matrix, (90, 90))

            # 转换为 HSV 图像
            hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

            # 获取九宫格的坐标位置
            grid_coordinates = get_grid_coordinates()

            # 检测棋子
            detect_chess_pieces(hsv_warped, grid_coordinates, warped)

            warped_counter = 2
            # 显示棋盘图像
            cv2.imshow('Warped Chessboard', warped)
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

        process_frame(frame)

        # 显示原始帧
        cv2.imshow('Original Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


