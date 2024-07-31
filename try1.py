import cv2
import numpy as np


class ChessBoardProcessor:
    def __init__(self):
        self.white_coordinates = []
        self.black_coordinates = []

        self.white_chess_coordinates = []
        self.black_chess_coordinates = []

        self.white_rot_coordinates = []
        self.black_rot_coordinates = []

        self.grid = np.zeros((3, 3), dtype=int)

    def coordinates_transformation(self, rotation_matrix, coordinate_x, coordinate_y):
        rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])
        homogeneous_coordinate = np.array([coordinate_x, coordinate_y, 1])
        rotated_coordinate = np.dot(rotation_matrix_3x3, homogeneous_coordinate)
        rotated_x = rotated_coordinate[0]
        rotated_y = rotated_coordinate[1]
        return rotated_x, rotated_y

    def draw_circle(self, image, coordinates, color):
        for coordinate in coordinates:
            if len(coordinate) == 3:
                x, y, radius = coordinate
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (int(x), int(y)), radius, color, 2)
                    cv2.circle(image, (int(x), int(y)), 2, color, 3)

    def the_chess_real_sit(self, a, b):
        x = 100 - 0.896 * a
        y = 200 - 0.896 * b
        return x, y

    def the_square_real_sit(self, c, d):
        x = 100 - 0.896 * c
        y = 200 - 0.896 * d
        return x, y

    def keep_point_order(self, points):
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sort_idx = np.argsort(angles)
        return points[sort_idx]

    def summarize_chess_info(self, grid_centers, white_positions, black_positions):
        summary = []
        for i, center in enumerate(grid_centers):
            row, col = i // 3, i % 3
            chess_info = {
                "格子中心坐标": center,
                "棋子颜色": "无",
                "棋子坐标": "无",
                "是否在棋盘上": "否"
            }

            if (row, col) in white_positions:
                chess_info["棋子颜色"] = "白色"
                index = white_positions.index((row, col))
                chess_info["棋子坐标"] = self.white_rot_coordinates[index]
                chess_info["是否在棋盘上"] = "是"

            elif (row, col) in black_positions:
                chess_info["棋子颜色"] = "黑色"
                index = black_positions.index((row, col))
                chess_info["棋子坐标"] = self.black_rot_coordinates[index]
                chess_info["是否在棋盘上"] = "是"

            summary.append(chess_info)
        return summary

    def chess(self, image):
        # 转换图像为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用霍夫圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=50, param2=30, minRadius=10, maxRadius=18)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                height, width = gray.shape
                offset = 5
                x_min = max(0, x - offset)
                y_min = max(0, y - offset)
                x_max = min(width, x + offset)
                y_max = min(height, y + offset)

                region = gray[y_min:y_max, x_min:x_max]
                mean_color = np.mean(region)

                if mean_color > 127:
                    self.white_coordinates.append((x, y, r))
                    print(f"White piece detected at ({x}, {y}), mean color: {mean_color}")
                else:
                    self.black_coordinates.append((x, y, r))
                    print(f"Black piece detected at ({x}, {y}), mean color: {mean_color}")

    def chess_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 10000
        max_area = 50000
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        if filtered_contours:
            rect = max(filtered_contours, key=cv2.contourArea)
            rot_rect = cv2.minAreaRect(rect)
            box = cv2.boxPoints(rot_rect)
            box = np.intp(box)

            box = self.keep_point_order(box)
            colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
            for i in range(4):
                cv2.circle(image, tuple(box[i]), 5, colors[i], -1)

            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

            rows = 3
            cols = 3
            width = int(rot_rect[1][0])
            height = int(rot_rect[1][1])
            cell_width = width // cols
            cell_height = height // rows

            center = tuple(map(int, rot_rect[0]))
            angle = rot_rect[2]
            if angle > 50:
                angle = angle - 90

            first_point = box[0]
            third_point = box[2]

            x1, y1 = first_point
            x2, y2 = third_point

            grid_centers = []
            for i in range(3):
                for j in range(3):
                    center_x = (x1 + (x2 - x1) / 6) + (x2 - x1) / 3 * j
                    center_y = (y1 + (y1 - y2) / 6) + (y1 - y2) / 3 * i
                    grid_centers.append((center_x, center_y))

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            x, y, w, h = cv2.boundingRect(cv2.transform(np.array([box]), rotation_matrix))
            cropped_image = rotated_image[y:y + h, x:x + w]

            for coordinate in self.white_coordinates[:]:
                wx, wy = self.coordinates_transformation(rotation_matrix, coordinate[0], coordinate[1])
                wx = wx - x
                wy = wy - y
                if 0 <= int(wx) < cropped_image.shape[1] and 0 <= int(wy) < cropped_image.shape[0]:
                    self.white_coordinates.remove(coordinate)
                    self.white_chess_coordinates.append(coordinate)
                    self.white_rot_coordinates.append((wx, wy))
                    cv2.circle(cropped_image, (int(wx), int(wy)), coordinate[2], (255, 255, 255), -1)

            for coordinate in self.black_coordinates[:]:
                wx, wy = self.coordinates_transformation(rotation_matrix, coordinate[0], coordinate[1])
                wx = wx - x
                wy = wy - y
                if 0 <= int(wx) < cropped_image.shape[1] and 0 <= int(wy) < cropped_image.shape[0]:
                    self.black_coordinates.remove(coordinate)
                    self.black_chess_coordinates.append(coordinate)
                    self.black_rot_coordinates.append((wx, wy))
                    cv2.circle(cropped_image, (int(wx), int(wy)), coordinate[2], (0, 0, 0), -1)

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print("Invalid cropping area.")
                cropped_image = np.zeros((1, 1, 3), dtype=np.uint8)
            else:
                cropped_image = rotated_image[y:y + h, x:x + w]

            for i in range(1, rows):
                cv2.line(cropped_image, (0, i * cell_height), (w, i * cell_height), (255, 0, 0), 2)
            for j in range(1, cols):
                cv2.line(cropped_image, (j * cell_width, 0), (j * cell_width, h), (255, 0, 0), 2)

            white_positions = []
            black_positions = []

            for wx, wy in self.white_rot_coordinates:
                row = int(wy // cell_height)
                col = int(wx // cell_width)
                white_positions.append((row, col))

                self.grid[row, col] = 1

            for bx, by in self.black_rot_coordinates:
                row = int(by // cell_height)
                col = int(bx // cell_width)
                black_positions.append((row, col))
                self.grid[row, col] = 2

            summary = self.summarize_chess_info(grid_centers, white_positions, black_positions)
            for info in summary:
                print(
                    f"格子中心坐标: {info['格子中心坐标']}, 棋子颜色: {info['棋子颜色']}, 棋子坐标: {info['棋子坐标']}, 是否在棋盘上: {info['是否在棋盘上']}")

            self.draw_circle(image, self.white_chess_coordinates, (0, 255, 0))
            self.draw_circle(image, self.black_chess_coordinates, (0, 255, 0))
            self.draw_circle(image, self.white_coordinates, (0, 0, 255))
            self.draw_circle(image, self.black_coordinates, (0, 0, 255))

            cv2.imshow('Grid', cropped_image)
            print(self.grid)

        else:
            print("No chessboard detected.")

        cv2.imshow("Rotated Image", image)
        cv2.waitKey(1)

    def process_frame(self, image):
        # 清空上一次检测的结果
        self.white_coordinates.clear()
        self.black_coordinates.clear()
        self.white_chess_coordinates.clear()
        self.black_chess_coordinates.clear()
        self.white_rot_coordinates.clear()
        self.black_rot_coordinates.clear()
        self.grid.fill(0)

        # 处理图像
        self.chess(image)
        self.chess_board(image)


if __name__ == "__main__":
    # 使用默认摄像头
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5.98)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
    else:
        processor = ChessBoardProcessor()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            image = img.copy()
            processor.process_frame(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
