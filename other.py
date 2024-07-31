import cv2
import numpy as np

# 初始化摄像头
cap = cv2.VideoCapture(1)  # 使用默认摄像头

def find_qipan(img):
    """ 检测棋盘并标记网格交叉点 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    cv2.imshow('Edges', edged)  # 显示边缘检测结果

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 确保检测到的是一个四边形
        if len(approx) == 4:
            corners = approx.reshape((4, 2))

            # 按顺序排列四个角
            corners = sorted(corners, key=lambda x: (x[1], x[0]))
            tl, bl, tr, br = corners[0], corners[1], corners[2], corners[3]

            cross_points = []
            for i in range(4):
                for j in range(4):
                    cross_x = int((tl[0] * (3 - i) + tr[0] * i) * (3 - j) / 9 +
                                  (bl[0] * (3 - i) + br[0] * i) * j / 9)
                    cross_y = int((tl[1] * (3 - i) + tr[1] * i) * (3 - j) / 9 +
                                  (bl[1] * (3 - i) + br[1] * i) * j / 9)
                    cross_points.append((cross_x, cross_y))
                    cv2.circle(img, (cross_x, cross_y), 3, (0, 255, 0), -1)

            centers = []
            for i in range(3):
                for j in range(3):
                    center_x = int((cross_points[i * 4 + j][0] + cross_points[i * 4 + j + 1][0] +
                                    cross_points[(i + 1) * 4 + j][0] + cross_points[(i + 1) * 4 + j + 1][0]) / 4)
                    center_y = int((cross_points[i * 4 + j][1] + cross_points[i * 4 + j + 1][1] +
                                    cross_points[(i + 1) * 4 + j][1] + cross_points[(i + 1) * 4 + j + 1][1]) / 4)
                    centers.append((center_x, center_y))
                    cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)

            if len(centers) == 9:
                centers = np.array(centers)
                rect = np.zeros((9, 2), dtype="float32")
                s = centers.sum(axis=1)
                idx_0 = np.argmin(s)
                idx_8 = np.argmax(s)
                diff = np.diff(centers, axis=1)
                idx_2 = np.argmin(diff)
                idx_6 = np.argmax(diff)
                rect[0] = centers[idx_0]
                rect[2] = centers[idx_2]
                rect[6] = centers[idx_6]
                rect[8] = centers[idx_8]
                idxes = [1, 3, 4, 5, 7]
                others = centers[~np.isin(range(len(centers)), [idx_0, idx_2, idx_6, idx_8])]
                idx_l = others[:, 0].argmin()
                idx_r = others[:, 0].argmax()
                idx_t = others[:, 1].argmin()
                idx_b = others[:, 1].argmax()
                found = [idx_l, idx_r, idx_t, idx_b]
                mask = np.isin(range(len(others)), found, invert=False)
                idx_c = np.where(~mask)[0]
                if len(idx_c) == 1:
                    rect[1] = others[idx_t]
                    rect[3] = others[idx_l]
                    rect[4] = others[idx_c]
                    rect[5] = others[idx_r]
                    rect[7] = others[idx_b]
                    for i in range(9):
                        cv2.putText(img, str(i + 1), (int(rect[i][0]), int(rect[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                return rect
    return np.array([])  # 返回空数组而不是空列表

def is_green_background(hsv_img, piece_center):
    """ 检查给定位置是否为绿色背景 """
    x, y = piece_center
    pixel_value = hsv_img[y, x]
    h, s, v = pixel_value
    return (35 <= h <= 85) and (50 <= s <= 255) and (20 <= v <= 255)

def find_qizi(img, centers):
    """ 检测棋子 """
    thresholds = [
        ([0, 0, 0, 180, 255, 120], (0, 0, 0)),  # 黑色，适当增加亮度阈值上限
        ([0, 0, 150, 180, 55, 255], (255, 255, 255))  # 白色，适当增加亮度下限
    ]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_background_color = (0, 255, 0)

    for thresh, color in thresholds:
        lower = np.array([thresh[0], thresh[2], thresh[4]])
        upper = np.array([thresh[1], thresh[3], thresh[5]])
        mask = cv2.inRange(hsv_img, lower, upper)
        cv2.imshow(f"Mask for color {color}", mask)  # 显示掩码结果

        blobs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, center in enumerate(centers):
            has_piece = False
            piece_color = None
            piece_value = 0  # 默认值
            for b in blobs:
                if cv2.contourArea(b) > 200:  # 降低面积阈值以检测较小的棋子
                    x, y, w, h = cv2.boundingRect(b)
                    piece_center = (x + w // 2, y + h // 2)
                    distance = np.linalg.norm(np.array(center) - np.array(piece_center))  # 计算棋子中心与格子中心的距离
                    if distance < w / 2 + 20:  # 允许更大的偏移量
                        has_piece = True
                        piece_color = "黑色" if np.all(color == [0, 0, 0]) else "白色"
                        if is_green_background(hsv_img, piece_center):
                            piece_value = 1
                            cv2.circle(img, piece_center, 5, green_background_color, 2)
                        else:
                            cv2.circle(img, piece_center, 5, color, 2)
                        break
            print(f"格子 {i + 1} 中心坐标: {center}, 是否有棋子: {'有' if has_piece else '无'}, 颜色: {piece_color if has_piece else '无'}, 属性值: {piece_value}")

def main():
    mode = 1  # 初始模式为棋盘检测
    centers = np.array([])  # 初始化为空数组

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 1:
            centers = find_qipan(frame)
        elif mode == 2:
            if centers.size == 0:  # 使用 .size 检查是否为空
                centers = find_qipan(frame)  # 确保在检查棋子之前检测棋盘中心
            find_qizi(frame, centers)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
