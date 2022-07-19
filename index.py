import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

img = cv2.imread('frame/frame140.jpeg')
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray = grayscale(img)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def mark_img(image, blue_threshold=200, green_threshold=200, red_threshold=200):  # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (image[:, :, 0] < bgr_threshold[0]) \
                 | (image[:, :, 1] < bgr_threshold[1]) \
                 | (image[:, :, 2] < bgr_threshold[2])
    mark[thresholds] = [0, 0, 0]
    return mark


def draw_lines(img, lines, color=(255, 255, 255), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)


# 사다리꼴 모형의 Points
vertices = np.array(
    [[(80, 650), (520, 280), (570, 280), (170, 650)],
     [(950, 600), (630, 280), (680, 280), (1100, 600)]],
    dtype=np.int32)
roi_img = region_of_interest(edges, vertices)  # vertices에 정한 점들 기준으로 ROI 이미지 생성

rho = 1
theta = np.pi / 180
threshold = 50
min_line_len = 100
max_line_gap = 5

lines = hough_lines(roi_img, rho, theta, threshold, min_line_len, max_line_gap)

# 차선과 이미지 합쳐짐
line_edges = weighted_img(lines, img, a=0.8, b=1., c=0.)

mark = np.copy(lines)  # roi_img 복사
mark = mark_img(lines)  # 흰색 차선 찾기

# 흰색 차선 검출한 부분을 원본 image에 overlap 하기
color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
img[color_thresholds] = [0, 0, 255]

plt.figure(figsize=(10, 8))
plt.imshow(mark)
plt.show()
