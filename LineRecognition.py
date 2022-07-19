import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
from tqdm import tqdm
import numpy as np
import timeit

video = cv2.VideoCapture('source/drive1.mp4')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame_width, frame_height))

past_left_line = []
past_right_line = []
past_left_inclination = []
past_right_inclination = []


# 흑백화
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 가우시안 블러
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# 엣지 따기
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


# 관심지역 설정
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(0, 0, 255), thickness=10):
    isLeftLine = False
    isRightLine = False

    font = cv2.FONT_HERSHEY_SIMPLEX

    for line in lines:
        for x1, y1, x2, y2 in line:
            inclination = (y2 - y1) / (x2 - x1)
            # 왼쪽 차선 기울기 양수
            # 오른쪽 차선 기울기 음수
            end_y = 380
            start_x = int(((650 - y1) / inclination) + x1)
            end_x = int(((end_y - y1) / inclination) + x1)
            if inclination > 0 and isLeftLine == False:
                isLeftLine = True
                past_left_line.append([(start_x, 650), (end_x, end_y)])
                past_left_inclination.append(inclination)
                pass
            elif inclination < 0 and isRightLine == False:
                isRightLine = True
                past_right_line.append([(start_x, 650), (end_x, end_y)])
                past_right_inclination.append(inclination)
                pass
            else:
                continue
            cv2.line(img, (start_x, 650), (end_x, end_y), color, thickness)

    # 차선이 한 개만 인식이 되었을때 직전의 차선을 그림
    if not isLeftLine and len(past_left_line) != 0:
        past_start_coord, past_end_coord = past_left_line[-1]
        cv2.line(img, past_start_coord, past_end_coord, color, thickness)
    if not isRightLine and len(past_right_line) != 0:
        past_start_coord, past_end_coord = past_right_line[-1]
        cv2.line(img, past_start_coord, past_end_coord, color, thickness)

    # 왼쪽 차선과 오른쪽 차선의 기울기 표시
    if len(past_right_inclination) != 0:
        cv2.putText(img, f'right: {round(past_right_inclination[-1], 2)}', (900, 400), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    if len(past_left_inclination) != 0:
        cv2.putText(img, f'left: {round(past_left_inclination[-1], 2)}', (200, 400), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # 차선의 기울기에 따른 핸들 방향 조절
    if len(past_right_inclination) != 0 and len(past_left_inclination) != 0 and past_left_inclination[-1] < 0.55 and \
            past_right_inclination[-1] > -0.9:
        cv2.putText(img, '<--', (550, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif len(past_right_inclination) != 0 and len(past_left_inclination) != 0 and past_left_inclination[-1] > 0.72 and \
            past_right_inclination[-1] > -0.65:
        cv2.putText(img, '-->', (550, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, '^', (550, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


# 차선과 이미지 합치기
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)


def videoDetector(cam):
    while True:
        try:
            start_t = timeit.default_timer()
            # 캡처 이미지 불러오기
            ret, img = cam.read()
            # 영상 압축
            img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)

            gray = grayscale(img)

            kernel_size = 5
            blur_gray = gaussian_blur(gray, kernel_size)

            low_threshold = 50
            high_threshold = 200
            edges = canny(blur_gray, low_threshold, high_threshold)

            height, width = img.shape[:2]  # 이미지 높이, 너비

            # 관심지역 위치
            vertices = np.array(
                [[(375, 400), (520, 280), (570, 280), (430, 400)],
                 [(730, 400), (630, 280), (680, 280), (830, 400)]],
                dtype=np.int32)

            mask = region_of_interest(edges, vertices)

            rho = 1
            theta = np.pi / 180
            threshold = 50
            min_line_len = 100
            max_line_gap = 5

            # 차선만 인식
            lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

            # 차선과 이미지 합쳐짐
            line_edges = weighted_img(lines, img, a=0.8, b=1., c=0.)

            terminate_t = timeit.default_timer()
            FPS = 'fps' + str(int(1. / (terminate_t - start_t)))
            cv2.putText(line_edges, FPS, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            out.write(line_edges)
        except TypeError as e:
            continue
        except OverflowError as e:
            continue
        except cv2.error as e:
            continue


videoDetector(video)
out.release()
