import cv2

video = cv2.VideoCapture('source/drive1.mp4')

count = 0
fps = video.get(cv2.CAP_PROP_FPS)

while video.isOpened():
    try:
        ret, image = video.read()

        if int(video.get(0)) % fps == 0:
            cv2.imwrite(f"frame/frame{count}.jpeg", image)
            print(f'Download {count}')
            count += 1
    except cv2.error:
        continue

print("Done!")
video.release()
