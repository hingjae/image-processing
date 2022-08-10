from string import hexdigits
import cv2
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray scale로 바꿔주는 이유는 정확도 향상을 위해서(channel이 많아지면 정확도가 떨어짐)
    frame_gray = cv2.equalizeHist(frame_gray) # equalizeHist 이미지를 단순화시킴

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) #멀티스케일을 한다
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2) # 사각형이 위치해야할 중간 위치 좌표
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # frame의 x, y좌표에 사각형을그림(색깔, 두께)
        faceROI = frame_gray[y:y+h,x:x+w] #방금 선택한 얼굴 안에서 다시 눈을 찾음
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25)) #반지름
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        cv2.imshow('Capture - Face detection', frame)


print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread("../image/marathon_01.jpg")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {} ".format(img.shape[2]))

(height, width) = img.shape[:2] #배열의 첫번째 두번째 값을 각각 할당

cv2.imshow("Original Image", img)

#opencv 패키지에 있는 미리 학습된 얼굴, 눈 인식 모델
face_cascade_name = 'C:/Users/ljh23/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'C:/Users/ljh23/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml'

#오픈cv에서 제공하는 함수를 변수에 넣어서 사용
face_cascade =cv2.CascadeClassifier()
eyes_cascade =cv2.CascadeClassifier()

#cascade함수에 로드 인식 모델을 로드해서 사용
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()