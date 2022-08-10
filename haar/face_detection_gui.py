import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

face_cascade_name = '../opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = '../opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
file_name = 'image/marathon_01.jpg'
title_name = 'Haar cascade object detection'
frame_width = 500

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    (height, width) = read_image.shape[:2]
    frameSize = int(sizeSpin.get())
    ratio = frameSize / width
    dimension = (frameSize, int(height * ratio))
    read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detectAndDisplay(read_image)

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
        # cv2.imshow('Capture - Face detection', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        detection.config(image=imgtk)
        detection.image = imgtk

#main
main = Tk()
main.title(title_name)
main.geometry() #컴포넌트를 배열할 수 있게해줌

read_image = cv2.imread("../image/marathon_01.jpg")
(height, width) = read_image.shape[:2] #배열의 첫번째 두번째 값을 각각 할당
ratio = frame_width / width #프레임 너비와 실제 그림 너비의 비율을 구함.
dimension = (frame_width, int(height * ratio))
read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA) #사이즈를 재조정

image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) #opencv는 BGR로 되어있는데 그림을 보여줄땐 RGB로 보여줘야함.
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

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

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4) # 4개의 column
sizeLabel=Label(main, text='Frame Width : ')       
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=frame_width)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=2000, increment=100, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)
detectAndDisplay(read_image)

main.mainloop()