import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

face_cascade_name = 'C:/Users/ljh23/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'C:/Users/ljh23/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml'
file_name = '../video/tedy_01.mp4'
title_name = 'Haar cascade object detection Video'
frame_width = 500
cap = cv2.VideoCapture()

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./video",title = "Select file",filetypes = (("MP4 files","*.mp4"),("all files","*.*")))
    print('File name : ', file_name)
    global cap #글로벌 변수 cap
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()

def detectAndDisplay():
    _, frame = cap.read()
    # frame을 gray로 바꿔줌. (인식 개선을 위함.)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) #얼굴들 찾기
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        faceROI = frame_gray[y:y+h,x:x+w] #얼굴영역을 찾고 그 안에서 눈을 찾음
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    # 밑에서 만들었던 frame 안에 넣는 작업
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #rgb로 바꿔줌
    img = Image.fromarray(cv2image) #
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk #lmain에 넣고
    lmain.configure(image=imgtk)
    lmain.after(10, detectAndDisplay) #detectAndDisplay를 다시 call 함. 

#main
main = Tk()
main.title(title_name)
main.geometry()

#Graphics window
label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=0, columnspan=4, sticky=(W, E))
imageFrame = Frame(main)
imageFrame.grid(row=2,column=0,columnspan=4)
  
#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()  #Starts GUI
