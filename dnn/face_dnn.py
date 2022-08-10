import cv2
import numpy as np

model_name = 'res10_300x300_ssd_iter_140000.caffemodel' # caffe model
prototxt_name = 'deploy.prototxt.txt' #모델의 설계도 아키텍처 구성도
min_confidence = 0.3 #최소확률
file_name = "../image/soccer_02.jpg"

def detectAndDisplay(frame):
    # pass the blob through the model and obtain the detections 
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name) #model객체를 만듦(model이름과 타입을 가져와)

    # Resizing to a fixed 300x300 pixels and then normalizing it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)) #blob에 image를 넣기전 resize, nomalizing

    model.setInput(blob) #model에 넣음
    detections = model.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2] #confidence는 확률

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence: #최소확률보다 크면 (0.5보다 크면)
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int") #int type으로 바꿈
                    print(confidence, startX, startY, endX, endY)
     
                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100) #소수점 둘째자리까지 확률 표시
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show the output image
    cv2.imshow("Face Detection by dnn", frame)

print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread(file_name)
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()