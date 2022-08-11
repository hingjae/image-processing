import cv2
import numpy as np

min_confidence = 0.5

# Load Yolo 욜로 모델을 가져올 때 이미 학습이된(weight) 파일을 가져와야함.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f: #파일을 read하고 f에 넣음
    classes = [line.strip() for line in f.readlines()] #coco.names를 한줄씩 읽으며 claasses에 삽입
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3)) # 물건마다 다른색깔을 부여할 수 있음

# Loading image
img = cv2.imread("../image/yolo_01.jpg") #이미지를 불러옴
#img = cv2.resize(img, None, fx=0.4, fy=0.4) #사이즈 줄이기
height, width, channels = img.shape # 이미지 객체에서 넓이 높이 채널을 가져옴.
cv2.imshow("Original Image", img)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores) #score중에 가장 큰 arg를 클래스 아이디에 넣음
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4) #NMSBoxed를 이용해 노이즈(한공간에 박스가 여러개 생기는 것)방지.
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)

cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
