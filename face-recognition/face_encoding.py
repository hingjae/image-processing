import cv2
import face_recognition
import pickle # 바이너리 프로토콜을 만드는 모듈 정보를 직렬로..

dataset_paths = ['dataset/son/', 'dataset/tedy/']
names = ['Son', 'Tedy']
number_images = 10 #이미지 개수는 단순화 하기 위해 10개만
image_type = '.jpg' # .png 도 가능
encoding_file = 'encodings.pickle'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths): #
    # extract the person name from names
    name = names[i]

    for idx in range(number_images): # 1~10번까지의 이미지를 읽음
        file_name = dataset_path + str(idx+1) + image_type #ex)'dataset/son/' + '1' + '.jpg'

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv는 bgr로 되어있기 때문에 rgb로 변환이 필요함.

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, #face_location()으로 얼굴 위치 찾기
            model=model_method) #model_method == 'cnn'

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes) #face_encodings()로 인코딩

        #인코딩은 128개의 실수로 되어있다.!
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)
        
# Save the facial encodings + names to disk
data = {"encodings": knownEncodings, "names": knownNames} #dic
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
