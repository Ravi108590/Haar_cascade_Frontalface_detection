import cv2 
import os #we can create dir, path, 

dataset = "dataset"
sub_folder = "champ"
#act like: dataset/champ
path = os.path.join(dataset,sub_folder)
#if not the exact folder then will create it 
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)    
#trained face detector
weight = "haarcascade_frontalface_default.xml"
#loads the trained classifier into the memory
haar_cascade = cv2.CascadeClassifier(weight)
#0-> default webcam, 1->external webcam, 2->another webcam
cam = cv2.VideoCapture(0)
count = 0
while count<31:
    _, img = cam.read()
    #no need color image only brightness need, which keep it faster, simple
    #to detect
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detects the faces at multiple scale like some image for way , close, ..
    #1.3:scale factor, 4:minimum neighbours
    face = haar_cascade.detectMultiScale(gray_img, 1.3, 4)

    for (x,y,w,h) in face:
        print(count)
        #(x,y):top-left, (x+w,y+h):Bottom-left, 2:thickness of color line
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 32)
        #only needs the faces dataset so need to resize the captured image to save it
        face_only = gray_img[y:y+h, x:x+w]
        resized_img = cv2.resize(face_only, (width, height))
        #saving into the created path
        cv2.imwrite("%s/%s.jpg" %(path,count),resized_img)
        count+=1

    cv2.imshow("Facedetection", img)
    #esc press for exit
    key = cv2.waitKey(10)

    if key == 27:
        break
print("Successfully stored the captured data") 
cam.release()
cv2.destroyAllWindows()
