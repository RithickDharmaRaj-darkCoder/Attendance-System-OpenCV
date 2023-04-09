import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'Students_images'
std_imgs = []
std_img_namelist = os.listdir(path)
std_namelist = []

for img in std_img_namelist:
    current_img = cv2.imread(f'{path}/{img}')
    std_imgs.append(current_img)
    std_namelist.append(os.path.splitext(img)[0])

def put_present(name):
    with open('Attendance.csv', 'r+') as f:
        data_in_line = f.readlines()
        namelist = []
        
        for line in data_in_line:
            entry = line.split(',')
            namelist.append(entry[0])
        
        if name not in namelist:
            time = datetime.now()
            Ftime = time.strftime('%H:%M')
            f.writelines(f'\n{name}, {Ftime}')
            print(f'{name} Present!')

def encode_imgs(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encoded_std_imgs = encode_imgs(std_imgs)
print(f'Encoding completed for {len(encoded_std_imgs)} students.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgF = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgF = cv2.cvtColor(imgF, cv2.COLOR_BGR2RGB)
    
    faces_in_Cframe = fr.face_locations(imgF)
    encoded_faces_in_Cframe = fr.face_encodings(imgF, faces_in_Cframe)
    
    for encodedface, facelocation in zip(encoded_faces_in_Cframe, faces_in_Cframe):
        matches = fr.compare_faces(encoded_std_imgs, encodedface)
        facedist = fr.face_distance(encoded_std_imgs, encodedface)
        matchIndex = np.argmin(facedist)
        
        if matches[matchIndex]:
            std_name = std_namelist[matchIndex].upper()
            
            y1, x2, y2, x1 = facelocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, std_name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print(std_name)
            put_present(std_name)
    
    cv2.imshow('Web Cam Frames', img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
