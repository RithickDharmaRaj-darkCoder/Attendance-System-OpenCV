import cv2
import numpy as np
import face_recognition as fr

# loading Image & converting to RGB
img_ts1 = fr.load_image_file('Images/Tony_Stark1.jpg')
img_ts1 = cv2.cvtColor(img_ts1, cv2.COLOR_BGR2RGB)
img_test = fr.load_image_file('Images/Tom_Holland.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

face_locate = fr.face_locations(img_ts1)[0]
encode_ts1 = fr.face_encodings(img_ts1)[0]
cv2.rectangle(img_ts1, (face_locate[3], face_locate[0]), (face_locate[1], face_locate[2]), (0, 255, 0), 2)

face_locate_test = fr.face_locations(img_test)[0]
encode_test = fr.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_locate_test[3], face_locate_test[0]), (face_locate_test[1], face_locate_test[2]), (255, 0, 255), 2)

# Compare the pics...
result = fr.compare_faces([encode_ts1], encode_test)
face_dis = fr.face_distance([encode_ts1], encode_test)
print(result, face_dis)

# Text in Output pic
cv2.putText(img_test, f'{result} {round(face_dis[0], 1)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

#cv2.imshow('Tonk Stark 1', img_ts1)
cv2.imshow('Test Image', img_test)
cv2.waitKey(0)

