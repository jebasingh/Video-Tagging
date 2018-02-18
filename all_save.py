import numpy as np
import cv2
import time
import os
from PIL import Image
cap = cv2.VideoCapture('aa.mp4')

face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade4 = cv2.CascadeClassifier('haarcascade_frontalface.xml')
#face_cascade5 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


p_frame_thresh = 300000
total=0
key=0
size = (480, 240)
ret, prev_frame = cap.read()
path = 'user'
#prev_frame = cv2.resize(prev_frame, size, interpolation = cv2.INTER_AREA)
start_time = time.time()

while(cap.isOpened()):
	ret, frame = cap.read()
	total=total+1
	
	if frame is not None:
		#frame = cv2.resize(frame,size, interpolation = cv2.INTER_AREA)
        	diff = cv2.absdiff(frame, prev_frame)
		
	else:
		break;
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count > p_frame_thresh:
		key=key+1
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces1 = face_cascade1.detectMultiScale(gray, 1.5, 3)
		faces4 = face_cascade4.detectMultiScale(gray, 1.5, 3)
		#faces5 = face_cascade5.detectMultiScale(gray, 1.5, 3)
		for (x,y,w,h) in faces1:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			#newpath= os.path.basename(str(key)+'.jpg')
			#cv2.imwrite(newpath,roi_gray)
		for (x,y,w,h) in faces4:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
		    
		cv2.imshow('gray',frame)
			
        prev_frame = frame
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
print key
print total

print("--- %s seconds ---" % (time.time() - start_time))
cap.release()
cv2.destroyAllWindows()




'''		for (x,y,w,h) in faces4:
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		    roi_gray = gray[y:y+h, x:x+w]
		    roi_color = frame[y:y+h, x:x+w]
		for (x,y,w,h) in faces5:
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
		    roi_gray = gray[y:y+h, x:x+w]
		    roi_color = frame[y:y+h, x:x+w]'''
