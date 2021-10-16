import numpy as np 
import cv2

#Haar cascade is a pre-trained classifier that already knows how to find something in an img/video.
#Very similar to CNNs, except they are hand-traing(tough!) not trained through a deep learning algorithm.
#Haar cascade is faster, but less accurate. CNNs are slower, but more accurate.
cap = cv2.VideoCapture(0)
#"cv2.data.haarcascades" is the path to where it is stored on our system after important cv2. "frontalface_default.xml" is the cascade we want to use. 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
	#Detects the location of the faces in coordinates. 
	#"1.3" is the scale factor of how much our template scanner shrinks each iteration, since the cascade likely wasn't trained on images of exactly our size.
	#"5" is how many candidate rectangles in close proximity together must exist before the program recognizes it as actually the image. Higher the more accurate.
	faces = face_cascade.detectMultiScale(gray, 1.3, 3)
	#faces returns (x, y) coordinates of the top left of the rectangle as well as the (w, h) dimensions of the rectangles, therefore:
	for (x, y, w, h) in faces: #loops because there could be multiple faces
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
		for (ex, ey, ew, eh) in eyes: #loops for case of multiple eyes
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1) #The reason why this modifies the original frame is because it is still a reference to the original frame, not a .copy().

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()