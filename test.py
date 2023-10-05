# import libraries of python OpenCV
import cv2

haar_cascade = 'cars.xml'
	
cap = cv2.VideoCapture(1)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# reads frames from a video
ret, frames = cap.read()
		
# convert frames to gray scale
gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
		
# Detects cars of different sizes in the input image
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# To draw a rectangle in each cars
for (x,y,w,h) in cars:
    cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
	
# Display frames in a window
cv2.imshow('video', frames)

# loop runs if capturing has been initialized.
while True:
	# reads frames from a video
	ret, frames = cap.read()
		
	# convert to gray scale of each frames
	gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
		
	
	# Detects cars of different sizes in the input image
	cars = car_cascade.detectMultiScale(gray, 1.1, 1)
		
	# To draw a rectangle in each cars
	for (x,y,w,h) in cars:
		cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
	
	# Display frames in a window
	cv2.imshow('video', frames)
		
	# Wait for Esc key to stop
	if cv2.waitKey(33) == 27:
		break
	
# De-allocate any associated memory usage
cv2.destroyAllWindows()


