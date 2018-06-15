import numpy as np
import cv2
import math

#https://bbs.archlinux.org/viewtopic.php?id=202682
#https://unix.stackexchange.com/questions/412214/how-to-manage-the-white-balance-in-webcam
#v4l2-ctl -c exposure_auto_priority=0

cap = cv2.VideoCapture(0)
bg_captured = False
hsv_captured = False
kernel = np.ones((3, 3), np.uint8)

while(1):
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	cv2.rectangle(frame,(0,0),(350,350),(0,255,0),2)
	cv2.imshow('original',frame)
	gray = cv2.cvtColor(frame[0:350, 0:350], cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame[0:350, 0:350], cv2.COLOR_BGR2HSV)

	if bg_captured:
		diff = cv2.absdiff(backimg,gray)
		cv2.imshow('diff', diff)

		retval, threshold = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
		threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

		opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		img_dilation = cv2.dilate(closing, kernel, iterations=1)
		cv2.imshow('img_dilation', img_dilation)
		cv2.imshow('threshold', threshold)

		bitwise = cv2.bitwise_and(frame[0:350, 0:350], frame[0:350, 0:350], mask=threshold)
		cv2.imshow('bitwise', bitwise)
		bitwisehsv = cv2.bitwise_and(hsv[0:350, 0:350], hsv[0:350, 0:350], mask=threshold)
		cv2.imshow('bitwisehsv', bitwisehsv)

	k = cv2.waitKey(20) & 0xff
	if k == 27:
		break
	if k == ord('b'):
		backimg = cv2.cvtColor(frame[0:350, 0:350], cv2.COLOR_BGR2GRAY)
		bg_captured = True


cap.release()
cv2.destroyAllWindows()
