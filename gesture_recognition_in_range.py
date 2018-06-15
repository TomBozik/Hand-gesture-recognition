import cv2
import numpy as np
import math

kernel = np.ones((3, 3), np.uint8)
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
getROI = False
bounds_created = False
img_for_ROI = None
img_for_ROI_captured = False
refPt = []

threshold = 15
blurValue = 25

def click_and_crop(event, x, y, flags, param):
	global x_start, y_start, x_end, y_end, cropping, getROI

	if event == cv2.EVENT_LBUTTONDOWN:
		x_start, y_start, x_end, y_end = x, y, x, y
		cropping = True

	elif event == cv2.EVENT_MOUSEMOVE:
		if cropping == True:
			x_end, y_end = x, y

	elif event == cv2.EVENT_LBUTTONUP:
		x_end, y_end = x, y
		cropping = False
		getROI = True


cap = cv2.VideoCapture(0)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
	ret, image = cap.read()
	image=cv2.flip(image,1)
	image_to_thresh = image.copy()
	hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

	if not getROI:
		if not cropping:
			cv2.imshow("image", image)
		elif cropping:
			if not img_for_ROI_captured:
				img_for_ROI = image.copy()
				img_for_ROI_captured = True

			i = img_for_ROI.copy()
			cv2.rectangle(i , (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
			cv2.imshow("image", i)
	else:
		if not bounds_created:
			refPt = [(x_start, y_start), (x_end, y_end)]
			roi = img_for_ROI[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			lower = np.array([hsvRoi[:,:,0].min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min()])
			upper = np.array([hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()])

			cv2.imshow("image", image)
			bounds_created = True

		else:
			mask = cv2.inRange(hsv, lower, upper)
			mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
			blur = cv2.GaussianBlur(mask, (blurValue, blurValue), 0)
			ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

			#contours
			thresh1 = thresh.copy()
			_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			length = len(contours)
			maxArea = -1
			if length > 0:
				for i in range(length):
					temp = contours[i]
					area = cv2.contourArea(temp)
					if area > maxArea:
						maxArea = area
						ci = i

				res = contours[ci]
				hull = cv2.convexHull(res)
				drawing = np.zeros(image.shape, np.uint8)
				cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
				cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

				cv2.drawContours(image, [res], 0, (0, 255, 0), 2)
				cv2.drawContours(image, [hull], 0, (0, 0, 255), 3)

				drawing = np.zeros(image.shape, np.uint8)
				cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
				cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

			#convex hull
			hull = cv2.convexHull(res, returnPoints=False)
		    #deffects
			defects = cv2.convexityDefects(res, hull)
			count_defects = 0

			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(res[s][0])
				end = tuple(res[e][0])
				far = tuple(res[f][0])

				a = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
				b = math.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
				c = math.sqrt((end[0]-far[0])**2 + (end[1]-far[1])**2)

		        #cosine rule
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

		        #ignore angles > 90
				if angle <= 90:
					count_defects += 1
					cv2.circle(drawing, far, 6, [0,0,255], -1)

			if count_defects == 1:
				cv2.putText(image,"This is 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			elif count_defects == 2:
				cv2.putText(image,"This is 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			elif count_defects == 3:
				cv2.putText(image,"This is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			elif count_defects == 4:
				cv2.putText(image,"This is 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

			cv2.imshow("image", image)
			cv2.imshow('thresh', thresh)
			cv2.imshow('drawing', drawing)

	k = cv2.waitKey(20) & 0xff
	if k == ord('q'):
		break
	if k == ord('r'):
		img_for_ROI_captured = False
		bounds_created = False
		getROI = False
		lower = None
		upper = None

cap.release()
cv2.destroyAllWindows()
