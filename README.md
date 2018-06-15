# Hand-gesture-recognition
This school project shows hand gesture recognition.  
Python + opencv

+ gesture_recognition_in_range.py
  - User select part of hand
  - Convert selected part to HSV
  - Get minimum and maximum values for every HSV channel
  - Use these values in function inRange()
  - Apply morphological transformations: opening, closing
  - Use contours, convexHull and convexityDeffects for finger counting

![alt text](https://raw.githubusercontent.com/TomBozik/Hand-gesture-recognition/master/images/inrange_start.png)
![alt text](https://raw.githubusercontent.com/TomBozik/Hand-gesture-recognition/master/images/inrang.png)

+ hand_detection_absdiff.py
  - Capture background without hand
  - Convert background to gray
  - Convert every frame to gray
  - Use absdiff() function to substarct background
  - Threshold (value 20 have best results)
  - Morphological transformations: opening, closing, dilation

  ![alt text](https://raw.githubusercontent.com/TomBozik/Hand-gesture-recognition/master/images/absdiff_start.png)
  ![alt text](https://raw.githubusercontent.com/TomBozik/Hand-gesture-recognition/master/images/absdiff.png)
