import cv2
import imutils
import numpy as np

image = cv2.imread("store_signs/000266.jpg")
image = cv2.resize(image, (250, 250))
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh1 = cv2.threshold(imagegray, 160, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 5, 2)
# thresh = cv2.threshold(imagegray, 157, 255, cv2.THRESH_TOZERO)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
# for cnt in cnts:
c = max(cnts, key=cv2.contourArea)
cv2.drawContours(image, c, -1, (0,255,0), 3)
cv2.imshow("000001", image)
cv2.imshow("000002", thresh)
cv2.imshow("000003", thresh1)
cv2.waitKey()
#
# for (i, c) in enumerate(cnts):
#     (x, y, w, h) = cv2.boundingRect(c)
#     roi = cv2.resize(thresh[y:y + h, x:x + w], (250, 250))
#     moments = cv2.HuMoments(cv2.moments(roi)).flatten()
#
# print ("MOMENTS #{}: {}".format(i + 1, moments))
# cv2.imshow("ROI #{}".format(i + 1), roi)
# cv2.waitKey(0)