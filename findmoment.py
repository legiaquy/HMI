import cv2


def moment(image):
    image = cv2.resize(image, (250, 250))
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 5, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    c = max(cnts, key=cv2.contourArea)
    # cv2.drawContours(image, c, -1, (0, 255, 0), 3)
    (x, y, w, h) = cv2.boundingRect(c)
    roi = cv2.resize(thresh[y:y + h, x:x + w], (50, 50))
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    return moments
