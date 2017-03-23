import numpy as np
import cv2
from matplotlib import pyplot as plt


min_area = 100*100
bowl_hmin = 75
bowl_hmax = 111
bowl_smin = 22
bowl_smax = 93
bowl_vmin = 121
bowl_vmax = 255
bowl_HSV_min = np.array([bowl_hmin, bowl_smin, bowl_vmin])
bowl_HSV_max = np.array([bowl_hmax,bowl_smax,bowl_vmax])

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('mask_image',cv2.WINDOW_NORMAL)

while(True):
    # Capture frame-by-frame
    ret, image_bgr = cap.read()
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bowl_mask = cv2.inRange(image_hsv, bowl_HSV_min, bowl_HSV_max)
    bowl = [image_bgr]

    bowl_kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((10,10) , np.uint8)
    bowl_opening = cv2.morphologyEx(bowl_mask.copy(), cv2.MORPH_OPEN, bowl_kernel)
    de_bowl = cv2.dilate(bowl_opening.copy(), kernel2, iterations=1)
    bowl_contours, bowl_hierarchy = cv2.findContours(de_bowl.copy(), 1, 2)


    for i in range(len(bowl_contours)):
        moments_bowl = cv2.moments(bowl_contours[i])
        bowl_cx = int(moments_bowl['m10'] / (moments_bowl['m00']))
        bowl_cy = int(moments_bowl['m01'] / (moments_bowl['m00']))
        bowl_area = moments_bowl['m00']
        cnt = bowl_contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_bgr, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)

        if bowl_area > min_area:
            cv2.circle(image_bgr, (bowl_cx, bowl_cy), 3, (255, 0, 0), -1)
            cv2.putText(image_bgr, (str(bowl_cx) + " , " + str(bowl_cy)), (bowl_cx - 40, bowl_cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image_bgr, "bowl", (bowl_cx - 20, bowl_cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.drawContours(image_bgr, bowl_contours, -1, (0, 133, 133), 2)
            bowl.append(image_bgr[y-10 : y+h+10 , x-10 : x+w+10 ])


    cv2.imshow('frame',image_bgr)
    cv2.imshow('mask_image',de_bowl)

    for i in range(len(bowl)):
        cv2.imshow('bowl%d' %(i),bowl[i])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()