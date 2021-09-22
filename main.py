import cv2
import numpy as np


from object_detector import *


#Load Aruco Detector

parameters=cv2.aruco.DetectorParameters_create()
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50) # 10cm-10cm aruco marker

#Load Object Detector
detector = HomogeneousBgDetector()

#Load Image

#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("output", 3024,4032)
img=cv2.imread("recodusuk3.jpg")




#Get Aruco Marker

corners,_,_=cv2.aruco.detectMarkers(img,aruco_dict,parameters=parameters)

#Draw polygon around the marker
int_corners=np.int0(corners)  #köşe noktaların koordinatları tamsayı olmak zorunda
cv2.polylines(img,int_corners,True,(0,255,0),5) ##aruco marker etrafına yeşil çizgi çiziyoruz

#Aruco Perimeter (aruco marker çevre uzunluğu)
aruco_perimeter=cv2.arcLength(corners[0],True)

#Pixel to cm ratio (pixel sayısını cm ye çeviren oran)
pixel_cm_ratio=aruco_perimeter/20 # aruco marker 10cm olduğu için çevre toplamı 40cm eder

contours=detector.detect_objects(img)

#draw objects boundaries
for cnt in contours:
    #Get rect
    rect=cv2.minAreaRect(cnt)#nesnenin etrafını çevreleyen minimum diktörtgen alanı koordinatlarını, genişlik ve uzunluğunu ve açısını verir
    (x,y),(w,h),angle=rect

    #Get width and Height of the objects by applying the Ratio pixxel to cm
    object_width=w/pixel_cm_ratio
    object_height=h/pixel_cm_ratio

    #Display rectangle
    box=cv2.boxPoints(rect)
    box=np.int0(box) # köşe nokta koordinatları tam sayı olmak zorunda

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1) #algılanan nesnelerin merkez noktasında çizilen çemberin içini (-1 değeri ile ) girilen renkle (kırmızı) doldurrur
    cv2.polylines(img, [box], True, (255, 0, 0), 2) #algılanan nesnelerin etrafında mavi renkli diktörtgen çizer
    cv2.putText(img,"Genislik {} cm".format(round(object_width,1)),(int(x-100),int(y-20)),cv2.FONT_HERSHEY_PLAIN,2,(100,200,0),2)
    cv2.putText(img,"Uzunluk {} cm".format(round(object_height,1)),(int(x-100),int(y+15)),cv2.FONT_HERSHEY_PLAIN,2,(100,200,0),2)

cv2.imshow("output",img)


cv2.waitKey(0)


