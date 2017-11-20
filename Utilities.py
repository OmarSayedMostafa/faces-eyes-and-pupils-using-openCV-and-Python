import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
# multiple cascades:
# https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def cv_display(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

def distance_from_cennter(x, y, centerX, centerY):

    return math.sqrt(math.pow(centerX-x,2)+math.pow(centerY-y,2))

def calculate_pixel_density(image, circle):

    #circle = (x, y, r)
    sum = 0

    for i in range(0, eh):
        for j in range(0, ew):
            if(distance_from_cennter(i,j,circle[0],circle[1]) <= circle[2]):
                #print(img_arr[i,j])
                sum+=img_arr[i,j]      
    return sum               

def ReadVideo (FileName):
    Frames=[]
    vidcap = cv2.VideoCapture(FileName)
    success,image = vidcap.read()
    success = True
    while success:
        success,image = vidcap.read()
        if success == False:
            break
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Frames.append(image)
    
    vidcap.release()
    return Frames

def extract_face_eye_pupil(img):
    #convert BGR image to gray scale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #get faces location using haarcascade 
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    #region
    #print(len(faces))

    for (x,y,w,h) in faces:
        #draw a blue rectangle arround face
        #cv2.circle(img,(x,y), 20, (0, 0, 255), -1)
        #cv2.circle(img,(int(x+w/2),y), 20, (0, 0, 255), -1)
        #cv2.circle(img,(x,int(y+h/2)), 20, (0, 0, 255), -1)

        cv2.rectangle(img,(x,y),(x + w,y + h),(255,0,0),2)
        #---------------------------------------------------------------------
        #slicing (cropping) the face from whole image
        roi_gray = gray_image[y:y + h, x:x + w] #gray to process on
        roi_color = img[y:y + h, x:x + w]       #colored to drow on
        #---------------------------------------------------------------------
        #get eyes location 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #print(len(eyes))
        #print (w, h)
        counter = 0
        face_center = x+(w/2),y+(h/2)
        for (ex,ey,ew,eh) in eyes:
            #counter +=1
            #if(counter >2):
            #    break;
            #handling more than 2 eyes in the face
            #print(y + ey , (y+h/2))
            if(y + ey > int(y+h/2)):
                print("wrong eye detected")
                continue   
            if(y+ey+int(eh/2)>int(y+h/2)):
                print("wrong eye detected")
                continue
            if(int(ew+eh)*2<150):         
                print("wrong eye detected")
                continue  
            
            
            #drawing rectangle arround eye
            cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey + eh),(0,255,0),2)
            #slicing (cropping) the eye from face slice           
            croped_colored_img = roi_color[ey:ey+eh, ex:ex+ew] #[y: y + h, x: x + w] 
            croped_gray_img = roi_gray[ey:ey+eh, ex:ex+ew] 
            #cv_display(croped_colored_img)
 
               
            #optional 2 algorithms to detect eye pupil , hint by contour algo. is more accurate and faster
            cv2.circle(croped_colored_img,  pupil_by_contour(croped_gray_img), 10, (0, 0, 255), -1)
            #cv2.circle(croped_colored_img,  pupil_by_houghcircle(croped_gray_img, ew, eh), 10, (0, 255, 0), -1)
            #cv_display(croped_gray_img)

    #endregion
            
    #cv_display(img)
    
    cv2.destroyAllWindows()
    return img

def pupil_by_contour(image):

     
    gray = image   
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #convert image to binary image (black & white) to get the pupil and other black objects only 
    retval, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    #remove any object and let only the most arrounded / circular object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

    #get contours (indvidual object in image "edges")
    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    drawing = np.copy(image)

    #if there more than one contour ( the pupil and the frame of the image ) so we need to check the one which its area is smaller so by
    #devide the contour area of pupil over the frame area (bounding box) it should get less than 1 so we pick this contour and reject others if exits
    for contour in contours:
        area = cv2.contourArea(contour)
        bounding_box = cv2.boundingRect(contour)
    
        extend = area / (bounding_box[2] * bounding_box[3])
    
        # reject the contours with big extend
        if extend > 7:
            continue
    
        # calculate countour center and draw a dot there in it's center
        m = cv2.moments(contour)
        if m['m00'] != 0:
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            return center

def pupil_by_houghcircle(image, ew, eh):
    
    croped_gray_img = image
    #equalize the image histogram
    cv2.equalizeHist(croped_gray_img,croped_gray_img)

    #_, croped_gray_img = cv2.threshold(croped_gray_img, 90, 255, cv2.THRESH_BINARY);
    #cv_display(croped_gray_img)
    
    #get all hough circles in the image
    circles = cv2.HoughCircles(croped_gray_img,cv2.HOUGH_GRADIENT, 1,int(ew/12), 250, 12, int(ew/10), int(eh/8))
    #croped_gray_img: The input image
    #cv2.HOUGH_GRADIENT: method: Method to be applied
    #1:dp: Inverse ratio of the accumulator resolution
    #int(ew/12) :minDist: Minimal distance between the center of one circle and another
    #250 : threshold: Threshold of the edge detector
    #12 :minArea: What’s the min area of a circle in the image?
    #int(ew/10) : minRadius: What’s the min radius of a circle in the image?
    # int(eh/8) : maxRadius: What’s the max radius of a circle in the image?



    #next step is to loop over all circles and find the most circle contains black or near to black pixels to make sure it contains the pupil
    # that can be calculated by get the most circle gives the least pixels summation
    minimum_sum = 99999999999
    sum = 0
    pupil_circul = None

    #convert img to numpay array to inhance the time (same complixity) " but there is an issue with looping on image pixels in python may take 7~10 sec to iterate on small image"
    img_arr = np.array(croped_gray_img)
            
    circles = np.uint16(np.around(circles))
    for c in circles[0,:]:             
        counter = 0
        for i in range(0, eh):
            for j in range(0, ew):
                if(distance_from_cennter(i,j,c[0],c[1]) < (c[2]/2)):
                    sum+=img_arr[i,j]
                    counter +=1
        if((sum/counter)<minimum_sum):
            minimum_sum = sum/counter
            pupil_circul = c
                 
        sum = 0
                                     
    #draw the outer circle
    center = (pupil_circul[0],pupil_circul[1])
    return center        

def live_cam():
    vidcap = cv2.VideoCapture(0)
    while vidcap.read():
        success,frame = vidcap.read()
        if success == False:
            break
        cv2.imshow("cam", live_facial_detection(frame))
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()
    vidcap.release()     

def live_facial_detection(img):
    #convert BGR image to gray scale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #get faces location using haarcascade 
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)



    for (x,y,w,h) in faces:
        #draw a blue rectangle arround face
        cv2.rectangle(img,(x,y),(x + w,y + h),(255,0,0),2)
        #---------------------------------------------------------------------
        #slicing (cropping) the face from whole image
        roi_gray = gray_image[y:y + h, x:x + w] #gray to process on
        roi_color = img[y:y + h, x:x + w]       #colored to drow on
        #---------------------------------------------------------------------
        #get eyes location 
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            
           
            #handling more than 2 eyes in the face
            print(y + ey , (y+h/2))
            if(y + ey > int(y+h/2)):
                print("wrong eye detected")
                continue   
            if(y+ey+int(eh/2)>int(y+h/2)):
                print("wrong eye detected")
                continue
            if(int(ew+eh)*2<150):         
                print("wrong eye detected")
                continue     
            
            
            #drawing rectangle arround eye
            cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey + eh),(0,255,0),2)
            #slicing (cropping) the eye from face slice           
            croped_colored_img = roi_color[ey:ey+eh, ex:ex+ew] #[y: y + h, x: x + w] 
            croped_gray_img = roi_gray[ey:ey+eh, ex:ex+ew] 
            #cv_display(croped_colored_img)
 
               
            #optional 2 algorithms to detect eye pupil , hint by contour algo. is more accurate and faster
            cv2.circle(croped_colored_img,  pupil_by_contour(croped_gray_img), 8, (0, 255, 0), -1)
            #cv2.circle(croped_colored_img,  pupil_by_houghcircle(croped_gray_img, ew, eh), 10, (0, 255, 0), -1)
            #cv_display(croped_gray_img)

    #endregion
    return img



def special_function_to_get_the_multiply_of_10_frame_numbers(FileName):
    counter = 0
    vidcap = cv2.VideoCapture(FileName)
    success,image = vidcap.read()
    success = True
    while success:
        success,Frame = vidcap.read()
        if success == False:
            break
        if(counter%10 ==0):
            extract_face_eye_pupil(Frame)
        counter += 1
    
    vidcap.release()


def saving_video_with_facial_detection(FileName):
    
    vidcap = cv2.VideoCapture(FileName)
    success,Frame = vidcap.read()
    success = True

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (Frame.shape[0],Frame.shape[1]))
    #out = cv2.VideoWriter('output.avi', -1, 20.0, (Frame.shape[0],Frame.shape[1]))
    #out=cv2.VideoWriter('video.avi',-1,1,(Frame.shape[0],Frame.shape[1]))

    counter = 0;
    while success:
        success,Frame = vidcap.read()
        if success == False:
            break
        
        print(counter)
        counter += 1
        if(counter%10==0):

            Frame = extract_face_eye_pupil(Frame)


            cv2.imwrite('sample output/'+str(counter)+'.jpg',Frame)
        #if(cv2.waitKey(1) & 0xFF == ord('q')):
        #    break
        
        
    #out.release()


    
    vidcap.release()

