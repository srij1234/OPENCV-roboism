
import cv2 as cv
import numpy as np
import imutils   #using this library for bounded rotation of image
''' rotate_image()function for non bounded rotation '''
def rotate_image(image, angle):#function for rotation (not bounded)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #centre at h/2,w/2 of a rectangle
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    #image matrix rotated
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    #storing in variable
    return result
'''this function does not change the dimensions of the image'''
img=cv.imread('CVtask.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
num1=0
num2=0
num3=0
num4=0
'''these numbers are defined to prevent a specific if statement from repeating itself'''

'''explained later'''
top = np.zeros((img.shape[0],img.shape[1],3),np.uint8)*255
contours1=[]
for i in contours:
    contours1.append(i)
contours=contours1
for c in contours: 
    approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c,True), True)
    if (len(approx)==4 and cv.contourArea(c)>50):#using contour are to remove noises
        rect=cv.minAreaRect(c)
for c in contours: 
    approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c,True), True)
    if (len(approx)==4 and cv.contourArea(c)>50):#using contour are to remove noises
        rect=cv.minAreaRect(c)
        '''rect stores coordinates of centre of rectangle having minimum area
        which inscribes the contour,hight width,angle at which the rectangle is rotated,these will
        be used in the code afterwards'''
        t=rect[1][1]/rect[1][0]
        if (t>0.9 and t<1.1):
            x=int(rect[0][0])
            y=int(rect[0][1])
            #all squares are detected
            first=img[y,x][0]==79 and img[y,x][1]==209 and img[y,x][2]==146 and num1==0
            second=img[y,x][0]==9 and img[y,x][1]==127 and img[y,x][2]==240 and num2==0
            third=img[y,x][0]==210 and img[y,x][1]==222 and img[y,x][2]==228 and num3==0
            if((first or second or third) and num4==0):
                contours.append(c)
            elif(img[y,x][0]==79 and img[y,x][1]==209 and img[y,x][2]==146 and num1==0 and num4==1):
                '''img[y,x]returns bgr code at the pixel on comparing color
                at centre of square and
                required colour we run following code'''
                num1=num1+1
                wt = cv.imread('1.jpg')
                '''i used online tool which we can find on chev.com to get the id of the aruco markers'''
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                #contours
                '''now we have to align the aruco marker with horizontal'''
                rect1=cv.minAreaRect(cc[0])#only one element in cc
                #minarearect function returns angle at index 2
                wt1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                '''storing area for future reference
                (particularly identification of outer edge of aruco marker)'''
                gray1 = cv.cvtColor(wt1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                '''finding contours of aligned aruco marker'''
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                ''' finding the contour which fits the aruco marker precisely ,
                the absolute value of difference in area would be minimum,
                difference would not be 0 because contourarea function returns approximate area
                while the area calculated by us is exact'''
                inde=l.index(min(l))
                k1=cc1[inde]
                #storing the contour for further use
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                wt1=wt1[yoff1:y_end1,xoff1:x_end1]
                #cropping the aligned aruco marker
                wt1 = cv.resize(wt1,(int(rect[1][0]),int(rect[1][1])))
                #rotating aruco marker for putting in main image and blank image
                tran = np.ones((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                #creating blank image for removal of black areas present at corners
                tran=imutils.rotate_bound(tran, rect[2])
                #rotating imagefor puting on main image 
                
                
                '''the problem that arises here is that the inside of the aruco markers which we will put
                will remain of the color of the box example- green ,orange, black,pink-peach
                to convert this to white i created another image "top"  which is black and of the 
                size of main image,we will put all the aruco markers on this black image  aswell and at last 
                we will take bitwise or which will put white colour at the center of aruco markers which we will 
                place later'''
                tran2 = np.zeros((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                wt1=imutils.rotate_bound(wt1, rect[2])
                # similar to tran but for image "top" 
               
                wt2=cv.bitwise_or(tran2,wt1)
                wt1=cv.bitwise_or(tran,wt1)
                xoff=int(rect[0][0] - wt1.shape[0]/2)
                yoff=int(rect[0][1] - wt1.shape[0]/2)
                x_end = int(rect[0][0] + wt1.shape[1]/2)
                y_end = int(rect[0][1] + wt1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                # here we finally place the aruco markers on main image and image top
                temp=cv.bitwise_not(wt1)
                temp2=cv.bitwise_not(new)
                new=cv.bitwise_or(temp,temp2)
                new=cv.bitwise_not(new)
                fin=top[yoff:y_end,xoff:x_end]
                wt2=cv.bitwise_or(wt2,fin)
                top[yoff:y_end,xoff:x_end]=wt2
                img[yoff:y_end,xoff:x_end] = new
                '''the following if statements are similar to 
                the if statement which we have discussed
                but for remaining colours and aruco markers'''
            elif(img[y,x][0]==9 and img[y,x][1]==127 and img[y,x][2]==240 and num2==0 and num4==1):
                num2=num2+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('2.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                rect1=cv.minAreaRect(cc[0])
                wt1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                gray1 = cv.cvtColor(wt1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                wt1=wt1[yoff1:y_end1,xoff1:x_end1]
                wt1 = cv.resize(wt1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                wt1=imutils.rotate_bound(wt1, rect[2])
                wt2=cv.bitwise_or(tran2,wt1)
                wt1=cv.bitwise_or(tran,wt1)
                xoff=int(rect[0][0] - wt1.shape[0]/2)
                yoff=int(rect[0][1] - wt1.shape[0]/2)
                x_end = int(rect[0][0] + wt1.shape[1]/2)
                y_end = int(rect[0][1] + wt1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(wt1)
                temp2=cv.bitwise_not(new)
                new=cv.bitwise_or(temp,temp2)
                new=cv.bitwise_not(new)
                fin=top[yoff:y_end,xoff:x_end]
                wt2=cv.bitwise_or(wt2,fin)
                top[yoff:y_end,xoff:x_end]=wt2
                img[yoff:y_end,xoff:x_end] = new
            elif(img[y,x][0]==210 and img[y,x][1]==222 and img[y,x][2]==228 and num3==0 and num4==1):
                num3=num3+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('4.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                rect1=cv.minAreaRect(cc[0])
                wt1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                gray1 = cv.cvtColor(wt1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                wt1=wt1[yoff1:y_end1,xoff1:x_end1]
                wt1 = cv.resize(wt1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                wt1=imutils.rotate_bound(wt1, rect[2])
                wt2=cv.bitwise_or(tran2,wt1)
                wt1=cv.bitwise_or(tran,wt1)
                xoff=int(rect[0][0] - wt1.shape[0]/2)
                yoff=int(rect[0][1] - wt1.shape[0]/2)
                x_end = int(rect[0][0] + wt1.shape[1]/2)
                y_end = int(rect[0][1] + wt1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(wt1)
                temp2=cv.bitwise_not(new)
                new=cv.bitwise_or(temp,temp2)
                new=cv.bitwise_not(new)
                fin=top[yoff:y_end,xoff:x_end]
                wt2=cv.bitwise_or(wt2,fin)
                top[yoff:y_end,xoff:x_end]=wt2
                img[yoff:y_end,xoff:x_end] = new
            elif(img[y,x][0]<1 and img[y,x][1]<1 and img[y,x][2]<1 and num4==0):
                num4=num4+1
                xoff,yoff,w,h = cv.boundingRect(c)
                wt = cv.imread('3.jpg')
                gray = cv.cvtColor(wt, cv.COLOR_BGR2GRAY)
                edged = cv.Canny(gray, 30, 200)
                cc, hh = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                rect1=cv.minAreaRect(cc[0])
                wt1=rotate_image(wt, rect1[2])
                area1=rect1[1][1]*rect1[1][0]
                gray1 = cv.cvtColor(wt1, cv.COLOR_BGR2GRAY)
                edged1 = cv.Canny(gray1, 30, 200)
                cc1, hh1 = cv.findContours(edged1,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                l=[]
                for k in range(len(cc1)):
                    l.append(abs(cv.contourArea(cc1[k])-area1))
                inde=l.index(min(l))
                k1=cc1[inde]
                xoff1,yoff1,w1,h1 = cv.boundingRect(k1)
                x_end1 = int(xoff1 + w1)
                y_end1= int(yoff1 + h1)
                wt1=wt1[yoff1:y_end1,xoff1:x_end1]
                wt1 = cv.resize(wt1,(int(rect[1][0]),int(rect[1][1])))
                tran = np.ones((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran=imutils.rotate_bound(tran, rect[2])
                tran2 = np.zeros((int(rect[1][1]),int(rect[1][0]),3),np.uint8)*255
                tran2=imutils.rotate_bound(tran2, rect[2])
                tran=cv.bitwise_not(tran)
                wt1=imutils.rotate_bound(wt1, rect[2])
                wt2=cv.bitwise_or(tran2,wt1)
                wt1=cv.bitwise_or(tran,wt1)
                xoff=int(rect[0][0] - wt1.shape[0]/2)
                yoff=int(rect[0][1] - wt1.shape[0]/2)
                x_end = int(rect[0][0] + wt1.shape[1]/2)
                y_end = int(rect[0][1] + wt1.shape[0]/2)
                new=img[yoff:y_end,xoff:x_end]
                temp=cv.bitwise_not(wt1)
                temp2=cv.bitwise_not(new)
                new=cv.bitwise_or(temp,temp2)
                new=cv.bitwise_not(new)
                fin=top[yoff:y_end,xoff:x_end]
                wt2=cv.bitwise_or(wt2,fin)
                top[yoff:y_end,xoff:x_end]=wt2
                img[yoff:y_end,xoff:x_end] = new
            
img=cv.bitwise_or(img,top)
#this generates the final image with white colour inside all aruco markers
#img=cv.resize(img,(1000,500))
#cv.imshow('j kk',top)
cv.imshow('sdfsdfabc',img)
cv.imwrite("final.png",img)

cv.waitKey(0)