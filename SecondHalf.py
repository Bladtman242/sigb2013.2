__author__ = 'Roed, Ghurt, Stoffi'
import calibrationExample as calibrate
import numpy as np
from SIGBToolsForSecond import *
import cv2

global K, dist_coefs, I1, I1Corners, box, cam1

I1 = cv2.imread("pattern.png")

I1Gray = cv2.cvtColor(I1,cv2.COLOR_RGB2GRAY)
K, dist_coefs = calibrate.loadMatrixes()


def cube_points(c,wid):
    """ Creates a list of points for plotting a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    #vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T
  # (x3,y3) = (corners[45][0][0],corners[45][0][1])
  #   (x4,y4) = (corners[53][0][0],corners[53][0][1])

def getCornerCoords(boardImg):

    """
    :param boardImg:
    :return: cornercoords and centercoord
    """
    patternSize = (9,6)
    found,corners=cv2.findChessboardCorners(boardImg, patternSize)
    if(found):
        (x1,y1) = (corners[0][0][0],corners[0][0][1])
        (x2,y2) = (corners[7][0][0],corners[7][0][1])
        (x3,y3) = (corners[45][0][0],corners[45][0][1])
        (x4,y4) = (corners[53][0][0],corners[53][0][1])
        return [(x4,y4),(x3,y3),(x1,y1),(x2,y2)]

def AugmentImage(img):

    I1 = cv2.imread("pattern.png")
    I2 = img
    I1Gray = cv2.cvtColor(I1,cv2.COLOR_RGB2GRAY)
    I2Gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    I1Corners = getCornerCoords(I1Gray)
    I2Corners = getCornerCoords(I2Gray)

    H = estimateHomography(I1Corners, I2Corners)

    K, dist_coefs = calibrate.loadMatrixes()

    box = cube_points((0,0,0),0.3)

    #This is K * [I | O]
    cam1 = Camera(hstack((K,dot(K,array([[0],[0],[-1]])) )) )

    box_cam1 = cam1.project(toHomogenious(box))

    # compute second camera matrix from cam1.
    cam2 = Camera(dot(H,cam1.P))

    #isolates the [Rotation | translation] (extrinsic parameters)
    A = dot(linalg.inv(K),cam2.P[:,:3])
    #estimate z axis from cross product.
    A = array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1],axis=0)]).T

    #add K again
    cam2.P[:,:3] = np.dot(K,A[0])

    # project with the second camera
    box_cam2 = np.array(cam2.project(toHomogenious(box)))



    cv2.imshow()
    imshow(I2)
    plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
    show()

I1Corners = getCornerCoords(I1Gray)
box = cube_points((0,0,0),0.3)
cam1 = Camera(hstack((K,dot(K,array([[0],[0],[-1]])) )) )

def AugmentFrame(img):
    globals()
    I2 = img
    I2Gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    I2Corners = getCornerCoords(I2Gray)
    if(I2Corners == None):
        return
    H = estimateHomography(I1Corners, I2Corners)




    #This is K * [I | O]


    # compute second camera matrix from cam1.
    cam2 = Camera(dot(H,cam1.P))

    #isolates the [Rotation | translation] (extrinsic parameters)
    A = dot(linalg.inv(K),cam2.P[:,:3])
    #estimate z axis from cross product.
    A = array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1],axis=0)]).T

    #add K again
    cam2.P[:,:3] = np.dot(K,A[0])

    # project with the second camera
    box_cam2 = np.array(cam2.project(toHomogenious(box)))

    #Bottom square
    cv2.line(I2,(int(box_cam2[0][0]),int(box_cam2[1][0])),(int(box_cam2[0][1]),int(box_cam2[1][1])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][1]),int(box_cam2[1][1])),(int(box_cam2[0][2]),int(box_cam2[1][2])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][3]),int(box_cam2[1][3])),(int(box_cam2[0][4]),int(box_cam2[1][4])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][3]),int(box_cam2[1][3])),(int(box_cam2[0][2]),int(box_cam2[1][2])),(255,255,0))

   #sides
    cv2.line(I2,(int(box_cam2[0][5]),int(box_cam2[1][5])),(int(box_cam2[0][0]),int(box_cam2[1][0])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][6]),int(box_cam2[1][6])),(int(box_cam2[0][1]),int(box_cam2[1][1])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][7]),int(box_cam2[1][7])),(int(box_cam2[0][2]),int(box_cam2[1][2])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][8]),int(box_cam2[1][8])),(int(box_cam2[0][3]),int(box_cam2[1][3])),(255,255,0))

    #top square
    cv2.line(I2,(int(box_cam2[0][5]),int(box_cam2[1][5])),(int(box_cam2[0][6]),int(box_cam2[1][6])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][6]),int(box_cam2[1][6])),(int(box_cam2[0][7]),int(box_cam2[1][7])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][7]),int(box_cam2[1][7])),(int(box_cam2[0][8]),int(box_cam2[1][8])),(255,255,0))
    cv2.line(I2,(int(box_cam2[0][8]),int(box_cam2[1][8])),(int(box_cam2[0][5]),int(box_cam2[1][5])),(255,255,0))

    # return I2



# I2 = cv2.imread("CalibrationImage2.jpg")
# I3 = cv2.imread("CalibrationImage3.jpg")
# I4 = cv2.imread("CalibrationImage4.jpg")
# I5 = cv2.imread("CalibrationImage5.jpg")
# I6 = cv2.imread("CalibrationImage6.jpg")

# AugmentFrame(I2)
# AugmentImage(I3)
# AugmentImage(I4)
# AugmentImage(I6)

def liveBox():
    globals()
    capture = cv2.VideoCapture(0)
    running = True
    while running:
        running, img =capture.read()
        imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ch = cv2.waitKey(1)
        if(ch==27) or (ch==ord('q')): #ESC
            running = False
        img=cv2.undistort(img, K, dist_coefs )
        found,corners=cv2.findChessboardCorners(imgGray, (9,6)  )
        if (found!=0):
            AugmentFrame(img)
        cv2.imshow("Calibrated",img)



liveBox()