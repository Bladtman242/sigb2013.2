__author__ = 'Roed, Ghurt, Stoffi'
import calibrationExample as calibrate
import numpy as np
from SIGBToolsForSecond import Camera as camera
import SIGBToolsForSecond as SIGBTools
import cv2


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

def AugmentImages():

    K, dist_coefs = calibrate.loadMatrixes()
    I1 = cv2.imread("CalibrationImage1.jpg")
    I1Gray = cv2.cvtColor(I1,cv2.COLOR_RGB2GRAY)
    I2 = cv2.imread("CalibrationImage2.jpg")
    I2Gray = cv2.cvtColor(I2,cv2.COLOR_RGB2GRAY)
    I3 = cv2.imread("CalibrationImage3.jpg")
    I3Gray = cv2.cvtColor(I3,cv2.COLOR_RGB2GRAY)
    I4 = cv2.imread("CalibrationImage4.jpg")
    I4Gray = cv2.cvtColor(I4,cv2.COLOR_RGB2GRAY)
    I5 = cv2.imread("CalibrationImage5.jpg")
    I5Gray = cv2.cvtColor(I5,cv2.COLOR_RGB2GRAY)

    patternSize = (9,6)

    found,corners=cv2.findChessboardCorners(I1Gray, patternSize)
    (x1,y1) = (corners[0][0][0],corners[0][0][1])
    (x2,y2) = (corners[8][0][0],corners[8][0][1])
    (x3,y3) = (corners[45][0][0],corners[45][0][1])
    (x3,y4) = (corners[53][0][0],corners[53][0][1])
    print corners[0]
    if (found!=0):
        # cv2.drawChessboardCorners(I1, patternSize, corners,found)
        cv2.circle(I1,(corners[45][0][0],corners[45][0][1]),5,(255,0,0))
        cv2.imshow("Calibrated",I1)
        cv2.waitKey(0)

calibrate.calibrationExample()
# AugmentImages()

box = cube_points()

# project bottom square in first image
cam1 = camera.Camera(np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )) )

# first points are the bottom square
box_cam1 = cam1.project(SIGBTools.toHomogenious(box[:,:5]))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(np.dot(H,cam1.P))
A = np.dot(np.linalg.inv(K),cam2.P[:,:3])
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = np.dot(K,A)
# project with the second camera
box_cam2 = cam2.project(SIGBTools.toHomogenious(box))


