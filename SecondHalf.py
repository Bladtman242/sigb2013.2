__author__ = 'Roed, Ghurt, Stoffi'
import calibrationExample as calibrate
import numpy as np
from SIGBToolsForSecond import *
import cv2
import math


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


def getCornerCoords(boardImg):

    """
    :param boardImg:
    :return: cornercoords and centercoord
    """
    patternSize = (9,6)
    found,corners=cv2.findChessboardCorners(boardImg, patternSize)
    (x1,y1) = (corners[0][0][0],corners[0][0][1])
    (x2,y2) = (corners[8][0][0],corners[8][0][1])
    (x3,y3) = (corners[45][0][0],corners[45][0][1])
    (x4,y4) = (corners[53][0][0],corners[53][0][1])
    return [(x4,y4),(x3,y3),(x1,y1),(x2,y2)]

def AugmentImages():

    I1 = cv2.imread("pattern.png")
    I1Gray = cv2.cvtColor(I1,cv2.COLOR_RGB2GRAY)
    I2 = cv2.imread("CalibrationImage7.jpg")
    I2Gray = cv2.cvtColor(I2,cv2.COLOR_RGB2GRAY)
    # I3 = cv2.imread("CalibrationImage3.jpg")
    # I3Gray = cv2.cvtColor(I3,cv2.COLOR_RGB2GRAY)
    # I4 = cv2.imread("CalibrationImage4.jpg")
    # I4Gray = cv2.cvtColor(I4,cv2.COLOR_RGB2GRAY)
    # I5 = cv2.imread("CalibrationImage5.jpg")
    # I5Gray = cv2.cvtColor(I5,cv2.COLOR_RGB2GRAY)

    I1Corners = getCornerCoords(I1Gray)
    I2Corners = getCornerCoords(I2Gray)

    H = estimateHomography(I1Corners, I2Corners)

    K, dist_coefs = calibrate.loadMatrixes()

    box = cube_points((0,0,0),0.3)

    # project bottom square in first image

    #This is K * [I | O]
    cam1 = Camera(hstack((K,dot(K,array([[0],[0],[-1]])) )) )

    box_cam1 = cam1.project(toHomogenious(box))

    #  projection of first box
    figure()
    imshow(I1)
    plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
    show()

    # Vi laver en homography mellem de to "flader" (skakbraeder) og ud fra den estimerer vi en projektion.
    # homografien er kun i 2d, hvor en projektion er i 3d.
    # Ud fra homografien estimerer vi en projektion (der har en ekstra kulonne til z-aksen).
    # Vi isolerer
    # compute second camera matrix from cam1 and H the homography is just multiplied on the matrix
    cam2 = Camera(dot(H,cam1.P))


    A = dot(linalg.inv(K),cam2.P[:,:3]) #isolates the [Rotation | translation] matrix to change it

    A = array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1],axis=0)]).T
    
    cam2.P[:,:3] = np.dot(K,A[0])

    # project with the second camera
    box_cam2 = np.array(cam2.project(toHomogenious(box)))



    figure()
    imshow(I2)
    plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
    show()

AugmentImages()



