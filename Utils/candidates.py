'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing
'''

import cv2 as cv
import numpy as np
import math

def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return qx, qy

class Candidate:
    '''
    A class used to represent a potential candidate

    ...

    Attributes
    ----------
    contour : contour
        the contour representing the candidates (len must be greater than 5)
    '''
    def __init__(self, contour):
        self.contour = contour
        self.ellipse = cv.fitEllipse(self.contour)
        self.fitRectangle = cv.minAreaRect(self.contour)

        self.center = self.ellipse[0]
        self.orientation = self.ellipse[2]
        self.centerNp = np.asarray(self.ellipse[0])

    def getDistance(self, si, sj):
        return np.linalg.norm(np.cross(
                    sj.centerNp-si.centerNp, si.centerNp-self.centerNp))/np.linalg.norm(sj.centerNp-si.centerNp)
    
    def drawDebugCandidates(self, img):
        cv.ellipse(img, self.ellipse, (128, 0, 128))
        cv.circle(img, tuple(np.intp(self.ellipse[0])), 1, (128, 128, 0), 8)
        box = cv.boxPoints(self.fitRectangle)
        box = np.intp(box)
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        xCenter = (x1 + x2)/2
        yCenter = (y1 + y2)/2
        cv.drawContours(img, [box], -1, (0, 0, 255))
        cv.circle(img, (int(xCenter), int(yCenter)), 1, (128, 0, 128), 8)

        hx_start = self.centerNp[0] - self.ellipse[1][1] / 2
        hy_start = self.centerNp[1]
        hx_end = self.centerNp[0] + self.ellipse[1][1] / 2
        hy_end = self.centerNp[1]

        h_start = (hx_start, hy_start)
        h_end = (hx_end, hy_end)

        h_start = (rotate(self.centerNp, h_start,
                            np.deg2rad(self.orientation+90)))
        h_end = (rotate(self.centerNp, h_end,
                        np.deg2rad(self.orientation+90)))

        cv.line(img, tuple(np.intp(h_start)), tuple(np.intp(h_end)), (255, 0, 255), 1)

        hx_start = self.centerNp[0] - self.ellipse[1][0] / 2
        hy_start = self.centerNp[1]
        hx_end = self.centerNp[0] + self.ellipse[1][0] / 2
        hy_end = self.centerNp[1]

        h_start = (hx_start, hy_start)
        h_end = (hx_end, hy_end)

        h_start = (rotate(self.centerNp, h_start,
                            np.deg2rad(self.orientation)))
        h_end = (rotate(self.centerNp, h_end,
                        np.deg2rad(self.orientation)))

        cv.line(img, tuple(np.intp(h_start)), tuple(np.intp(h_end)), (255, 255, 0), 1)

        cv.putText(img, f'{self.ellipse[2]}',(int(xCenter), int(yCenter)),cv.FONT_HERSHEY_PLAIN,1,(0,0,255))

    def __str__(self):
        return str(self.ellipse)