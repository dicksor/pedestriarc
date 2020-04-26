'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing

 Papers:    https://www.researchgate.net/publication/320675020_Crosswalk_navigation_for_people_with_visual_impairments_on_a_wearable_device
			https://ieeexplore.ieee.org/document/7873114
			https://pdfs.semanticscholar.org/19d9/becce00a500bdd1cad5a19f9f16175347096.pdf
'''

#==========================#
#----------import----------#
#==========================#

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random as rng
from sklearn import linear_model, datasets

#==========================#
#---------functions--------#
#==========================#


def imgThreshold(img):
	pass


def process(img):
	pass


def lineCalc(vx, vy, x0, y0):
	scale = 10
	x1 = x0+scale*vx
	y1 = y0+scale*vy
	m = (y1-y0)/(x1-x0)
	b = y1-m*x1
	return m, b

# the angle at the vanishing point


def angle(pt1, pt2):
	x1, y1 = pt1
	x2, y2 = pt2
	inner_product = x1*x2 + y1*y2
	len1 = math.hypot(x1, y1)
	len2 = math.hypot(x2, y2)
	print(len1)
	print(len2)
	a = math.acos(inner_product/(len1*len2))
	return a*180/math.pi

# vanishing point - cramer's rule


def lineIntersect(m1, b1, m2, b2):
	# a1*x+b1*y=c1
	# a2*x+b2*y=c2
	# convert to cramer's system
	a_1 = -m1
	b_1 = 1
	c_1 = b1

	a_2 = -m2
	b_2 = 1
	c_2 = b2

	d = a_1*b_2 - a_2*b_1  # determinant
	dx = c_1*b_2 - c_2*b_1
	dy = a_1*c_2 - a_2*c_1

	intersectionX = dx/d
	intersectionY = dy/d
	return intersectionX, intersectionY

def is_contour_bad(c, HEIGHT, WIDTH):
	size_max = 0.2 * HEIGHT * WIDTH
	size_min = 0.5 * 10 ** -3 * HEIGHT * WIDTH

	if len(c) > 5:
		ellipse = cv.fitEllipse(c)
		area = ellipse[1][0] * ellipse[1][1]
		print(ellipse)
		if ellipse[2] < 20:
			return True
		if area > size_max or area < size_min:
			return True
	else:
		return True

# Main
# Used for debugging the modules
if __name__ == "__main__":

	radius = 500 #px

	bxLeft = []
	byLeft = []
	bxbyLeftArray = []
	bxbyRightArray = []
	bxRight = []
	byRight = []
	boundedLeft = []
	boundedRight = []

	img_color = cv.imread('tests/cam_20200112_145100.jpg', cv.IMREAD_UNCHANGED)
	img = cv.imread('tests/cam_20200112_145100.jpg', 0)
	img_blur = cv.medianBlur(img, 5)

	HEIGHT, WIDTH = img_blur.shape

	# The block size parameters is base on the size of the given dataset
	# A neural network based on smaller resolution could be useful for the future (cf: first papers)
	th2 = cv.adaptiveThreshold(
		img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 257, 2)
	th3 = cv.adaptiveThreshold(
		img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 257, 2)

	# Finding the potential candidates
	CANDIDATES = []

	size_max = 0.2 * HEIGHT * WIDTH
	size_min = 0.5 * 10 ** -3 * HEIGHT * WIDTH
	print(f'sizeMax: {size_max}, sizeMin: {size_min}')

	# 3. find contours and  draw the green lines on the white strips
	contours, hierarchy = cv.findContours(
		th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	mask = np.ones(th3.shape[:2], dtype="uint8") * 255

	print(f'ContoursNB: {len(contours)}')

	# loop over the contours
	for c in contours:
		if is_contour_bad(c, HEIGHT, WIDTH):
			cv.drawContours(mask, [c], -1, 0, -1)

	cv.imshow("MASK", mask)
	mask = cv.bitwise_and(th3, th3, mask=mask)

	titles = ['Original Image', 'Adaptive Mean Thresholding',
			  'Adaptive Gaussian Thresholding', 'After Mask']
	images = [img_color, th2, th3, mask]

	for i in range(4):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	contours, hierarchy = cv.findContours(
		mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	cv.drawContours(img_color, contours, -1, (255, 0, 0))

	for i in contours:
		if len(i) > 5:
			ellipse = cv.fitEllipse(i)
			cv.ellipse(img_color, ellipse, (0,255,0))
			minRect = cv.minAreaRect(i)
			box = cv.boxPoints(minRect)
			box = np.intp(box)
			cv.drawContours(img_color, [box], -1, (0,0,255))

			bx_1, by_1, bx_2, by_2 = box[0][0], box[0][1], box[2][0], box[2][1]
			#cv.rectangle(img_color,(bx,by),(bx+bw,by+bh),(0,255,0),2)

			cv.line(img_color, (box[0][0], box[0][1]), (box[2][0], box[2][1]),
					(0, 255, 0), 2)  # draw the a contour line
			bxRight.append(bx_2)  # right line
			byRight.append(by_2)  # right line
			bxLeft.append(bx_1)  # left line
			byLeft.append(by_1)  # left line
			bxbyLeftArray.append([bx_1, by_1])  # x,y for the left line
			bxbyRightArray.append([bx_2, by_2])  # x,y for the left line
			cv.circle(img_color, (int(bx_1), int(by_1)), 5,
					(0, 250, 250), 2)  # circles -> left line
			cv.circle(img_color, (int(bx_2), int(by_2)), 5,
					(250, 0, 0), 2)  # circles -> right line

	cv.imshow("TEST", img_color)
	cv.waitKey()

	# calculate median average for each line
	medianR = np.median(bxbyRightArray, axis=0)
	medianL = np.median(bxbyLeftArray, axis=0)

	bxbyLeftArray = np.asarray(bxbyLeftArray)
	bxbyRightArray = np.asarray(bxbyRightArray)

	# 4. are the points bounded within the median circle?
	for i in bxbyLeftArray:
		if (((medianL[0] - i[0])**2 + (medianL[1] - i[1])**2) < radius**2) == True:
			boundedLeft.append(i)

	boundedLeft = np.asarray(boundedLeft)

	for i in bxbyRightArray:
		if (((medianR[0] - i[0])**2 + (medianR[1] - i[1])**2) < radius**2) == True:
			boundedRight.append(i)

	boundedRight = np.asarray(boundedRight)

	# 5. RANSAC Algorithm

	# select the points enclosed within the circle (from the last part)
	bxLeft = np.asarray(boundedLeft[:, 0])
	byLeft = np.asarray(boundedLeft[:, 1])
	bxRight = np.asarray(boundedRight[:, 0])
	byRight = np.asarray(boundedRight[:, 1])

	# transpose x of the right and the left line
	bxLeftT = np.array([bxLeft]).reshape(-1, 1)
	bxRightT = np.array([bxRight]).reshape(-1, 1)

	# run ransac for LEFT
	model_ransac = linear_model.RANSACRegressor()
	ransacX = model_ransac.fit(bxLeftT, byLeft)
	inlier_maskL = model_ransac.inlier_mask_  # right mask

	# run ransac for RIGHT
	ransacY = model_ransac.fit(bxRightT, byRight)
	inlier_maskR = model_ransac.inlier_mask_  # left mask

	# draw RANSAC selected circles
	for i, element in enumerate(boundedRight[inlier_maskR]):
	   # print(i,element[0])
		# circles -> right line
		cv.circle(img_color, (element[0], element[1]), 10, (250, 100, 100), 2)

	for i, element in enumerate(boundedLeft[inlier_maskL]):
	   # print(i,element[0])
		# circles -> right line
		cv.circle(img_color, (element[0], element[1]), 10, (100, 100, 250), 2)


	# 6. Calcuate the intersection point of the bounding lines
	# unit vector + a point on each line
	vx, vy, x0, y0 = cv.fitLine(boundedLeft[inlier_maskL],cv.DIST_L2,0,0.01,0.01) 
	vx_R, vy_R, x0_R, y0_R = cv.fitLine(boundedRight[inlier_maskR],cv.DIST_L2,0,0.01,0.01)

	# get m*x+b
	m_L,b_L=lineCalc(vx, vy, x0, y0)
	m_R,b_R=lineCalc(vx_R, vy_R, x0_R, y0_R)

	# calculate intersention 
	intersectionX,intersectionY = lineIntersect(m_R,b_R,m_L,b_L)

	# 7. draw the bounding lines and the intersection point
	m = radius*10 
	if (intersectionY < HEIGHT/2 ):
		cv.circle(img_color,(int(intersectionX),int(intersectionY)),10,(0,0,255),15)
		cv.line(img_color,(x0-m*vx, y0-m*vy), (x0+m*vx, y0+m*vy),(255,0,0),3)
		cv.line(img_color,(x0_R-m*vx_R, y0_R-m*vy_R), (x0_R+m*vx_R, y0_R+m*vy_R),(255,0,0),3)

	# cv.drawContours(img, contours, -1, (255, 255, 0))
	# RANSAC
	# Add outlier data
	plt.show()

	titles = ['Original Image', 'Adaptive Mean Thresholding',
			  'Adaptive Gaussian Thresholding']
	images = [img_color, th2, th3]

	for i in range(3):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()
