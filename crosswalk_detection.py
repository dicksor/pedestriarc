'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing

 Papers:    https://www.researchgate.net/publication/320675020_Crosswalk_navigation_for_people_with_visual_impairments_on_a_wearable_device
			https://ieeexplore.ieee.org/document/7873114
			https://pdfs.semanticscholar.org/19d9/becce00a500bdd1cad5a19f9f16175347096.pdf
			http://stephense.com/research/papers/mva03.pdf
'''

#==========================#
#----------import----------#
#==========================#

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random as rng
from sklearn import linear_model, datasets
import statistics
import functools
import math
import random

#==========================#
#---------functions--------#
#==========================#


def random_color():
	rgbl = [255, 0, 0]
	random.shuffle(rgbl)
	return tuple(rgbl)


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


def dot(vA, vB):
	return vA[0]*vB[0]+vA[1]*vB[1]


def ang(lineA, lineB):
	# Get nicer vector form
	vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
	vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
	# Get dot prod
	dot_prod = dot(vA, vB)
	# Get magnitudes
	magA = dot(vA, vA)**0.5
	magB = dot(vB, vB)**0.5
	# Get cosine value
	cos_ = dot_prod/magA/magB
	# Get angle in radians and then convert to degrees
	angle = math.acos(dot_prod/magB/magA)
	# Basically doing angle <- angle mod 360
	ang_deg = math.degrees(angle) % 360

	if ang_deg-180 >= 0:
		# As in if statement
		return 360 - ang_deg
	else:

		return ang_deg


def rotate_point(point, angle):
	pass


def is_contour_bad(c, HEIGHT, WIDTH, img_color):
	size_max = 0.2 * HEIGHT * WIDTH
	size_min = 0.5 * 10 ** -3 * HEIGHT * WIDTH

	if len(c) > 5:
		ellipse = cv.fitEllipse(c)
		area = ellipse[1][0] * ellipse[1][1]
		if ellipse[2] < 20:
			return True
		if area > size_max or area < size_min:
			return True
		if not cv.pointPolygonTest(c, ellipse[0], False):
			return True
	else:
		return True


def draw_debug_candidates(c, img_color):
	ellipse = cv.fitEllipse(c)
	area = ellipse[1][0] * ellipse[1][1]
	cv.ellipse(img_color, ellipse, (128, 0, 128))
	cv.circle(img_color, tuple(np.intp(ellipse[0])), 1, (128, 128, 0), 8)
	minRect = cv.minAreaRect(c)
	box = cv.boxPoints(minRect)
	box = np.intp(box)
	x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
	xCenter = (x1 + x2)/2
	yCenter = (y1 + y2)/2
	cv.drawContours(img_color, [box], -1, (0, 0, 255))
	cv.circle(img_color, (int(xCenter), int(yCenter)), 1, (128, 0, 128), 8)

	return ellipse


class Segments:

	def intersection(self, s1, s2):
		left = max(min(s1[0], s1[2]), min(s2[0], s2[2]))
		right = min(max(s1[0], s1[2]), max(s2[0], s2[2]))
		top = max(min(s1[1], s1[3]), min(s2[1], s2[3]))
		bottom = min(max(s1[1], s1[3]), max(s2[1], s2[3]))

		if top > bottom or left > right:
			return ('NO INTERSECTION', list())
		if (top, left) == (bottom, right):
			return ('POINT INTERSECTION', list((left, top)))
		return ('SEGMENT INTERSECTION', list((left, bottom, right, top)))


def consistency_calculs(_data):
	C = np.array([[0.05, 0.3, 0.3, 1]])
	data = np.array([_data])

	#Transposition for calculus
	C = np.transpose(C)
	return np.dot(data, C)[0][0]




# Main
# Used for debugging the modules
def process(img_src_filename):

	img_color = cv.imread(img_src_filename, cv.IMREAD_UNCHANGED)
	img = cv.imread(img_src_filename, 0)

	# Remove strange border artifacts
	img_color = img[5:-5, 5:-5]
	img = img[5:-5, 5:-5]

	img_blur = cv.medianBlur(img, 5)

	HEIGHT, WIDTH = img_blur.shape
	if WIDTH > 680:
		return

	# The block size parameters is base on the size of the given dataset
	# A neural network based on smaller resolution could be useful for the future (cf: first papers)
	th2 = cv.adaptiveThreshold(
		img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 157, -10)
	th3 = cv.adaptiveThreshold(
		img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 157, -10)

	# Finding the potential candidates
	size_max = 0.2 * HEIGHT * WIDTH
	size_min = 0.5 * 10 ** -3 * HEIGHT * WIDTH

	# 3. find contours and  draw the green lines on the white strips
	contours, hierarchy = cv.findContours(
		th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	mask = np.ones(th3.shape[:2], dtype="uint8") * 255

	# loop over the contours
	CONTOURS = []

	for c in contours:
		if is_contour_bad(c, HEIGHT, WIDTH, img_color):
			# FOR DEBUG
			cv.drawContours(mask, [c], -1, 0, -1)
		else:
			CONTOURS.append(c)

	mask = cv.bitwise_and(th3, th3, mask=mask)

	CANDIDATES = []
	for c in CONTOURS:
		CANDIDATES.append(draw_debug_candidates(c, img_color))

	# ==============
	# CONSITENCY !!!
	# ==============
	CONSITENCY_SET = list()
	for si in CANDIDATES:
		for sj in CANDIDATES:

			SETIJ = list()
			ORIENTTATIONIJ = list()
			DISTANCEIJ = list()
			WIDTHIJ = list()
			LENGTHIJ = list()
			COLORIJ = list()

			# Don't use same initial set
			if sj == si:
				continue

			# One consistency set
			center_si, center_sj = np.asarray(si[0]), np.asarray(sj[0])
			coefficients = np.polyfit(
				(center_si[0], center_sj[0]), (center_si[1], center_sj[1]), 1)
			y_1 = int(coefficients[0] * 0 + coefficients[1])
			y_2 = int(coefficients[0] * WIDTH + coefficients[1])

			tadam = cv.imread(img_src_filename,
							  cv.IMREAD_UNCHANGED)
			cv.line(tadam, (0, y_1), (WIDTH, y_2), (0, 0, 0), 1)
			for sk in CANDIDATES:
				tadam2 = tadam.copy()
				if sj == sk or si == sk:
					continue

				if sk[1][0]/sk[1][1] > 0.5:
					continue

				EL = 0.1 * WIDTH
				# sk distance from line of si sj
				center_sk = np.asarray(sk[0])
				d = np.linalg.norm(np.cross(
					center_sj-center_si, center_si-center_sk))/np.linalg.norm(center_sj-center_si)
				if d > EL:
					continue

				# sk parralel
				angle = np.rad2deg(np.arctan2(
					center_sj[1] - center_si[1], center_sj[0] - center_si[0]))
				if abs(angle - sk[2]) < 15:
					continue

				# sk intersect
				hx_start = center_sk[0] - sk[1][1] / 2
				hy_start = center_sk[1]
				hx_end = center_sk[0] + sk[1][1] / 2
				hy_end = center_sk[1]

				h_start = (hx_start, hy_start)
				h_end = (hx_end, hy_end)

				h_start = (rotate(center_sk, h_start, np.deg2rad(sk[2]+90)))
				h_end = (rotate(center_sk, h_end, np.deg2rad(sk[2]+90)))

				lij = [0, y_1, WIDTH, y_2]
				seg_k = [h_start[0], h_start[1], h_end[0], h_end[1]]

				cv.line(tadam2, tuple(np.intp(h_start)),
						tuple(np.intp(h_end)), (0, 0, 0), 1)

				# if len(Segments().intersection(lij, seg_k)[1]) == 0:
				#	continue

				# ... sk pass all
				SK_WITH_CALCULATED_VALUES = [sk, d]
				SETIJ.append(SK_WITH_CALCULATED_VALUES)
				ORIENTTATIONIJ.append(sk[2])
				DISTANCEIJ.append(d)
				WIDTHIJ.append(sk[1][0])
				LENGTHIJ.append(sk[1][1])

			# Orientation
			mean_orientation = np.mean(ORIENTTATIONIJ)
			mean_distance = np.mean(DISTANCEIJ)
			mean_width = np.mean(WIDTHIJ)
			mean_length = np.mean(LENGTHIJ)

			SETIJ = [c for c in SETIJ if c[0][2] < mean_orientation +
					 10 and c[0][2] > mean_orientation - 10]

			SETIJ_WITH_CALCULATED_VALUES = [
				SETIJ, mean_orientation, mean_distance, mean_width, mean_length]

			CONSITENCY_SET.append(SETIJ_WITH_CALCULATED_VALUES)

	# Remove set which are too small
	CONSITENCY_SET = [s for s in CONSITENCY_SET if len(s[0]) >= 3]

	# Calculate the variance for each set
	VARIANCES_SET = list()
	for idx, sc in enumerate(CONSITENCY_SET):
		D1 = 0
		D2 = 0
		D3 = 0
		D4 = 0

		mean_orientation = sc[1]
		mean_distance = sc[2]
		mean_width = sc[3]
		mean_length = sc[4]

		D1 = functools.reduce(
			lambda a, b: a+b, [(s[1] - mean_distance) ** 2 for s in sc[0]]) / len(sc)
		D2 = functools.reduce(
			lambda a, b: a+b, [(s[0][1][0] - mean_width) ** 2 for s in sc[0]]) / len(sc)
		D3 = functools.reduce(
			lambda a, b: a+b, [(s[0][1][1] - mean_length) ** 2 for s in sc[0]]) / len(sc)
		D4 = functools.reduce(
			lambda a, b: a+b, [(s[0][2] - mean_orientation) ** 2 for s in sc[0]]) / len(sc)
		VARIANCES_SET.append([D1, D2, D3, D4, idx])

	#FINAL_SET = [s for s in VARIANCES_SET if s[3] < 5]
	if len(VARIANCES_SET) == 0:
		print('no crosswalk')
		return None
		#exit()

	FINAL_SET = VARIANCES_SET[0]
	for idx, s in enumerate(VARIANCES_SET):
		if consistency_calculs(FINAL_SET[:4]) < consistency_calculs(s[:4]):
			FINAL_SET = s

	for idx, s in enumerate(CONSITENCY_SET):
		if all(elem in s[0] for elem in CONSITENCY_SET[FINAL_SET[4]][0]):
			if len(s[0]) > len(CONSITENCY_SET[FINAL_SET[4]][0]):
				pass
				print('HEHE')

	tadam = cv.imread(img_src_filename, cv.IMREAD_UNCHANGED)

	'''
	print(CONSITENCY_SET)
	for c in CONSITENCY_SET:
		tadam2 = tadam.copy()
		color = random_color()
		print(f'c: {c}')
		for c2 in c[0]:
			print(f'c2 {c2}')
			cv.ellipse(tadam2, c2[0], color, 2)
		cv.imshow("Test", tadam2)
		cv.waitKey()
	'''

	blank_image = np.zeros((HEIGHT,WIDTH), np.uint8)
	for c in CONSITENCY_SET[FINAL_SET[4]][0]:
		cv.ellipse(tadam, c[0], (255, 255, 255), -1)
		cv.ellipse(blank_image, c[0], (255, 255, 255), -1)

	# BOUNDING BOX
	contours, hierarchy = cv.findContours(blank_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	boxes = []
	for c in contours:
		(x, y, w, h) = cv.boundingRect(c)
		boxes.append([x,y, x+w,y+h])

	boxes = np.asarray(boxes)
	left = np.min(boxes[:,0])
	top = np.min(boxes[:,1])
	right = np.max(boxes[:,2])
	bottom = np.max(boxes[:,3])

	cv.rectangle(blank_image, (left,top), (right,bottom), (255, 0, 0), 2)

	#return ((left,top), (right,bottom))

	#boundingBox = cv.minAreaRect(allPoint)
	#cv.drawContours(tadam, boundingBox, (255, 255, 255), random.randint(0, 5))

		#cv.imshow("Test", tadam)

	'''
	if len(FINAL_SET) > 0:
		print(FINAL_SET)
		print(len(CONSITENCY_SET[FINAL_SET[1][4]][0]))

		tadam = cv.imread('tests/cam_20200112_145100.jpg', cv.IMREAD_UNCHANGED)
		for c in CONSITENCY_SET[FINAL_SET[0][4]][0]:
			print(c[0])
			cv.ellipse(tadam, c[0], (0,0,0), 2)
			cv.imshow("Test", tadam)
	'''

	# DEBUG
	
	titles = ['Original Image', 'Adaptive Mean Thresholding',
			  'Adaptive Gaussian Thresholding', 'After Mask']
	images = [img_color, th2, th3, blank_image]

	for i in range(4):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	contours, hierarchy = cv.findContours(
		mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	

	#cv.drawContours(img_color, contours, -1, (255, 0, 0))
	#cv.imshow("Resul Candidates", img_color)
	#cv.waitKey()

import os
import glob

if __name__ == "__main__":

	filenames = list()

	for filename in glob.iglob('tests/' + '**/*.jpg', recursive=True):
		filenames.append(filename)

	for filename in filenames:
		print(process(filename))