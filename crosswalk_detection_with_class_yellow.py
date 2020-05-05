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
import glob
import os
from Utils.candidates import Candidate
from Utils.consistencySet import ConsistencySet

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#==========================#
#---------functions--------#
#==========================#

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

def colorDetection(image):
	hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

	'''yellow'''
	# Range for upper range
	# MAGIC_NUMBER for yellow range
	yellow_lower = np.array([20, 50, 50])
	yellow_upper = np.array([60, 255, 255])
	mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

	return mask_yellow

# Main
# Used for debugging the modules
def process(img_src_filename, DEBUG = False):

	# Read the images
	img_color = cv.imread(img_src_filename, cv.IMREAD_UNCHANGED)
	img = cv.imread(img_src_filename, 0)

	# IF THE DATASET HAS A GOOD WHITE BALANCE
	mask_yellow = colorDetection(img_color)
	img = cv.bitwise_and(img, img, mask=mask_yellow)

	# Remove strange border artifacts
	img_color = img[5:-5, 5:-5]
	img = img[5:-5, 5:-5]

	# Apply a small blur to the image
	img_blur = cv.medianBlur(img, 5)

	# Check size of the images
	HEIGHT, WIDTH = img_blur.shape
	if WIDTH > 680:
		return

	# The block size parameters is base on the size of the given dataset
	# A neural network based on smaller resolution could be useful for the future (cf: first papers)
	# MAGIC_NUMBER for threshold
	th3 = cv.adaptiveThreshold(
		img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 157, -20)

	if DEBUG:
		cv.imshow('threshold', th2)
		cv.waitKey()

	# Calculate the size_max and size_min of a candidates
	# MAGIC_NUMBER for size of the candidates
	size_max = 0.2 * HEIGHT * WIDTH
	size_min = 0.5 * 10 ** -3 * HEIGHT * WIDTH

	# Find the contours in the image
	contours, hierarchy = cv.findContours(
		th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	# loop over the contours and check for potential candidates
	CANDIDATES = []
	mask = np.ones(th3.shape[:2], dtype="uint8") * 255
	for c in contours:
		if is_contour_bad(c, HEIGHT, WIDTH, img_color):
			# FOR DEBUG
			cv.drawContours(mask, [c], -1, 0, -1)
		else:
			CANDIDATES.append(Candidate(c))

	mask = cv.bitwise_and(th3, th3, mask=mask)

	# Debug
	for c in CANDIDATES:
		c.drawDebugCandidates(img_color)

	# ==============
	# CONSITENCY !!!
	# ==============
	CONSITENCY_SET = list()
	for si in CANDIDATES:
		for sj in CANDIDATES:

			SETIJ = ConsistencySet(si, sj)

			# Don't use same initial set
			if sj == si:
				continue

			# One consistency set
			coefficients = SETIJ.coefficients

			y_1 = int(coefficients[0] * 0 + coefficients[1])
			y_2 = int(coefficients[0] * WIDTH + coefficients[1])

			# Draw line for debug
			tadam = cv.imread(img_src_filename,
							  cv.IMREAD_UNCHANGED)
			cv.line(tadam, (0, y_1), (WIDTH, y_2), (0, 0, 0), 1)

			for sk in CANDIDATES:

				# Eliminate length / width greater than 0.5
				if sk.ellipse[1][0]/sk.ellipse[1][1] > 0.5:
					continue

				EL = 0.01 * WIDTH
				# sk distance from line of si sj
				d = sk.getDistance(si, sj)
				if d > EL:
					continue

				# sk parralel
				if abs(SETIJ.angle - (sk.orientation % 90)) < 25 or abs(SETIJ.angle - (sk.orientation % 90)) > 155:
					continue

				# sk intersect
				hx_start = sk.centerNp[0] - sk.ellipse[1][1] / 2
				hy_start = sk.centerNp[1]
				hx_end = sk.centerNp[0] + sk.ellipse[1][1] / 2
				hy_end = sk.centerNp[1]

				h_start = (hx_start, hy_start)
				h_end = (hx_end, hy_end)

				h_start = (rotate(sk.centerNp, h_start,
								  np.deg2rad(sk.orientation+90)))
				h_end = (rotate(sk.centerNp, h_end,
								np.deg2rad(sk.orientation+90)))

				lij = [0, y_1, WIDTH, y_2]
				seg_k_no = [h_start[0], h_start[1], h_end[0], h_end[1]]
				seg_k_yes = [h_start[0], h_start[1], h_end[0], h_end[1]]

				if len(Segments().intersection(lij, seg_k_no)[1]) == 0:
					continue

				# ... sk pass all
				SETIJ.append(sk)

			# Orientation
			SETIJ.candidates = [c for c in SETIJ if c.orientation < SETIJ.mean_orientation +
					 10 and c.orientation > SETIJ.mean_orientation - 10]

			CONSITENCY_SET.append(SETIJ)

	# Remove set which are too small
	CONSITENCY_SET = [s for s in CONSITENCY_SET if len(s.candidates) >= 3]

	if DEBUG:
		for set_c in CONSITENCY_SET:
			img_debug = cv.imread(img_src_filename, cv.IMREAD_UNCHANGED)
			set_c.drawDebugSet(img_debug)
			cv.imshow('Debug', img_debug)
			cv.waitKey()
	
	# If there is not any remaining set, assume that there is no crosswalk
	if len(CONSITENCY_SET) == 0:
		return None

	# Find the consistency_set with the best consistency value
	FINAL_SET = CONSITENCY_SET[0]
	for s in CONSITENCY_SET:
		if s.consistency_val < FINAL_SET.consistency_val:
			FINAL_SET = s

	tadam = cv.imread(img_src_filename, cv.IMREAD_UNCHANGED)

	blank_image = np.zeros((HEIGHT, WIDTH), np.uint8)
	for c in FINAL_SET.candidates:
		cv.ellipse(tadam, c.ellipse, (255, 255, 255), -1)
		cv.ellipse(blank_image, c.ellipse, (255, 255, 255), -1)
	cv.ellipse(blank_image, FINAL_SET.si.ellipse, (255, 255, 255), -1)
	cv.ellipse(blank_image, FINAL_SET.sj.ellipse, (255, 255, 255), -1)

	# BOUNDING BOX
	contours, hierarchy = cv.findContours(
		blank_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	boxes = []
	for c in contours:
		(x, y, w, h) = cv.boundingRect(c)
		boxes.append([x, y, x+w, y+h])

	boxes = np.asarray(boxes)
	left = np.min(boxes[:, 0])
	top = np.min(boxes[:, 1])
	right = np.max(boxes[:, 2])
	bottom = np.max(boxes[:, 3])

	cv.rectangle(blank_image, (left, top), (right, bottom), (255, 0, 0), 2)

	# DEBUG
	if DEBUG:
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

	return ((left,top), (right,bottom))


if __name__ == "__main__":

	filenames = list()

	for filename in glob.iglob('ownSet/' + '**/*.jpg', recursive=True):
		filenames.append(filename)

	for filename in filenames:
		print(process(filename, True))
