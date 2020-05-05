'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing
'''

import cv2 as cv
import numpy as np
import functools

def consistency_calculs(_data):
	C = np.array([[0.1, 0.05, 0.05, 1]])
	data = np.array([_data])

	# Transposition for calculus
	C = np.transpose(C)
	return np.dot(data, C)[0][0]

class ConsistencySet:
	'''
	A class used to represent a consistency set

	...

	Attributes
	----------
	si : Candidates
		first candidates representing the consistency of the set
	sj : Candidates
		second candidates representing the consistency of the set
	'''
	def __init__(self, si, sj):
		self.si = si
		self.sj = sj
		self.candidates = list()

		self.coefficients = np.polyfit(
				(si.centerNp[0], sj.centerNp[0]), (si.centerNp[1], sj.centerNp[1]), 1)
		self.angle = np.rad2deg(np.arctan2(
					sj.centerNp[1] - si.centerNp[1], sj.centerNp[0] - si.centerNp[0]))

		self.mean_orientation = 0
		self.mean_distance = 0
		self.mean_width = 0
		self.mean_length = 0

		self.variance_orientation = 0
		self.variance_distance = 0
		self.variance_width = 0
		self.variance_length = 0

		self.consistency_val = 0

	def calculateValue(self):
		# Means
		self.mean_orientation = np.mean([c.orientation for c in self.candidates])
		self.mean_distance = np.mean([c.getDistance(self.si, self.sj) for c in self.candidates])
		self.mean_width = np.mean([c.ellipse[1][0] for c in self.candidates])
		self.mean_length = np.mean([c.ellipse[1][1] for c in self.candidates])

		# Variances
		self.variance_distance = functools.reduce(lambda a, b: a+b, [(c.getDistance(self.si, self.sj) - self.mean_distance) ** 2 for c in self.candidates]) / len(self.candidates)
		self.variance_width = functools.reduce(lambda a, b: a+b, [(c.ellipse[1][0] - self.mean_width) ** 2 for c in self.candidates]) / len(self.candidates)
		self.variance_length = functools.reduce(lambda a, b: a+b, [(c.ellipse[1][1] - self.mean_length) ** 2 for c in self.candidates]) / len(self.candidates)
		self.variance_orientation = functools.reduce(lambda a, b: a+b, [(c.orientation - self.mean_orientation) ** 2 for c in self.candidates]) / len(self.candidates)

		self.consistency_val = consistency_calculs([self.variance_distance, self.variance_length, self.variance_width, self.variance_orientation])

	def drawDebugSet(self, img):
		font = cv.FONT_HERSHEY_SIMPLEX

		for c in self.candidates:
			c.drawDebugCandidates(img)

		cv.ellipse(img, self.si.ellipse, (255, 0, 0))
		cv.ellipse(img, self.sj.ellipse, (0, 255, 0))

		y_1 = int(self.coefficients[0] * 0 + self.coefficients[1])
		y_2 = int(self.coefficients[0] * 640 + self.coefficients[1])
		cv.line(img, (0, y_1), (640, y_2), (0, 0, 0), 1)

		cv.putText(img, f'{self.variance_orientation}',(10,20),font,1,(255,0,0))
		cv.putText(img, f'{self.variance_distance}',(10,50),font,1,(255,0,0))
		cv.putText(img, f'{self.variance_width}',(10,80),font,1,(255,0,0))
		cv.putText(img, f'{self.variance_length}',(10,110),font,1,(255,0,0))
		cv.putText(img, f'{self.angle}',(10,140),font,1,(255,0,0))

	def append(self, candidate):
		self.candidates.append(candidate)
		self.calculateValue()

	def __getitem__(self, item):
		return self.candidates[item] # delegate to li.__getitem__

	def __str__(self):
		return f'Consitency set: {self.si}, {self.sj}, {len(self.candidates)}'