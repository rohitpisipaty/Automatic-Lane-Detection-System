import cv2
import numpy as np
import os
import math
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Region of Interest defined by the trapezoid-shaped vertices is kept by applying a mask.
#Rest of the image is blacked.	
def ROI(image1, vertices):
	
	#Black mask is defined.
	mask = np.zeros_like(image1)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(image1.shape) > 2:
		# i.e. 3 or 4 depending on your image
		NumOfChannels = image1.shape[2]  
		maskColorNeglect = (255,) * NumOfChannels
	else:
		maskColorNeglect = 255
	
	#Filling color inside trapezoid-shape
	cv2.fillPoly(mask, vertices, maskColorNeglect)
	
	#returning the image only where mask pixels are nonzero
	imageMask = cv2.bitwise_and(image1, mask)
	return imageMask

# Hough Transform
anglerho = 2 
angletheta = 1 * np.pi/180 
# minimum number of intersections in Hough grid cell
threshold = 15	
#minimum number of pixels to make a line 
lenLineMin = 10 
# maximum gap to between connectable line segments(in pixels)
lineGapMax = 20	


#Converts the image to grayscale image
def convertGray(image1):
	return cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

#Applies gausssian blur on grayscale image
def transformGaussian(image1, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(image1, (kernel_size, kernel_size), 0)
	
#Applies canny transform on grayscale image and gausssian image
def transformCanny(image1, thresholdLow, thresholdHigh):
	return cv2.Canny(image1, thresholdLow, thresholdHigh)


#Lines for lanes are drawn. Color of line can be set here.
#Lines are segregated into left and right lines depending upon their slopes.
def lineMarking(image1, lines, color=[0, 255, 255], thickness=10):
	# Lines are not drawn, if error occured.
	if lines is None:
		return
	if len(lines) == 0:
		return
	rightLine = True
	leftLine = True
	
	#Slopes of all lines are found and slopes with abs(slope) > thresholdSlope are considered
	thresholdSlope = 0.8
	slopes = []
	linesRecent = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
#Slope Calculation
		#Avoiding division by 0
		if x2 - x1 == 0.: 
			#Infinite slope
			slope = 999.  
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		#Lines are refined based on slope
		if abs(slope) > thresholdSlope:
			slopes.append(slope)
			linesRecent.append(line)
		
	lines = linesRecent
	
	# If a line has positive slope it is made as a Right lane 
	# and if the line has negative slope it is made as Left line
	lineRight = []
	lineLeft = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		#Center of image's X-coordinate
		xCenterImg = image1.shape[1] / 2  
		if slopes[i] > 0 and x1 > xCenterImg and x2 > xCenterImg:
			lineRight.append(line)
		elif slopes[i] < 0 and x1 < xCenterImg and x2 < xCenterImg:
			lineLeft.append(line)
	
	#Finding best fit lines for left and right lanes. Linear regression is used.
	#Lines for right lanes
	lineRightX = []
	lineRighty = []
	
	for line in lineRight:
		x1, y1, x2, y2 = line[0]
		
		lineRightX.append(x1)
		lineRightX.append(x2)
		
		lineRighty.append(y1)
		lineRighty.append(y2)
		
	if len(lineRightX) > 0:
		slopeRight, cRight = np.polyfit(lineRightX, lineRighty, 1)  # y = m*x + c
	else:
		slopeRight, cRight = 1, 1
		rightLine = False
		
	#Lines for left lanes
	lineLeftX = []
	lineLefty = []
	
	for line in lineLeft:
		x1, y1, x2, y2 = line[0]
		
		lineLeftX.append(x1)
		lineLeftX.append(x2)
		
		lineLefty.append(y1)
		lineLefty.append(y2)
		
	if len(lineLeftX) > 0:
		slopeLeft, cLeft = np.polyfit(lineLeftX, lineLefty, 1)  # y = m*x + c
	else:
		slopeLeft, cLeft = 1, 1
		leftLine = False
	
	#To draw the line,two end points are found for left and right lines.
	#Equation of line : y = m*x + c ==> x = (y - c)/m
	y1 = image1.shape[0]
	y2 = image1.shape[0] * (1 - heigthROI)
	
	lineRightX1 = (y1 - cRight) / slopeRight
	lineRightX2 = (y2 - cRight) / slopeRight
	
	lineLeftX1 = (y1 - cLeft) / slopeLeft
	lineLeftX2 = (y2 - cLeft) / slopeLeft
	
	y1 = int(y1)
	y2 = int(y2)
	lineRightX1 = int(lineRightX1)
	lineRightX2 = int(lineRightX2)
	lineLeftX1 = int(lineLeftX1)
	lineLeftX2 = int(lineLeftX2)
	
	#Left and right lines are drawn on image
	if rightLine:
		cv2.line(image1, (lineRightX1, y1), (lineRightX2, y2), color, thickness)
	if leftLine:
		cv2.line(image1, (lineLeftX1, y1), (lineLeftX2, y2), color, thickness)

#Hough lines are drawn on image1(output of canny)
def linesHough(image1, anglerho, angletheta, threshold, min_line_len, lineGapMax):
	lines = cv2.HoughLinesP(image1, anglerho, angletheta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=lineGapMax)
	x = image1.shape[0]
	y = image1.shape[1]
	imageLine = np.zeros((x,y,3), dtype=np.uint8)  # 3-channel RGB image
	lineMarking(imageLine, lines)
	return imageLine

#image1 and initial image are combined to get final image that is our original images
#with lane lines detected imposed on it.
def imageWeight(image1, initial_img, alpha=0.8, beta=1., gamma=0.):
	return cv2.addWeighted(initial_img, alpha, image1, beta, gamma)

#Original image is refined to produce image with only white and yellow pixels
def refineImage(image):

	#Refining white pixels
	threshWhite = 75 #130
	whiteLow = np.array([threshWhite, threshWhite, threshWhite])
	whiteHigh = np.array([255, 255, 255])
	maskWhite = cv2.inRange(image, whiteLow, whiteHigh)
	imageWhite = cv2.bitwise_and(image, image, mask=maskWhite)
        
        
	#Refining yellow pixels
	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	yellowLow = np.array([90,100,100])
	yellowHigh = np.array([110,255,255])
	maskYellow = cv2.inRange(HSV, yellowLow, yellowHigh)
	imageYellow = cv2.bitwise_and(image, image, mask=maskYellow)

	#Combining white pixel and yellow pixel images.
	imageCombine = cv2.addWeighted(imageWhite, 1., imageYellow, 1., 0.)

	return imageCombine

#Starting processing on image from here
def processImage(image_in):
	#White pixels and yellow pixels are kept in the image while blacking out rest of the pixels.
	image = refineImage(image_in)

	#Converting input image into grayscale image.
	imageGray = convertGray(image)

	#Applies Gaussian smoothing
	imageGrayBlur = transformGaussian(imageGray, kernel_size)

	#Applies Canny Edge Detector
	edges = transformCanny(imageGrayBlur, thresholdLow, thresholdHigh)

	#Creates our region of interest using trapezoidal shape.
	imshape = image.shape
	vertices = np.array([[\
		((imshape[1] * (1 - lengthBottom)) // 2, imshape[0]),\
		((imshape[1] * (1 - widthTop)) // 2, imshape[0] - imshape[0] * heigthROI),\
		(imshape[1] - (imshape[1] * (1 - widthTop)) // 2, imshape[0] - imshape[0] * heigthROI),\
		(imshape[1] - (imshape[1] * (1 - lengthBottom)) // 2, imshape[0])]]\
		, dtype=np.int32)
	masked_edges = ROI(edges, vertices)

	#Performing Hough transform on edge detected image
	lineImage = linesHough(masked_edges, anglerho, angletheta, threshold, lenLineMin, lineGapMax)
	
	#Drawing lane lines on the original image
	originalImage = image_in.astype('uint8')
	processedImage = imageWeight(lineImage, originalImage)

	return processedImage

#for the given input image, processed image is saved to the output image
def imageProcessing(input_file, output_file):
	processedImage = processImage(mpimg.imread(input_file))
	plt.imsave(output_file, processedImage)

##for the given input video, processed image is saved to the output video
def videoProcessing(input_file, output_file):
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(processImage)
	annotated_video.write_videofile(output_file, audio=False)

# we are selecting a region of interest in Trapezoidal shape 

#Trapezoids bottom edge width (in percentage)
lengthBottom = 0.6 

#Trapezoids Top edge width (in percentage)
widthTop = 0.4  

#Trapezoids Height (in percentage)
heigthROI = 0.25  


#Thresholds for Canny
thresholdLow = 50
thresholdHigh = 150


#Kernel size for Gaussian
kernel_size = 3


	
	
	
# Main script
if __name__ == '__main__':
	from optparse import OptionParser

	# Configure command line options
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest="input_file",
					help="Input video/image file")
	parser.add_option("-o", "--output_file", dest="output_file",
					help="Output (destination) video/image file")
	parser.add_option("-I", "--image_only",
					action="store_true", dest="image_only", default=False,
					help="Annotate image (defaults to annotating video)")

	# Get and parse command line options
	options, args = parser.parse_args()

	input_file = options.input_file
	output_file = options.output_file
	image_only = options.image_only

	if image_only:
		imageProcessing(input_file, output_file)
	else:
		videoProcessing(input_file, output_file)
