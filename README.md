# Automatic-Lane-Detection-System
# 
# 
# Approach:
# 
# Initially, we read the image from the video. We then only keep white and yellow pixels as the lanes are only marked using white and yellow lines. We define lower and upper threshold to obtain different shades of white and yellow. For white, the lower threshold ranges from 130-200, whereas the upper threshold is 255. For yellow, the lower threshold for RGB color is [ 90,100,100] and the upper threshold is [110,255,255]. We use these thresholds and mask the pixels of yellow and white by using the cv2.inRange() function and then perform bitwise and operation on the original image and the output of cv2.inRange() to get the image which only consists of white and yellow pixels. 
# Now we convert our image which only has white and yellow pixels to grayscale and then we apply gaussian blur to remove any noise present in the image. We then use the canny edge detector to detect the edges in the image with a low threshold of 50 and high threshold of 150. Then we define our region of interest in the image in the shape of  a trapezoid and create another image only with region of interest which fully contains the lanes. We define the vertices of the trapezoid and the region outside the vertices is set to black. 
# Then we perform Probabilistic Hough transform on the resultant image with 10 being the minimum number of pixels making up a line and 20 being the maximum gap in pixels between connectable line segments. 
# We calculate the slope of the lines and then divide the lines into right and left lines, the right lines must have a positive slope and should be on the right side of the image whereas the left lines must have negative slope and should be on the left side of the image. 
# We also define a slope threshold which is equal to 0.5. We only consider lines whose slope is greater than the above defined slope threshold. 
# We then perform polynomial regression on the obtained right and left lines to find the best fit for lines of the lane. We use np.polyfit() function to extrapolate. 
# We then finally calculate two endpoints for each of left and right lines to draw the lines and draw them using cv2.line() function with a thickness 10 and color red which are passed as parameters to draw_lines() function along with the resultant image after we perform Probabilistic Hough transform.  
# Our project works for both images and a video, we follow the same approach for drawing lines on video as well, we use fl_image() of MoviePy and then follow the same approach as above to get our desired results. 
