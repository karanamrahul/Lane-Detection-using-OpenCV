import matplotlib.pyplot as plt
import numpy as np
import cv2
from hough import *



def process_image(image):
    """
    @breif - Takes in an image and uses Canny Edge Detection and Hough Transformations
    to detect lines in the image and returns the hough image,lines, and the 
    detected edges.
    
    Args:
        image: The image to be processed
        
    Returns:
        hough_image: The hough image
        mask: The mask of the edges
        lines: The lines detected by the hough transform
        edges: The edges detected by the canny edge detection
    """

    img_shape = image.shape
    
    # These vertices are used to mask the region of interest
    vertices = np.array([[(0,img_shape[0]), (9*img_shape[1]/20, 11*img_shape[0]/18), (11*img_shape[1]/20, 11*img_shape[0]/18), (img_shape[1],img_shape[0])]], dtype=np.int32)
   
   
    #  Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blur, 25, 100)

    # Apply mask to remove unwanted area
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)
    
    # We use the below code to show the masked image in the image ( I used this approach to show the masked image in the image)
    img_r_2,_=region_of_interest(image)
    
    # Apply the mask using and operation
    new_img=cv2.bitwise_and(img_r_2,img_r_2,mask=masked)
    
    # Convert to grayscale
    new_img=cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Hough Probabilistic Transform to detect lines
    lines = cv2.HoughLinesP(new_img, 1, np.pi/180, 14, np.array([]), minLineLength=30, maxLineGap=60)
    hough_image = np.zeros((*new_img.shape, 3), dtype=np.uint8)


    return hough_image,masked,lines,edges

def draw_lines(img, lines,color_left,color_right,thickness=13):
    """
    @breif : This function is used to draw the lines on the image
    
    Args:
        img: Input image
        lines: Lines detected by the hough transform
        color_left: Color of the left line
        color_right: Color of the right line
        
    Returns:
        img: The image with the lines drawn on it

    """
    
    # Initial values of the left and right lines
    left_slope_current = 0
    right_slope_current = 0
    left_init = [0, 0, 0]
    right_init = [0, 0, 0]
    weight = 0.9

    # Creating empty arrays to store the lines detected
    right_y_coords = []
    right_x_coords = []
    right_slopes = []

    left_y_coords = []
    left_x_coords = []
    left_slopes = []

    image_center = img.shape[1] / 2

    image_bottom = img.shape[0]
    
    
    # We iterate over the lines detected
    for line in lines:
        for x1,y1,x2,y2 in line:
            # We fit a first order polynomial to the line
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # We check if the slope is positive or negative
            if .35 < np.absolute(slope) <= .85:
                # We check if the slope is greater than the image center or not and we store the lines in the corresponding arrays
                if slope > 0 and x1 > image_center and x2 > image_center:
                    right_y_coords.append(y1)
                    right_y_coords.append(y2)
                    right_x_coords.append(x1)
                    right_x_coords.append(x2)
                    right_slopes.append(slope)
                elif slope < 0 and x1 < image_center and x2 < image_center:
                    left_y_coords.append(y1)
                    left_y_coords.append(y2)
                    left_x_coords.append(x1)
                    left_x_coords.append(x2)
                    left_slopes.append(slope)
   
    

    # We check if the left and right lines are detected
    
    # If the right line is detected, we calculate the slope and the y-intercept
    # We then calculate the y-coordinates of the right line
    # We then calculate the x-coordinates of the right line
    if right_y_coords:
        right_index = right_y_coords.index(min(right_y_coords))
        right_x1 = right_x_coords[right_index]
        right_y1 = right_y_coords[right_index]
        right_slope = np.median(right_slopes)
        if right_slope_current != 0:
            right_slope = right_slope + (right_slope_current - right_slope) * weight
        # We calculate the x-coordinates of the left line
        right_x2 = int(right_x1 + (image_bottom - right_y1) / right_slope)

        if right_slope_current != 0:
            right_x1 = int(right_x1 + (right_init[0] - right_x1) * weight)
            right_y1 = int(right_y1 + (right_init[1] - right_y1) * weight)
            right_x2 = int(right_x2 + (right_init[2] - right_x2) * weight)

        right_slope_current = right_slope
        right_init = [right_x1, right_y1, right_x2]
        # We draw the right line on the image with the calculated coordinates,color and thickness
        cv2.line(img, (right_x1, right_y1), (right_x2, image_bottom), color_right, thickness)
        
        
    

    if left_y_coords:
        # If the left line is detected, we calculate the slope and the y-intercept
        
        left_index = left_y_coords.index(min(left_y_coords))
        left_x1 = left_x_coords[left_index]
        left_y1 = left_y_coords[left_index]
        # We find the median of the slopes
        left_slope = np.median(left_slopes)
        if left_slope_current != 0:
            left_slope = left_slope + (left_slope_current - left_slope) * weight
        # We calculate the x-coordinates of the left line
        left_x2 = int(left_x1 + (image_bottom - left_y1) / left_slope)

        # If the slope is not zero, we calculate the x-coordinates of the left line
        # We then calculate the y-coordinates of the left line
        if left_slope_current != 0:
            left_x1 = int(left_x1 + (left_init[0] - left_x1) * weight)
            left_y1 = int(left_y1 + (left_init[1] - left_y1) * weight)

        left_slope_current = left_slope
        left_init = [left_x1, left_y1, left_x2]
        # We draw the left line on the image with the calculated coordinates,color and thickness
        cv2.line(img, (left_x1, left_y1), (left_x2, image_bottom), color_left, thickness)
        
        
        
    


def diff_dash_solid_line(image):
    """
    @brief     This function is used to detect the dashed and solid lines in the image.
    
    Arguments: 
    
    image:     The image in which the lines are to be detected.
    
    Returns:
    
    image:     The image with the detected dashed and solid lines.
    left_current_x:     The current x coordinate of the left line.
    right_current_x:    The current x coordinate of the right line.
    non_zero_pixel_count:    The number of non zero pixels in the image.
    left_count:    The number of times the left line has been detected.
    right_count:   The number of times the right line has been detected.
    """
    
    
    # This code is used to find the histogram peaks and find the maximum value of the histogram peaks
    # I have divided them to two halves and found the maximum value of the histogram peaks in each half
    # This method only works with the original image , when the image is flipped the histogram peaks are not found properly
    img_hist=np.sum(image[image.shape[0]//2:,:], axis=0)
    image_center_current=int(img_hist.shape[0]/2)
    right_current_x=np.argmax(img_hist[image_center_current:])+image_center_current
    left_current_x=np.argmax(img_hist[:image_center_current])
    non_zero_pixels=img_hist.nonzero()
    count_pixels=cv2.findNonZero(image)
    
    
    
    # The below code is used to find the divide the image into two parts
    # Then we count the number of pixels in each part
    
    mid_reg=int(image.shape[1]/2)
    left_part=image[:,:mid_reg]
    right_part=image[:,mid_reg:]
    
    # using the count pixels function we find the number of pixels in each part
    left_count = cv2.findNonZero(left_part)
    right_count=cv2.findNonZero(right_part)
 

    return left_current_x,right_current_x,non_zero_pixels,left_count,right_count


def region_of_interest(img):
    """
    @breif : This function is used to mask the image to get only the region of interest
    
    Args:
        img: Input image
        
    Returns:
        img_masked: Masked image
    """
    img_shape = img.shape[:2]
    vertices = np.array([[(0,img_shape[0]), (9*img_shape[1]/20, 11*img_shape[0]/18), (11*img_shape[1]/20, 11*img_shape[0]/18), (img_shape[1],img_shape[0])]], dtype=np.int32)
    mask = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(mask, [vertices], (255,255,255))
    image= cv2.bitwise_and(mask,img)
    masked_img=cv2.bitwise_and(img,mask)
    return image,masked_img
      
    
def show_histogram(image):
    """
    @breif : This function is used to show the histogram of the image

    Args:
        image (_type_): _description_
    """
    histogram=np.sum(image[image.shape[0]//2:,:],axis=0)
    plt.plot(histogram)
    plt.show()
    
def show_final_image(image,hough_image,lines,l_count,r_count):
    """ 
    @breif : This function is used to show the final image with the lane lines
    
    Args:
        image: Input image
        hough_image: Blank Image to show the output
        lines : Lines detected by the Hough transform
        l_count: The number of pixels in the left lane
        r_count: The number of pixels in the right lane
        
        
    Returns:
        final_image: Final image with the lane lines
    
    """
   # Here we check whether the lines detected are solid or dashed
   # if the number of pixels in the left lane is greater than the number of pixels in the right lane
   # then the lines are solid in the left lane and dashed in the right lane and vice-versa.
   
    if  r_count.shape[0] > l_count.shape[0]:
        color_left = (0,0,255)
        text1="Left Lane: Detected Dashed Lines"
        color_right= (0,255,0)
        text2="Right Lane: Detected Solid Lines"
    else:
         print("left")
         color_left = (0,255,0)   
         text1 = "Left Lane: Detected Solid Lines"
         color_right = (0,0,255)    
         text2="Right Lane: Detected Dashed Lines" 
    
    
    # We draw the lines on the blank image using the color specified above
    draw_lines(hough_image,lines,color_left,color_right)
    
    # We then add our blank image to the original image and show it
    processed = cv2.addWeighted(image, 0.8, hough_image, 1, 0)
    
    # We add the text to the image
    cv2.putText(processed,text1,(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
    cv2.putText(processed,text2,(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)

    
    return processed


######################################################################################################

# Source points for homography.
bird_eye_coords_= np.float32([[410,335], [535, 334], [780, 479], [150, 496]])
# Destination points for homography
world_coords_ = np.float32([[50, 0], [250, 0], [250, 500], [0, 500]])



if __name__ == "__main__":
    source = 'problem2/whiteline.mp4'
    cap = cv2.VideoCapture(source)
    print('Video Rendering started ...')
    print("Detecting Lane from Video ...")
    output_filename='problem2/output_video.mp4'
    size=(960,540)
    result = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    Frame=0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            Frame+=1
            print('Frame: ',Frame) 
            
            # Uncomment the below line to see the video flipped horizontally
            # img_=cv2.flip(img,1) 
            
            hough_image,masked_edges,lines,edges=process_image(img)
            im,mask=region_of_interest(img)
            h_, mask = cv2.findHomography( bird_eye_coords_,world_coords_,cv2.RANSAC,5.0)
            im2_n=cv2.warpPerspective(masked_edges,h_,(300,600),flags=cv2.INTER_LINEAR)
            l,r,nxy,lcount,rcount=diff_dash_solid_line(im2_n)
            final_output=show_final_image(img,hough_image,lines,lcount,rcount)
            cv2.imshow('frame',final_output)
            
            # Uncomment the below line to see the hough space
            # show_hough_plots(img,edges,masked_edges)
            
            # uncomment the below line to save the video
            # result.write(final_output)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
cap.release()  
result.release()      
cv2.destroyAllWindows()
print("The video has been processed successfully")