import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


def get_undistorted_image_params(filename,image):
    """
    @brief      This function is used to return the K matrix and distortion coefficients
    @param      image: The image to be processed
                filename: The filename for the image calibration ( Chesboard images ) can be found
                in the same folder as the main file.
    @return     dist,mtx: The distortion coefficients and the K matrix
    """
    
    obj_coords=np.zeros((6*9,3),np.float32)
    obj_coords[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    obj_pts=[]
    img_pts=[]
    img_files=glob.glob(filename)
    for img in img_files:
        img_cal=cv2.imread(img)
        img_gray=cv2.cvtColor(img_cal,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(img_gray,(9,6),None)
        
        if ret:
            obj_pts.append(obj_coords)
            img_pts.append(corners)
    img_res=(image.shape[1],image.shape[0])
    r, mtx, dist,_,_ = cv2.calibrateCamera(obj_pts, img_pts, (img_res), None, None)
    
    
    return dist,mtx
"""
   These parameters are found out by running the above code on the test images
   They are used to undistort the images
   Reference: https://docs.opencv.org/3.4/d7/d53/tutorial_py_calibration.html
"""
dist=np.array([[-0.24688833 ,-0.02372814, -0.00109843 , 0.00035105 ,-0.00259139]])
K_matrix=np.array([[1.15777942e+03, 0.00000000e+00 ,6.67111049e+02],
 [0.00000000e+00, 1.15282305e+03, 3.86129069e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])



def undistort(img,K_matrix,dist):
    """
    @breif This function will use the above K_matrix 
           and dist to undistort the image
    """
    undist=cv2.undistort(img,K_matrix,dist,None,K_matrix)
    return undist

    
def process_image(image):
    """This function will process the image and return the edges
    detected using the canny edge detection algorithm.

    Args:
        image (numpy:ndarray): The image to be processed

    Returns:
        image (numpy:ndarray): The image with the edges detected
    """
 
    image_cropped=image[420:720, 40:1280, :]
    
    image_undist=undistort(image_cropped,K_matrix,dist)
    
    # Converting to HLS color space to detect the yellow lines
    image_hls=cv2.cvtColor(image_undist,cv2.COLOR_BGR2HLS)
    
    # Detecting the yellow color in the image and applying a threshold to the image
    yellow_mask_lower=np.array([20,100,100],np.uint8)
    yellow_mask_upper=np.array([30,255,255],np.uint8)
    masked_yellow=cv2.inRange(image_hls,yellow_mask_lower,yellow_mask_upper)
    yellow_image=cv2.bitwise_and(image_hls,image_hls,mask=masked_yellow).astype(np.uint8)
    
    
    # Detecting the white color in the image and applying a threshold to the image
    white_mask_lower=np.array([0,200,0],np.uint8)
    white_mask_upper=np.array([255,255,255],np.uint8)
    masked_white=cv2.inRange(image_hls,white_mask_lower,white_mask_upper)
    
    white_image=cv2.bitwise_and(image_hls,image_hls,mask=masked_white).astype(np.uint8)
    
    # Now we combine the two images to get the yellow and white lines
    detected_lanes=cv2.bitwise_or(yellow_image,white_image)
    detected_lanes_bgr=cv2.cvtColor(detected_lanes,cv2.COLOR_HLS2BGR)
    detected_lanes_gray=cv2.cvtColor(detected_lanes_bgr,cv2.COLOR_BGR2GRAY)
    
    # We use bilateral filter to smooth out the image
    blur=cv2.bilateralFilter(detected_lanes_gray,9,120,100)

    
    # Now we detect edges in the image using Canny edge detection
    edges = cv2.Canny(blur,100,200,apertureSize = 3)
    
    return edges
    
def show_histogram(image):
    histogram=np.sum(image[image.shape[0]//2:,:],axis=0)
    plt.plot(histogram)
    plt.show()
    
def predict_turn(center_coords,left_coords,right_coords):
    """
    @breif This function will predict the turn based on the 
           center, left and right coordinates
    """
    lane_center_pos=left_coords+(right_coords-left_coords)/2
    if (lane_center_pos - center_coords < 0):
        return ('Turning Smooth left')
    elif (lane_center_pos - center_coords < 8):
        return 'Straight'
    else :
        return 'Turning Smooth right'
 
 
def draw_lines(image,left_lane_coords,right_lane_coords):
    # Draw the lines on the image
    left_lane_coords=np.int32(left_lane_coords)
    right_lane_coords=np.int32(right_lane_coords)
    cv2.polylines(image,[left_lane_coords],False,(255,0,124),5)
    cv2.polylines(image,[right_lane_coords],False,(255,0,124),3)
    return image
 
 
def draw_arrows(image,lane_center_pos):
    # Draw the arrows on the image
    # lane_center_pos=left_coords+(right_coords-left_coords)/2
    cv2.arrowedLine(image,(lane_center_pos[0][0],lane_center_pos[1][1]),(lane_center_pos[1][0],lane_center_pos[0][0]+50),(255,0,0),5)
    return image



def radiusofcurvature(image,left_fit,right_fit):
    """
    @breif This function will calculate the radius of curvature in meters given the 
           left and right lane fits
  
    Args:
        image (numpy:ndarray): The image to be processed
        left_fit (numpy:ndarray): The left lane line parameters
        right_fit (numpy:ndarray): The right lane line parameters

    Returns:
       left_curvature (float): The left lane curvature in meters
       right_curvature (float): The right lane curvature in meters
    """
     
     #For our Image , we have a resolution of 1280x720 pixels
     #Lane Length is 32 meters to get our Y-axis in meters we need to divide by 720
    height=image.shape[0]
    #Define conversions in x and y from pixels space to meters
    y_per_pixel=32/height
    
    left_curverad= ((1 + (2*left_fit[0]*y_per_pixel + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad= ((1 + (2*right_fit[0]*y_per_pixel + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad,right_curverad

def sliding_window(image,sliding_image,nwindows=12,margin=50,minpix=50,draw_windows=True):
    """This function will slide the window across the image and return the
    left and right lane coordinates along with the fitted polynomial coefficents.

    Args:
        image (numpy:ndarray): The image to be processed
        sliding_image (numpy:ndarray): The image to be processed(Color_BGR)
        nwindows (int): The number of windows to be used
        margin (int): The margin to be used
        minpix (int): The minimum number of pixels to be used
        draw_windows (bool): Whether to draw the windows or not

    Returns:
        left_lane_coords (numpy:ndarray): The left lane coordinates
        right_lane_coords (numpy:ndarray): The right lane coordinates
        l_fit (numpy:ndarray): The left lane polynomial coefficients
        r_fit (numpy:ndarray): The right lane polynomial coefficients
        predicted_turn (str): The predicted turn
        out_img (numpy:ndarray): The image with the lane lines drawn
        sliding_image (numpy:ndarray): The image with the lane lines drawn(Color_BGR)
    """
    # We find out the histogram of the image
    img_hist=np.sum(image,axis=0)
    # Divide the image into left and right halves
    midpoint_current=int(img_hist.shape[0]/2)
    # Finding the image center position
    image_center=int(image.shape[1]/2)
    right_current_x=np.argmax(img_hist[midpoint_current:])+midpoint_current
    left_current_x=np.argmax(img_hist[:midpoint_current])
    
    out_img = np.dstack((image,image,image))*255


    # Step through the windows one by one
    # Window height will divide the image into nwindows 600/12=50
    window_height=int(image.shape[0]/nwindows)
    # We use the nonzero array to find the non zero pixels in the image
    nonzero=image.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    # We created left and right line arrays to store the coordinates of the left and right lines
    left_lane_inds=[]
    right_lane_inds=[]
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low=image.shape[0]-(window+1)*window_height
        win_y_high=image.shape[0]-window*window_height
        win_x_low_left=left_current_x - margin
        win_x_high_left=left_current_x + margin
        win_x_low_right=right_current_x - margin
        win_x_high_right=right_current_x + margin
        # Draw the windows on the visualization image
        # Sliding_image is the image with the lane lines drawn(Color_BGR)
        # out_img is the image with the lane lines drawn
        if draw_windows==True:
            cv2.rectangle(sliding_image,(win_x_low_left,win_y_low),(win_x_high_left,win_y_high),(0,255,255),1)
            cv2.rectangle(sliding_image,(win_x_low_right,win_y_low),(win_x_high_right,win_y_high),(0,255,0),1) 
            cv2.rectangle(out_img,(win_x_low_left,win_y_low),(win_x_high_left,win_y_high),(0,255,255),1)
            cv2.rectangle(out_img,(win_x_low_right,win_y_low),(win_x_high_right,win_y_high),(0,255,0),1) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds= ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low_left) & (nonzerox < win_x_high_left)).nonzero()[0]
        good_right_inds= ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low_right) & (nonzerox < win_x_high_right)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If there are more number of pixels than minpix then we will use the mean of the pixels
        if len(good_left_inds) > minpix:
            left_current_x=int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current_x=int(np.mean(nonzerox[good_right_inds]))
            
            
    # Concatenate the arrays of indices
    left_lane_inds=np.concatenate(left_lane_inds)
    right_lane_inds=np.concatenate(right_lane_inds)
    
    
    # Extract left and right line pixel positions
    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
    
    # Set the left and right line 
    left=np.array([leftx,lefty])
    right=np.array([rightx,righty])
    
    
    # Fit a second order polynomial to each
    left_fit=np.polyfit(lefty,leftx,2)
    right_fit=np.polyfit(righty,rightx,2)
    
    # Visualization of Lane lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 255]
    # Predicted turn
    turn_prediction = predict_turn(image_center,left_current_x,right_current_x)  
    return left_fit,right_fit,out_img,turn_prediction,left,right,sliding_image


def project_back(image,left_fit,right_fit,cropped_img,prediction):
        """This function will slide the window across the image and return the
    left and right lane coordinates along with the fitted polynomial coefficents.

    Args:
        image (numpy:ndarray): The image with the fitted lane lines
        left_fit (numpy:ndarray): The left lane line polynomial coefficients
        right_fit (numpy:ndarray): The right lane line polynomial coefficients
        cropped_img (numpy:ndarray): The cropped image
        prediction (int): The predicted turn

    Returns:
        
    """
        
        plot_y = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fit_x=left_fit[0]*plot_y**2+left_fit[1]*plot_y+left_fit[2]
        right_fit_x=right_fit[0]*plot_y**2+right_fit[1]*plot_y+right_fit[2]
        
        # Creating a blank image to draw the lines on
        blank_img=np.zeros_like(image).astype(np.uint8)
        
        # Findings the radius of curvature of the lane lines
        radius_of_curvature=radiusofcurvature(image,left_fit,right_fit)
        
        
        # Adding the color channel for our image
        warp_color_channel=np.dstack((blank_img,blank_img,blank_img))
        
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        left_points=np.array([np.transpose(np.vstack([left_fit_x,plot_y]))])
        right_points=np.array([np.flipud(np.transpose(np.vstack([right_fit_x,plot_y])))])
        final_points=np.hstack((left_points,right_points)).astype(int)
        
        # Lane Center Postion Calculation for the center of the lane
        lane_center_pos=left_points+(right_points-left_points)/2
        lane_center_pos=np.int32(lane_center_pos)
        lane_center_pos=lane_center_pos[0].reshape(600,2)
        

        # Draw the lane onto the warped blank image using fillPoly
        cv2.fillPoly(warp_color_channel,[final_points],(0,200,0))
        
        # Draw the lane onto the warped blank image using polylines  
        warp_color_channel=draw_lines(warp_color_channel,left_points,right_points)
        
        # Draw the arrow onto the warped blank image using arrowedLine
        warp_color_channel=draw_arrows(warp_color_channel,lane_center_pos)
        
        
        # Now we warp our lane image to our original image using inverse perspective matrix
        h_inv=cv2.getPerspectiveTransform(world_coords_,bird_eye_coords_)
        final_warped_image=cv2.warpPerspective(warp_color_channel,h_inv,(cropped_img.shape[1],cropped_img.shape[0]))
        
        # Now we add the original image and the lane image
        final_output=cv2.addWeighted(cropped_img,1,final_warped_image,0.3,0)
        
        # Add the radius of curvature and the predicted turn
        cv2.putText(final_output,'Turn:'+str(prediction),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(final_output,'Radius of Curvature Left:'+str(round(radius_of_curvature[0],0))+" m",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
        cv2.putText(final_output,'Radius of Curvature Right:'+str(round(radius_of_curvature[1],0))+ " m",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
        
        return final_output,left_points,right_points,warp_color_channel
    
def show_two_images(image1,image2):
    

    im1=cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255) )
    im2=cv2.copyMakeBorder(image2,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255) )
    cv2.putText(im1,'Yellow Mask',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(im2,'White Mask',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    Hor=np.concatenate((im1,im2),axis=1)
    cv2.imwrite('problem3/output_images/output_image.jpg',Hor)
    cv2.imshow('Distort vs Undistort',Hor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_output_video(org_img,edges_img,bird_img,sliding_img,final_output,warped):
    
    height, width = 1080, 1920
    final_img=np.zeros((height,width,3), np.uint8)
    edges_img=np.dstack((edges_img,edges_img,edges_img))
    bird_img=np.dstack((bird_img,bird_img,bird_img))
    
    cv2.putText(org_img,'[1] Main Image',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(edges_img,'[2] Detected Lanes',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    
    final_img[0:720,0:1280] = cv2.resize(final_output, (1280,720), interpolation=cv2.INTER_AREA)
    final_img[0:360,1280:1920] = cv2.resize(org_img, (640,360), interpolation=cv2.INTER_AREA)
    final_img[360:720,1280:1920] = cv2.resize(edges_img, (640,360), interpolation=cv2.INTER_AREA)
    final_img[720:1080,1280:1920] = cv2.resize(bird_img, (640,360), interpolation=cv2.INTER_AREA)
    final_img[720:1080,0:640] = cv2.resize(warped, (640,360), interpolation=cv2.INTER_AREA)
    neon= np.zeros((100, final_img.shape[1], 3), np.uint8)
    neon[:] = (255, 0, 180) 
    final_img[720:1080,640:1280] = cv2.resize(sliding_img, (640,360), interpolation=cv2.INTER_AREA)
    final_img=cv2.vconcat((final_img,neon))
    cv2.putText(final_img,'[3] Bird eye view',(1333,788), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(final_img,'[5] Polynomial fit',(61,802), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(final_img,'[4] Sliding Window',(608,787), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(final_img,'[6] Lane Detection',(40,675), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 3, 0)
    cv2.putText(final_img,'[6] Lane Detection',(532,1158), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    cv2.putText(final_img,'[1] Main Image',(31,1116), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    cv2.putText(final_img,'[2] Detected Lanes',(524,1110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    cv2.putText(final_img,'[3] Bird eye view',(989,1110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    cv2.putText(final_img,'[5] Polynomial fit',(31,1163), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    cv2.putText(final_img,'[4] Sliding Window',(1406,1109), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255), 2, 0)
    # cv2.imwrite('problem3/output_images/output_final.jpg',final_img)
    return final_img

    
    
# Source points for homography.
bird_eye_coords_= np.float32([[500, 50], [686, 41], [1078, 253], [231, 259]])
# Destination points for homography
world_coords_ = np.float32([[50, 0], [250, 0], [250, 500], [0, 500]])

# Calibration using checkerboad images
filename_cal='problem3/camera_cal/calibration*.jpg'

# Load the Video
filename='problem3/challenge.mp4'
cap=cv2.VideoCapture(filename)
print("Detecting Lane from Video ...")
output_filename='problem3/output_images/output_video.mp4'
size=(1920,1180)
result = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

Frame = 0
while (cap.isOpened()):
    ret, img = cap.read()
    if ret:
        Frame+=1
        print('Frame: ',Frame)
        # cv2.imshow('Original frame',img)
        img_edges=process_image(img)
        cropped_img=img[420:720, 40:1280, :]
        h_, mask = cv2.findHomography( bird_eye_coords_,world_coords_,cv2.RANSAC,5.0)
        im2_n=cv2.warpPerspective(img_edges,h_,(300,600),flags=cv2.INTER_LINEAR)
        # We use the below im_sliding to show the colored image
        im_sliding=cv2.warpPerspective(cropped_img,h_,(300,600),flags=cv2.INTER_LINEAR)
        l_fit,r_fit,out_img,pred,l_,r_,image=sliding_window(im2_n,im_sliding)
        # cv2.imshow('Fitting a Polynomial',out_img)
        # cv2.imshow("Sliding Window Technique",image)
        final_output,left,right,warped_color=project_back(im2_n,l_fit,r_fit,cropped_img,pred)
        lanes_img=draw_lines(im_sliding,left,right)
        # cv2.imshow("Edges",img_edges)
        # cv2.imshow('output',final_output)
        # cv2.imshow('warpped',warped_color)
        # cv2.imshow("Detected Lanes",lanes_img)
        fin_img=show_output_video(img,img_edges,im2_n,out_img,final_output,warped_color)
        cv2.imshow('Final Output',fin_img)

        result.write(fin_img)
        # print('Prediction: ',pred)
        




        #Uncomment the below line to show the histogram of the image
        # show_histogram(im2_n)
        
        # Uncomment the belwo line to find the Camera Calibration and distortion coefficients
        # dist,mtx=get_undistorted_image(filename_cal,img)
        # print(dist)
        # print("Matrix: ",mtx)
        
      
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
result.release()
cv2.destroyAllWindows()
print("The video has been processed successfully")


        






