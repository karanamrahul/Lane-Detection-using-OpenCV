import numpy as np
import cv2
import matplotlib.pyplot as plt


def cumulative_image_hist(img):
    """
    Create a cumulative histogram of the given image.
    """
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cum_hist = hist.cumsum()
    cum_hist = cum_hist * hist.max() / cum_hist.max()
    cum_hist_normalized = ((cum_hist / cum_hist.max()) * 255).astype(np.int8)
    return cum_hist_normalized
    
def show_histogram(img):
    """
    @breif : This function is used to show the histogram of the given image.
    
    Args:
        img: The image to be shown.
        
    Returns:
        None
    """
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].imshow(img,cmap="gray")
    ax[0].axis('off')
    ax[0].set_title("Image")

    ax[1].hist(img.ravel(), bins=40, range=(0.0, 256.0), ec='k') #calculating histogram
    ax[1].set_title("Histogram")
    ax[1].set_xlabel("range")
    plt.show()
  
def show_equalized_histogram(img_eq):
    """
    @breif : This function is used to show the histogram of the given image which is equalized.
    
    Args:
        img_eq: The image to be shown.
        
    Returns:
        None
    """

    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].imshow(img_eq,cmap="gray")
    ax[0].axis('off')
    ax[0].set_title("Image")
    ax[1].hist(img_eq.ravel(), bins=40, range=(0.0, 256.0), ec='k') #calculating histogram
    ax[1].set_title("Histogram")
    ax[1].set_xlabel("range")
    plt.show()



def show_equalized_image_with_histogram(img_eq):
    """
    @brief: This function is used to show the equalized image with histogram.
    
    Args:   
        img_eq: The image to be shown.
        
    Returns:
        None
    """
    img_eq=np.asarray(img_eq,dtype=np.uint8)
    plt.hist(img_eq.flatten(), bins=40, range=(0.0, 256.0), ec='k') #calculating histogram
    # plt.gca().invert_xaxis()
    plt.show()
   

def create_equalized_image(img):
    """
    @brief : This function is used to create the equalized image.
    
    Args:
        img: The image to be equalized.
        
    Returns:
        img_eq: The equalized image.
        
    """
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(img)
    cum_hist=cumulative_image_hist(v)
    height,width=v.shape
    img_eq=np.zeros(v.shape)
    for i in range(height):
        for j in range(width):
            img_eq[i][j]=cum_hist[v[i][j]]
    img_eq=img_eq.astype(np.uint8)
    img_eq=cv2.merge([h,s,img_eq])
    img_eq=cv2.cvtColor(img_eq,cv2.COLOR_HSV2BGR)
    return img_eq

def show_equalized_image(img_eq):
    plt.imshow(img_eq)
    plt.show()

def histogram_equalization(img):
    """
    @brief : This function is used to perform histogram equalization on the given image.
    
    Args:
        img: The image to be equalized.
    
    Returns:
        img_eq: The equalized image.
    """
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = np.uint8(cdf/cdf.max()*255)
    equilized = cdf_normalized[img]
    return equilized


def split_hsv_merge_adaptive(img):
    """
    @brief:Split the given image into hsv channels and merge them back.
    
    Args:
        img: The image to be split.
    
    Returns:
        img_2: The merged image.
    """

    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_h=img_hsv[:,:,0]
    img_s=img_hsv[:,:,1]
    img_v=img_hsv[:,:,2]
    img_v_2=adaptive_histogram_equalization(img_v)
    img_hsv_2=cv2.merge((img_h,img_s,img_v_2))
    # Adaptive Histogram Equalization
    img_2=cv2.cvtColor(img_hsv_2,cv2.COLOR_HSV2BGR)

    return img_2


def adaptive_histogram_equalization(img):
    """
    @brief: Perform adaptive histogram equalization on the given image.
    
    Args:
        img: The image to be equalized.
    Returns:
        img_eq: The equalized image.(Adaptive Histogram Equalization)
    """
    img_height=int(img.shape[0]/8)
    img_width=int(img.shape[1]/8)
    for i in range(0,img.shape[0],img_height):
        for j in range(0,img.shape[1],img_width):
                img_block=img[i:i+img_height,j:j+img_width]
                img_block_eq=histogram_equalization(img_block)
                img[i:i+img_height,j:j+img_width]=img_block_eq
                img[int(i*img_height):int((i+1)*img_height),int(j*img_width):int((j+1)*img_width)]=histogram_equalization(img[int(i*img_height):int((i+1)*img_height),int(j*img_width):int((j+1)*img_width)])
    
    img=cv2.bilateralFilter(img,5,75,75)

    return img

    
    

def CLAHE(img):
    """
    Perform CLAHE on the given image using the opencv inbuilt funciton.
    """
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_h=img_hsv[:,:,0]
    img_s=img_hsv[:,:,1]
    img_v=img_hsv[:,:,2]
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img_v_2=clahe.apply(img_v)
    img_hsv_2=cv2.merge((img_h,img_s,img_v_2))

    img_cl=cv2.cvtColor(img_hsv_2,cv2.COLOR_HSV2BGR)


    return img_cl


def show_two_images(image1,image2):
    

    im1=cv2.copyMakeBorder(image1,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255) )
    im2=cv2.copyMakeBorder(image2,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255) )
    cv2.putText(im1,'CLAHE Output',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,124), 3, 0)
    cv2.putText(im2,'Adaptive Histogram Equalization',(30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,124), 3, 0)
    Hor=np.concatenate((im1,im2),axis=1)
    return Hor
 
    
class histogram_equalization_demo:
    def __init__(self, img):
        self.img = img
        self.img_eq=create_equalized_image(img)
        self.img_ad=split_hsv_merge_adaptive(img)
        self.img_cl = CLAHE(img)

    def show(self):
        cv2.imshow('Original', self.img)
        cv2.imshow('Histogram Equalization', self.img_eq)
        cv2.imshow('Adaptive Histogram Equalization', self.img_ad)
        cv2.imshow('CLAHE', self.img_cl)
     

if __name__ == '__main__':
    output_filename='problem1/output_video_3.mp4'
    size=(2488,390)
    result = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(25):
        img=cv2.imread('problem1/adaptive_hist_data/{:010d}.png'.format(i))
        # Uncomment the line below to show all the outputs
        histogram_equalization_demo(img).show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # h_cat=show_two_images(CLAHE(img),split_hsv_merge_adaptive(img))
        # cv2.imwrite('problem1/output_image_3.jpg',h_cat)
        # result.write(h_cat)
        # cv2.imwrite('problem1/adaptive_hist_data/histogram_eq_output/{:010d}_eq.png'.format(i), histogram_equalization(img))
        # cv2.imwrite('problem1/adaptive_hist_data/adaptive_he/{:010d}_ad.png'.format(i), adaptive_histogram_equalization(img))
        # cv2.imwrite('problem1/adaptive_hist_data/clahe_out/{:010d}_cl.png'.format(i), CLAHE(img))
    result.release()
    print("Video saved as {}".format(output_filename))   



