import cv2
import numpy as np
import matplotlib.pyplot as plt


def houghLine(image):
    """ Implementation of Standard Hough Line transform in parameter space ( Hough Space ).

    Args:
        image (numpy narray): The image is the output from the canny or any other edge detector and it should 
                              be in grayscale.
                                
        Returns:
            lines (numpy narray): A 2D array of the lines in the image. Each line is represented by a tuple of 
                                the form (rho, theta).
    """
    height, width = image.shape
    len_diag=int(np.round(np.sqrt(height**2+width**2)))
    # Intialize theta and rho arrays
    theta = np.linspace(-np.pi / 2, np.pi / 2, len_diag)
    rho = np.linspace(-len_diag, len_diag, 2*len_diag)
    
    # Create a 2D array of theta and rho values a.k.a accumulator
    accumulator = np.zeros((2*len(rho), len(theta)))
    
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    for x in range(height):
        for y in range(width):
            if image[x, y] != 0:
                for t in range(len(theta)):
                    r = x * np.cos(theta[t]) + y * np.sin(theta[t])
                    accumulator[int(r + len_diag), t] += image[x, y]
                    
    # Find the maximum value in the accumulator
    accumulator_max = np.max(accumulator)
    return accumulator, theta, rho, accumulator_max

def show_hough_plots(image,edges,masked):
    accumulator, theta, rho, accumulator_max = houghLine(edges)
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.title("Detected Edges")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(accumulator, cmap='gray')
    plt.axis('off')
    plt.title("Hough Space")
    fig.add_subplot(rows, columns, 4)
    plt.imshow(masked, cmap='gray')
    plt.axis('off')
    plt.title("Detected Lines")
    plt.show()
