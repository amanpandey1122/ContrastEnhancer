import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Convert the image to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Calculate the histogram
    hist,bins=np.histogram(gray.flatten(),256,[0,256])

    # Calculate the cumulative distribution function (CDF)
    cdf=hist.cumsum()
    cdf_normalized=cdf*hist.max()/cdf.max()

    # Perform histogram equalization
    equalized=np.interp(gray.flatten(),bins[:-1],cdf_normalized)

    # Reshape the equalized values to the original image shape
    equalized=equalized.reshape(gray.shape)

    # Convert back to uint8
    equalized=np.uint8(equalized)

    return equalized

def main():
    # Load an image
    image=cv2.imread('your_image_path.jpg')

    if image is None:
        print("Error: Image not loaded.")
        return

    # Perform histogram equalization
    equalized_image=histogram_equalization(image)

    # Display the original and equalized images
    plt.subplot(121),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    plt.subplot(122),plt.imshow(equalized_image,cmap='gray'),plt.title('Equalized Image')
    plt.show()

if __name__=="__main__":
    main()
