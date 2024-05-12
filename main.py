import cv2 # OpenCV library https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import numpy as np


# Default click Coords for the click event also prevents crashing when there isnt a selected contour
click_x = 0  
click_y = 0


# User click position
def click_event(event, x, y, flags, param):
    global click_x, click_y 
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y


# Reading the original image w cv2
image = cv2.imread('sampletab2.jpg') # Reads as matrix
#original_image=image


# Gamma Correction to adjust brightness change the image contrast
gamma = 0.5 # Gamma value
corrected_image = np.power(image / 255.0, gamma) * 255.0 # Getting gamma corrected image (matrix array manipulation)
corrected_image = np.uint8(corrected_image) # Colour range (0-255)
cv2.imshow('Gamma Symbols', corrected_image) # Display the image
cv2.waitKey(0) # So the image doesn't immediatly close


gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) # Convert col to gray using inbuuilt fun bgr2gray

cv2.imshow('Gray Symbols', gray)
cv2.waitKey(0)

#Contrast---Dont Do
#equalized_image = cv2.equalizeHist(gray)

ret, thresholded_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/
processed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Processed_image Symbols', processed_image)
cv2.waitKey(0)


# Identify possible objects, https://learnopencv.com/contour-detection-using-opencv-python-c/
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # https://docs.opencv.org/4.x/da/d32/samples_2cpp_2contours2_8cpp-example.html

# Define min and max area only specfic symobols are detected
    # !! Need to do pre-processing later (resize the image) !!
min_area = 100
max_area = 500
filtered_contours = [contour for contour in contours if min_area < cv2.contourArea(contour) < max_area] # Gets the contours that fufil the min and max range


for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # drawing the contours as rectangles by using the starting position and width and height


cv2.imshow('Symbols', image)
cv2.setMouseCallback('Symbols', click_event)  # Click event call back
cv2.waitKey(0)


# Gets the selected contour (rectangle)
selected_contour_index = -1
for i, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    if click_x and click_y:
        if x < click_x < x + w and y < click_y < y + h:
            selected_contour_index = i
            break


# Gets the selected from the original img
if selected_contour_index != -1:
    x, y, w, h = cv2.boundingRect(filtered_contours[selected_contour_index])
    cropped_image = image[y:y+h, x:x+w]
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    
    save_path = 'cropped_image.jpg'
    cv2.imwrite(save_path, cropped_image)
    print(f"Cropped image saved as {save_path}")

cv2.destroyAllWindows()