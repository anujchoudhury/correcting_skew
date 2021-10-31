from __future__ import print_function
import cv2 
import argparse
import numpy as np
max_value = 255

max_type = 4
max_binary_value = 255

low_k = 1  # slider start position
high_k = 42  # maximal slider position

trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'
trackbar_img_value="Img"
trackbar_blur_value="ksize"
trackbar_blur_value_msize="msize"



min_img_value=0
max_img_value=99

def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)    
    gblursize = cv2.getTrackbarPos(trackbar_blur_value, window_name)  # returns trackbar position    
    msize = cv2.getTrackbarPos(trackbar_blur_value_msize, window_name)  # returns trackbar position

    img_position= str(cv2.getTrackbarPos(trackbar_img_value, window_name)).zfill(2)
    src = cv2.imread(cv2.samples.findFile(args.folder+img_position+".jpg"))
    src_copy=src.copy()
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    # rus_internalpassport - 1,226,1,0
    str1="("+"threshold_type="+str(threshold_type)+",threshold_value="+str(threshold_value)+",gblursize="+str(gblursize)+",msize="+str(msize)+")"
    gblursize = 2*gblursize+1  # medianBlur allows only odd ksize values
    # Blures input image
    dst = cv2.GaussianBlur(dst, (gblursize,gblursize),0)  # source, kernel size    
    msize = 2*msize+1  # medianBlur allows only odd ksize values
    # Blurs input image
    dst = cv2.medianBlur(dst, msize)  # source, kernel size

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    # dst_color = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(src_copy,[np.int0((cv2.boxPoints(cv2.minAreaRect(contours[0]))))],0,(0,255,0),20)
    src_copy = cv2.putText(src_copy, str(str1), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, src_copy)




parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='00.jpg')
parser.add_argument('--folder', help='folder of input image.', default='/Users/anuj/Downloads/scan_rotated/images/lva_passport/')
args = parser.parse_args()

cv2.namedWindow(window_name)

cv2.createTrackbar(trackbar_type, window_name , 1, max_type, Threshold_Demo)
cv2.createTrackbar(trackbar_value, window_name , 245, max_value, Threshold_Demo)
cv2.createTrackbar(trackbar_blur_value, window_name, low_k, high_k, Threshold_Demo)
cv2.createTrackbar(trackbar_blur_value_msize, window_name , 0, max_img_value, Threshold_Demo)
cv2.createTrackbar(trackbar_img_value, window_name , 0, max_img_value, Threshold_Demo)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
# cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()