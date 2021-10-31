from __future__ import print_function
import cv2 
import argparse
max_value = 255

max_type = 4
max_binary_value = 255

trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'
trackbar_img_value="Img"
trackbar_blur_value="ksize"
trackbar_blur_value_mszie="msize"

min_img_value=0
max_img_value=99

font = cv2.FONT_HERSHEY_SIMPLEX
def Threshold_Demo(val):
    img_position= str(cv2.getTrackbarPos(trackbar_img_value, window_name)).zfill(2)
    src = cv2.imread(cv2.samples.findFile(args.folder+img_position+".jpg"))
    # print(src.shape[:2])
    # Convert the image to Gray
    blue, green, red = cv2.split(src)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)    
    ksize = cv2.getTrackbarPos('ksize', window_name)  # returns trackbar position    
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    # dst=cv2.putText(dst, f"{threshold_value}", (120, 200), font, 5, (0,0,0), 3)
    print(threshold_type,threshold_value,ksize)
    
    ksize = cv2.getTrackbarPos(trackbar_blur_value, window_name)  # returns trackbar position
    ksize = 2*ksize+3  # medianBlur allows only odd ksize values

    # Blures input image
    dst = cv2.GaussianBlur(dst, (ksize,ksize),0)  # source, kernel size


    ksize = cv2.getTrackbarPos(trackbar_blur_value, window_name)  # returns trackbar position
    ksize = -2*ksize-1  # medianBlur allows only odd ksize values

    # Blures input image
    dst = cv2.medianBlur(dst, ksize)  # source, kernel size


    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    cnt = contours[0]
    dst_color = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dst_color,[cnt],0,(0,255,0),20)
    # wide = cv2.Canny(dst, 10, 200)
    # mid = cv2.Canny(dst, 30, 150)
    # tight = cv2.Canny(dst, 240, 250)
    # # show the output Canny edge maps
    # cv2.imshow("Wide Edge Map", wide)
    # cv2.imshow("Mid Edge Map", mid)
    # cv2.imshow("Tight Edge Map", tight)
    cv2.imshow(window_name, dst_color)




parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='00.jpg')
parser.add_argument('--folder', help='folder of input image.', default='/Users/anuj/Downloads/scan_rotated/images/lva_passport/')
args = parser.parse_args()

cv2.namedWindow(window_name)

cv2.createTrackbar(trackbar_type, window_name , 0, max_type, Threshold_Demo)
cv2.createTrackbar(trackbar_value, window_name , 245, max_value, Threshold_Demo)
cv2.createTrackbar(trackbar_img_value, window_name , 0, max_img_value, Threshold_Demo)
# Creates Trackbar with slider position and callback function
low_k = 1  # slider start position
high_k = 42  # maximal slider position
cv2.createTrackbar(trackbar_blur_value, window_name, low_k, high_k, Threshold_Demo)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
# cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()