from __future__ import print_function
import cv2
import argparse
import numpy as np
import glob


max_value = 250

default_k = 0  # slider start position
high_k = 16  # maximal slider position

trackbar_type = 'Edge Detection Type'
max_type_edge = 3
edge_ksize_max = 7
edge_ksize_param = 'Edge Ksize'
trackbar_value = 'edge_param'
trackbar_value_1 = 'edge_param_1'

window_name = 'Threshold Demo'
trackbar_img_value = "Img"
trackbar_blur_value_gaussian = "ksize"
trackbar_blur_value_msize = "msize"

minLineLength_value = "minLineLength"
maxLineGap_value = "maxLineGap"

min_img_value = 0
max_img_value = 99
max_blurvalue = 30
Thetaval = "Theta"
maxThetaval = 250


def Threshold_Demo(val):
    # 0: sobelx
    # 1: sobely
    # 2: sobelxy
    # 3: canny
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    threshold_value_1 = cv2.getTrackbarPos(trackbar_value_1, window_name)
    ksize_param = cv2.getTrackbarPos(edge_ksize_param, window_name)
    gblursize = cv2.getTrackbarPos(trackbar_blur_value_gaussian, window_name)
    msize = cv2.getTrackbarPos(trackbar_blur_value_msize, window_name)
    ksize_param = 2*ksize_param+1
    img_position = (cv2.getTrackbarPos(trackbar_img_value, window_name))
    filename = files[img_position]
    src = cv2.imread(cv2.samples.findFile(filename))
    src_copy = src.copy()

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = src_gray.copy()

    if(threshold_type == 0):
        dst = cv2.Sobel(dst, 5*threshold_value, dx=1, dy=0, ksize=ksize_param)
    elif(threshold_type == 1):
        dst = cv2.Sobel(dst, 5*threshold_value, dx=0, dy=1, ksize=ksize_param)
    elif(threshold_type == 2):
        dst = cv2.Sobel(dst, 5*threshold_value, dx=1, dy=1, ksize=ksize_param)
    elif(threshold_type == 3):
        dst = cv2.Canny(dst, threshold1=5*threshold_value,
                        threshold2=5*threshold_value_1*threshold_value, apertureSize=ksize_param)

    minLineLength = cv2.getTrackbarPos(minLineLength_value, window_name)
    maxLineGap = cv2.getTrackbarPos(maxLineGap_value, window_name)
    theta = cv2.getTrackbarPos(Thetaval, window_name)
    theta=theta*2
    str1 = "("+"threshold_type="+str(threshold_type)+",threshold_value=" + \
        str(threshold_value)+",threshold_value_1=" + \
        str(threshold_value_1)+",gblursize=" + str(gblursize)
    str2 = "msize="+str(msize)+",img_position=" + str(img_position)+",ksize_param=" + str(
        ksize_param)+",minLineLength=" + str(minLineLength)+",theta=" + str(theta)+")"
    gblursize = 2*gblursize+1  # medianBlur allows only odd ksize values
    # Blurs input image
    dst = cv2.GaussianBlur(dst, (gblursize, gblursize),
                           0)  # source, kernel size
    msize = 2*msize+1  # medianBlur allows only odd ksize values
    # Blurs input image
    dst = cv2.medianBlur(dst, msize)  # source, kernel size
    cv2.imshow(window_name, dst)
    # cv2.imshow("window_name", src_copy)
    src_new_cpy = src_copy.copy()
    dst = np.asarray(dst, dtype="uint8")

    lines = cv2.HoughLinesP(dst, rho=1, theta=(10/theta) * np.pi/(36), threshold=100,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    for item in lines:
        for x1, y1, x2, y2 in item:
            cv2.line(src_new_cpy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    src_new_cpy = cv2.putText(src_new_cpy, str(str1), (30, 45),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    src_new_cpy = cv2.putText(src_new_cpy, str(str2), (30, 65),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("edge_window", src_new_cpy)


parser = argparse.ArgumentParser(
    description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='00.jpg')
parser.add_argument('--folder', help='folder of input image.',
                    default='/Users/anuj/Downloads/scan_rotated/images/lva_passport/')
args = parser.parse_args()

files = [f for f in glob.glob(args.folder+"/*")]
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name, 3,
                   max_type_edge, Threshold_Demo)
cv2.createTrackbar(edge_ksize_param, window_name, 3,
                   edge_ksize_max, Threshold_Demo)

cv2.createTrackbar(trackbar_value, window_name, 100, max_value, Threshold_Demo)
cv2.createTrackbar(trackbar_value_1, window_name,
                   100, max_value, Threshold_Demo)

cv2.createTrackbar(trackbar_blur_value_gaussian, window_name,
                   default_k, high_k, Threshold_Demo)
cv2.createTrackbar(trackbar_blur_value_msize, window_name,
                   0, max_blurvalue, Threshold_Demo)

cv2.createTrackbar(trackbar_img_value, window_name,
                   0, len(files), Threshold_Demo)
cv2.createTrackbar(minLineLength_value, window_name, 60, 200, Threshold_Demo)
cv2.createTrackbar(maxLineGap_value, window_name, 20, 80, Threshold_Demo)

cv2.createTrackbar(Thetaval, window_name, 10, maxThetaval, Threshold_Demo)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
# cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
