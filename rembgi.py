import glob
import argparse
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import cv2
input_path = '/Users/anuj/Downloads/archive/intact/side/0707979392424_side.png'
output_path = 'out.png'


trackbar_value = 'Value'
window_name = 'Threshold Demo'
trackbar_img_value = "Img"


min_img_value = 0
max_img_value = 99


def Threshold_Demo(val):

    img_position = (cv2.getTrackbarPos(trackbar_img_value, window_name))
    filename = files[img_position]
    print(filename)
    f = np.fromfile(filename)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")

    open_cv_image = np.array(img)
    # Convert RGB to BGR
    src = open_cv_image[:, :, ::-1].copy()

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        src_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cv2.drawContours(src, [contours[0]], 0, (255, 0, 0), 5)
    cv2.drawContours(
        src, [np.int0((cv2.boxPoints(cv2.minAreaRect(contours[0]))))], 0, (0, 255, 0), 5)

    cv2.imshow("window_name", cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
    orig_img = cv2.imread(cv2.samples.findFile(filename))

    cv2.drawContours(orig_img, [contours[0]], 0, (255, 0, 0), 5)
    mask = np.zeros(orig_img.shape[:2], dtype="uint8")

    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(orig_img.shape[:2], dtype="uint8")
    roi_corners = np.array(
        [np.int0(cv2.boxPoints(cv2.minAreaRect(contours[0])))], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = orig_img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(orig_img, orig_img, mask=mask)
    cv2.drawContours(
        orig_img, [np.int0((cv2.boxPoints(cv2.minAreaRect(contours[0]))))], 0, (0, 255, 0), 5)

    cv2.imshow(window_name, cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    cv2.imshow("mask", cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    

parser = argparse.ArgumentParser(
    description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='00.jpg')
parser.add_argument('--folder', help='folder of input image.',
                    default='/Users/anuj/Downloads/scan_rotated/images/lva_passport/')
args = parser.parse_args()

files = [f for f in glob.glob(args.folder+"/*")]
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_img_value, window_name,
                   0, len(files), Threshold_Demo)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
# cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
