from imutils import face_utils
import cv2
import numpy as np
import glob

import cv2
import numpy as np
from PIL import Image
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
def four_point_transform(image, pts):
    
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    if maxHeight>maxWidth:
        warped = np.rot90(warped)
    
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    dets, scores, idx = detector.run(gray, 1, -1)
    # for i, d in enumerate(dets):
    if(len(dets) > 0):
        print("Detection {}, score: {}, face_type:{}".format(dets[0], scores[0], idx[0]))
    if(len(dets)==0 or scores[0]<1):
        warped = np.rot90(warped)
        warped = np.rot90(warped)
    return warped

def get_3channel(gray):

    img = np.zeros((*gray.shape, 3), dtype='uint8')
    for i in range(3):
        img[:,:, i] = gray

    return img

def inplane_correct():

    for image_path in image_paths:
        print(image_path)
        img = cv2.imread(image_path)
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = get_3channel(mask)
        mask = cv2.bitwise_not(mask)

        (thresh, bw_img) = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if debug:
            out = bw_img.copy()
            out = cv2.resize(out, None, fx=0.25, fy=0.25)
            cv2.imshow('bw_img ',out)
            cv2.waitKey(-1)

        contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        warped = four_point_transform(img, box)
        cv2.imshow('warped ',warped)
        cv2.waitKey(-1)

if __name__ == "__main__":
    
    debug = 0
    image_paths = glob.glob("images/*")
    inplane_correct()
