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
    

def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal
def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)

    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def show_img(img):
    while True:
        cv2.imshow('face_alignment_app', img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()


def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal


def rotate_opencv(img, nose_center, angle):
    M = cv2.getRotationMatrix2D(nose_center, angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return rotated



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    if(len(dets)==0 or scores[0]<0):
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
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        # # loop over the face detections
        # for (i, rect) in enumerate(rects):
        #     # determine the facial landmarks for the face region, then
        #     # convert the facial landmark (x, y)-coordinates to a NumPy
        #     # array
        #     shape = predictor(gray, rect)
        #     shape = face_utils.shape_to_np(shape)
        #     # convert dlib's rectangle to a OpenCV-style bounding box
        #     # [i.e., (x, y, w, h)], then draw the face bounding box
        #     (x, y, w, h) = face_utils.rect_to_bb(rect)
        #     cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     # show the face number
        #     cv2.putText(warped, "Face #{}".format(i + 1), (x - 10, y - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     # loop over the (x, y)-coordinates for the facial landmarks
        #     # and draw them on the image
        #     for (x, y) in shape:
        #         cv2.circle(warped, (x, y), 1, (0, 0, 255), -1)
        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = predictor(gray, rect)
                shape = shape_to_normal(shape)
                nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
                center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                center_pred = (int((x + w) / 2), int((y + y) / 2))
                dets, scores, idx = detector.run(warped, 1, -1)

                cv2.circle(warped, (left_eye), 5, (0,0,255), 10)
                cv2.circle(warped, (right_eye), 5, (0,0,255), 10)
                cv2.circle(warped, (nose), 5, (0,255,0), 10)
                cv2.imshow("Output", warped)
                cv2.waitKey(0)
    
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", warped)
        cv2.waitKey(0)
        cv2.imshow('warped ',warped)
        cv2.waitKey(-1)

if __name__ == "__main__":

    debug = 0
    image_paths = glob.glob("images/*")
    inplane_correct()