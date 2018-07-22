import numpy as np
import imutils
import cv2

from imutils import perspective

BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 15000

def find_cards(thresh_image):
    
    # Find contours
    dummy, cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If there are no contours, do nothing
    if len(cnts) == 0:
        return []
    
    cards = []

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts)):
        size = cv2.contourArea(cnts[i])
        peri = cv2.arcLength(cnts[i],True)
        approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (len(approx) == 4)) and (hier[0][i][3] == -1):
            cards.append(cnts[i])

    return cards


def preprocess_card(contour, image, ratio):
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, 0.01*peri, True)
    warp = perspective.four_point_transform(image, approx.reshape(4, 2) * ratio)

    if (warp.shape[0] < warp.shape[1]):
        warp = imutils.rotate_bound(warp, angle=90)

    warp = cv2.resize(warp, (450, 700)) # make it into an actual card shape
    
    # Get the top left corner of the card
    corner = warp[0:165, 0:165]
    corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
    cv2.imshow("Output", corner_zoom)
    cv2.waitKey(0)

    gray = cv2.cvtColor(corner_zoom, cv2.COLOR_BGR2GRAY)

    img_w, img_h = np.shape(corner_zoom)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    retval, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    dummy, cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)

    # because the shapes might be slightly different, lets try and make a smaller section
    # to focus on when reading the marking

    for i in range(len(cnts)):
        c = cnts[i]

        size = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.005*peri, True)

        if (size > 10000 and len(approx) > 4 and len(approx) < 8):

            # We need three points to make a new rectangle.
            # 1. middle point, lower and rightmost
            # 2. Furthest point right, and lowest
            # 3. Furthest point south, and right most
            lm = [];
            rm = [];
            bm = [];

            for i in range(len(approx)):

                pt = approx[i][0]
                print pt
                if pt[0] > 650:
                    if len(bm) == 0:
                        bm = pt
                    elif len(bm) > 0 and bm[1] < pt[1]:
                        bm = pt
                    bm[0] = corner_zoom.shape[0]

                elif pt[1] > 650:
                    print pt
                    if len(rm) == 0:
                        rm = pt
                    elif len(rm) > 0 and rm[0] < pt[0]:
                        rm = pt
                    rm[1] = corner_zoom.shape[1]

                else:
                    if len(lm) == 0:
                        lm = pt
                    elif lm[0] < pt[0]:
                        lm = pt
                    elif lm[1] < pt[1]:
                        lm = pt

            
            break;

    cv2.circle(corner_zoom, tuple(lm), 5, (240, 0, 159), -1)
    cv2.circle(corner_zoom, tuple(bm), 5, (240, 0, 159), -1)
    cv2.circle(corner_zoom, tuple(rm), 5, (240, 0, 159), -1)

    cv2.imshow("Image", corner_zoom)
    cv2.waitKey(0)

    coords = [tuple(lm), tuple(rm), tuple(bm), tuple([corner_zoom.shape[0], corner_zoom.shape[1]])];
    coords = np.array(coords, dtype = "float32")
    warp = perspective.four_point_transform(corner_zoom, coords)

    # image = thresh[292:659, 262:659]
    cv2.imshow("Image", warp)
    cv2.waitKey(0)



##########################################################################################################################################

# Actual Workings

##########################################################################################################################################

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
original_image = cv2.imread("card2.jpg")
ratio = original_image.shape[1] / float(2000)
image = imutils.resize(original_image, width=2000)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
img_w, img_h = np.shape(image)[:2]
bkg_level = gray[int(img_h/100)][int(img_w/2)]
thresh_level = bkg_level + BKG_THRESH

retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

cards = find_cards(thresh)

if len(cards) != 0:
    # For each contour detected:
    for i in range(len(cards)):
        preprocess_card(cards[i], original_image, ratio)


#cv2.imshow("Output", image)
#cv2.waitKey(0)