import cv2 as cv

def equalizeImage(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img