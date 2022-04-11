import cv2 as cv
import numpy as np

if __name__ == "__main__":
    capture = cv.VideoCapture(0)
    while True:
        # each milisecond the current frame from the camera is saved in the variable 'frame'
        ret, frame = capture.read()
        width = int(capture.get(3))     # the parameter 3 means that it will return the WIDTH of frame
        height = int(capture.get(4))

        # creating a new "frame"
        # np.zeros fills the whole array with 0 -> all black window
        image = np.zeros(frame.shape, np.uint8)

        # making the original frame smaller so that i can fit more of them in the window

        smaller_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        image[:height//2, :width//2] = smaller_frame   # top left( im iterating until half of the height
        # and half of the width (starting from the top left corner
        image[height // 2:, :width // 2] = cv.rotate(smaller_frame, cv.cv2.ROTATE_180)    # bottom left
        # height // 2: means start at half of the height and go to the end
        image[:height // 2:, width // 2:] = cv.rotate(smaller_frame, cv.cv2.ROTATE_180)     # top right
        image[height // 2:, width // 2:] = smaller_frame    # bottom right

        # show the current frame
        cv.imshow('frame', image)
        # cv.waitkey is gonna wait for one milisecond and then return the value of the key that we press
        # so if we press q the loop stops
        # ord returns the ASCII value of 'q'
        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()