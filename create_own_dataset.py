import time
import cv2
import imutils
from imutils.video import VideoStream

from helper import make_if_not_exist

DATA_ROOT = 'dataset'
SUBJECT = 'phuc'
MSET = 'train'  # train/test
IM_WIDTH = 640
IM_HEIGHT = 480

if __name__ == '__main__':
    print('[INFO] Checking directory...')
    # create dataset directory
    DATA_DIR = '{}/{}/{}'.format(DATA_ROOT, MSET, SUBJECT)
    make_if_not_exist(DATA_DIR)

    print('[INFO] Starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(1.)
    total = 0

    # loop over the frames from the video stream
    while True:
        # grab frame
        frame = vs.read()
        frame = imutils.resize(frame, IM_WIDTH, IM_HEIGHT)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            # ESC press
            break

        if key == 99:
            # 'c' press
            im_name = '{}/{}.png'.format(DATA_DIR, total + 1)
            cv2.imwrite(im_name, frame)
            total += 1

    print('[INFO] {} face images stored'.format(total))
    print('[INFO] Finish...')
    cv2.destroyAllWindows()
    vs.stop()
