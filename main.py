import time

import cv2 as cv
import numpy as np
import pyautogui


# goal: identify white leds in the video stream, and use them to perform gestures
def main():
    print(cv.__version__)
    # get webcam
    cap = cv.VideoCapture(0)
    # create background subtractor
    fgbg = cv.createBackgroundSubtractorMOG2()
    # get kernel for morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # buffer of previous distances between leds
    distances = []
    # fps counting variables
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        # get frame
        _, frame = cap.read()
        # flip frame (webcam is mirrored)
        frame = cv.flip(frame, 1)
        # apply background subtraction
        fgmask = fgbg.apply(frame)
        # apply morphological operations (dilate, erode)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        # grayscale frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # find white pixels in gray with binary threshold
        _, thresh = cv.threshold(gray, 253, 255, cv.THRESH_BINARY)
        # find white pixels that are moving in fgmask
        merged = cv.bitwise_and(thresh, thresh, mask=fgmask)
        # find contours
        contours, _ = cv.findContours(merged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # adaptivley find contours around the lights
        if len(contours) > 0:
            # remove contours with small area
            contours = [c for c in contours if cv.contourArea(c) > 100]
            # get centers of contours
            centers = [cv.moments(contour) for contour in contours]
            # get x and y coordinates of centers
            centers = [
                (int(center["m10"] / center["m00"]), int(center["m01"] / center["m00"]))
                for center in centers
            ]
            # draw the contours and centers
            for i in range(len(contours)):
                # generate color from even or odd
                color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
                # cv.drawContours(frame, contours, i, (0, 255, 0), 2)
                cv.circle(frame, centers[i], 5, color, -1)
                # text with index above center
                cv.putText(
                    frame,
                    str(i),
                    centers[i],
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv.LINE_AA,
                )

            # use centers to calculate distance between leds
            if len(centers) > 1:
                distances = [
                    np.linalg.norm(np.array(centers[i]) - np.array(centers[i + 1]))
                    for i in range(0, len(centers) - 1, 2)
                ]
                # print distances
                print(distances)
                # is distance between leds is too small, click
                if len(distances) > 0 and min(distances) < 100:
                    pyautogui.click()
                    print("click")
            # move mouse
            if len(centers) > 0:
                # get center of all centers
                center = np.mean(centers, axis=0)

                # get screen size
                screen_width, screen_height = pyautogui.size()
                # get frame size
                frame_width, frame_height = frame.shape[1], frame.shape[0]
                # calculate mouse position
                mouse_x = int(center[0] / frame_width * screen_width)
                mouse_y = int(center[1] / frame_height * screen_height)
                # show center
                cv.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                # move mouse to center
                pyautogui.moveTo(mouse_x, mouse_y)

                print("move mouse to", mouse_x, mouse_y)
            #

        # fps
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv.putText(
            frame,
            "FPS: " + str(fps),
            (7, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (100, 255, 0),
            3,
            cv.LINE_AA,
        )

        # resize fgmask and thresh
        fgmask = cv.resize(fgmask, (0, 0), fx=0.5, fy=0.5)
        thresh = cv.resize(thresh, (0, 0), fx=0.5, fy=0.5)
        # concatenate fgmask and thresh
        debug = np.concatenate((fgmask, thresh), axis=1)

        # exit on q
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        # display frame
        cv.imshow("frame", frame)
        cv.imshow("debug", debug)
        # quit on ESC button
        if cv.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
