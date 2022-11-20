import threading
import time

import cv2 as cv
import numpy as np
import pyautogui


def draw_centers(frame, centers):
    for center in centers:
        cv.circle(frame, center, 5, (0, 0, 255), -1)
        cv.putText(
            frame, str(center), center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )


def connect_centers(frame, centers, distance):
    if len(centers) == 2:
        cv.line(frame, centers[0], centers[1], (0, 255, 0), 2)
        cv.putText(
            frame,
            str(distance),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )


frame_width, frame_height = 1280, 720
frame = np.zeros((frame_height, frame_width, 3), np.uint8)
buffer = np.zeros((frame_height, frame_width, 3), np.uint8)
prev_buffer = buffer
mouse_x, mouse_y = pyautogui.position()
running = True
clicked = False


def mouse():
    while running:
        if pyautogui.position() != (mouse_x, mouse_y):
            pyautogui.moveTo(mouse_x, mouse_y)
        global clicked
        if clicked:
            pyautogui.click()
            clicked = False


def camera():
    pyautogui.FAILSAFE = False
    screen_width, screen_height = pyautogui.size()

    fgbg = cv.createBackgroundSubtractorMOG2()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    distances = []
    prev_frame_time = 0
    new_frame_time = 0

    while running:
        global frame, buffer, prev_buffer, mouse_x, mouse_y, writing
        temp_frame = frame
        temp_buffer = np.zeros((frame_height, frame_width, 3), np.uint8)
        fgmask = fgbg.apply(temp_frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)

        gray = cv.cvtColor(temp_frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
        merged = cv.bitwise_and(thresh, thresh, mask=fgmask)

        contours, _ = cv.findContours(merged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 0:
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]

            centers = [cv.moments(contour) for contour in contours]
            centers = [
                (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                for M in centers
                if M["m00"] != 0
            ]

            distances = [
                np.linalg.norm(np.array(centers[i]) - np.array(centers[i + 1]))
                for i in range(0, len(centers) - 1, 2)
            ]

            draw_centers(temp_buffer, centers)
            connect_centers(temp_buffer, centers, distances)
            if len(distances) > 0 and distances[0] < 50:
                cv.putText(
                    buffer,
                    "Click",
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                global clicked
                clicked = True
            if len(centers) > 0:
                center = np.mean(centers, axis=0)
                global mouse_x, mouse_y
                mouse_x = int(center[0] / frame_width * screen_width)
                mouse_y = int(center[1] / frame_height * screen_height)
                cv.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv.putText(
            temp_buffer,
            "FPS: " + str(fps),
            (7, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (100, 255, 0),
            3,
            cv.LINE_AA,
        )
        buffer = temp_buffer


cap = cv.VideoCapture(0)
camera_thread = threading.Thread(target=camera).start()
mouse_thread = threading.Thread(target=mouse).start()
while True:
    frame = cv.flip(cap.read()[1], 1)
    # make black areas in buffer transparent
    frame = cv.addWeighted(frame, 1, buffer, 0.5, 0)

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        running = False
        break
mouse_thread.join()
camera_thread.join()
cap.release()
cv.destroyAllWindows()
