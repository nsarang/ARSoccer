import cv2
import numpy as np
import time
import copy
import os
import glob
import pygame

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from kalman_filter import KalmanFilter
from tracker import Tracker
import queue, threading, subprocess
import keras
import keras.backend as K
import matplotlib.pylab as plt
from math import atan2, pi
from connection import DataReciever, DataSender


class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def release(self):
        self.cap.release()


def brush_circle(event, x, y, flags, param):
    global cornerPoints
    if len(cornerPoints) >= 4:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        cornerPoints.append([x, y])


def ptInRectangle(pt, rect):
    return (
        (pt[0] >= rect[0][0] and pt[1] >= rect[0][1])
        and (pt[0] <= rect[1][0] and pt[1] >= rect[1][1])
        and (pt[0] >= rect[2][0] and pt[1] <= rect[2][1])
        and (pt[0] <= rect[3][0] and pt[1] <= rect[3][1])
    )


def ptInCircle(pt, circle_cent, radius):
    return np.linalg.norm(pt - circle_cent) <= radius


def intersection_ball_object(rect_cords, circle_cent, radius):
    return (
        ptInRectangle(circle_cent, rect_cords)
        or (ptInCircle(rect_cords[0], circle_cent, radius))
        or (ptInCircle(rect_cords[1], circle_cent, radius))
        or (ptInCircle(rect_cords[2], circle_cent, radius))
        or (ptInCircle(rect_cords[3], circle_cent, radius))
    )


def is_valid_contour(x, y, w, h, thresh):
    center = np.array([[x + w / 2], [y + h / 2]])
    p = cornerPoints
    upscaledPts = [
        [p[0][0] - thresh, p[0][1] - thresh],
        [p[1][0] + thresh, p[1][1] - thresh],
        [p[2][0] - thresh, p[2][1] + thresh],
        [p[3][0] + thresh, p[3][1] + thresh],
    ]

    return (
        ptInRectangle(center, upscaledPts)
        and w >= blob_min_width
        and h >= blob_min_height
    )


def sum_vectors(dir_1, len_1, dir_2, len_2):
    dir_1 = np.asarray(dir_1)
    dir_2 = np.asarray(dir_2)

    if np.all(dir_1 == .0):
        return dir_2, len_2
    if np.all(dir_2 == .0):
        return dir_1, len_1
    
    print(dir_1, len_1)
    print(dir_2, len_2)

    vel_1 = len_1 * (dir_1 / (dir_1 ** 2).sum() ** 0.5)
    vel_2 = len_2 * (dir_2 / (dir_2 ** 2).sum() ** 0.5)
    result = vel_1 + vel_2

    res_len = (result ** 2).sum() ** 0.5
    res_unit = result / res_len
    print('res', res_unit, res_len)
    return res_unit, res_len


if __name__ == "__main__":
    if os.name == "posix":
        subprocess.Popen([os.getcwd() + "/Ar_Soccer_Demo1/V.app/Contents/MacOS/V"])
    else:
        subprocess.Popen([os.getcwd() + "/AR_win/AR_win.exe"])

    global cornerPoints
    font = cv2.FONT_HERSHEY_PLAIN
    centers = []
    cornerPoints = []
    FPS = 30

    input_dim = (384, 512, 3)
    K.set_learning_phase(0)  # 0 testing, 1 training mode

    field_width = 151
    field_height = 91
    radius = 4

    blob_min_width = 4
    blob_min_height = 4

    boundary_thresh = 15

    frame_start_time = None

    # Create object tracker
    tracker = Tracker(200, 3, 3, 1)

    hflip = 1
    vflip = 0
    if hflip and vflip:
        c = -1
    else:
        c = 0 if vflip else 1

    # Capture livestream
    cap = VideoCapture("http://192.168.43.1:8080/video")
    # cap = VideoCapture(0)

    ds = DataSender("127.0.0.1", 1835)
    dr = DataReciever("127.0.0.1", 1836)

    print("[INFO] loading model...")
    with open("context.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights("context.h5")
    print("[INFO] Loaded model from disk")

    cv2.namedWindow("Select")
    cv2.setMouseCallback("Select", brush_circle)
    # keep looping until the 'q' key is pressed
    while len(cornerPoints) < 4:
        start = time.time()
        frame = cap.read()
        # frame = cv2.flip(frame, flipCode=c)
        end = time.time()
        # print("[INFO] taking pic took " + str((end-start)*1000) + " ms")

        for (x, y) in cornerPoints:
            cv2.circle(frame, (x, y), 9, (255, 0, 0), -1)

        cv2.imshow("Select", frame)
        if cv2.waitKey(5) == 27:
            break

    # close all open windows
    cv2.destroyAllWindows()

    pts1 = np.float32(cornerPoints)
    pts_field = np.float32(
        [[0, 0], [field_width, 0], [0, field_height], [field_width, field_height]]
    )
    M = cv2.getPerspectiveTransform(pts1, pts_field)
    h_mat, mask = cv2.findHomography(pts1, pts_field, cv2.RANSAC)
    inv = np.linalg.inv(h_mat)

    clock = pygame.time.Clock()
    fr_counter = 0
    fr_shot = -5
    fr_id = None

    while True:
        centers = []
        cords = []
        start = time.time()
        frame = cap.read()
        # frame = cv2.flip(frame, flipCode=c)
        end = time.time()
        # print("[INFO] taking pic took " + str((end-start)*1000) + " ms")

        orig_frame = copy.copy(frame)

        new = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB) / 255.0
        img = cv2.resize(new, (input_dim[1], input_dim[0]), cv2.INTER_AREA)
        lr = cv2.resize(img, (input_dim[1] // 4, input_dim[0] // 4), cv2.INTER_AREA)

        start = time.time()
        heat_map = model.predict([img[np.newaxis, ...], lr[np.newaxis, ...]])[0]
        end = time.time()
        # print("[INFO] segmentation took " + str((end-start)*1000) + " ms")

        shoe_mask = np.zeros(shape=(48, 64), dtype=np.uint8)
        idx_sort = np.argsort(heat_map)[..., -2:]
        shoe_mask[np.any(idx_sort == 19, axis=-1)] = 255
        shoe_mask[np.any(idx_sort == 18, axis=-1)] = 255

        thresh, im_bw = cv2.threshold(shoe_mask, 128, 255, cv2.THRESH_BINARY)
        im_bw = cv2.resize(im_bw, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)

        kernel = np.ones((3, 3), np.uint8)
        im_bw = cv2.erode(im_bw, kernel, iterations=3)

        if os.name == "posix":
            _, contours, hierarchy = cv2.findContours(
                im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        else:
            contours, hierarchy = cv2.findContours(
                im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

        # Find centers of all detected objects
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print('blob', h, w)

            if not is_valid_contour(x, y, w, h, boundary_thresh):
                continue

            pts = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]).reshape(
                -1, 1, 2
            )
            newPts = cv2.perspectiveTransform(pts, h_mat).reshape(4, 2)

            center = np.array(
                [
                    [(newPts[0][0] + newPts[1][0]) / 2],
                    [(newPts[0][1] + newPts[2][1]) / 2],
                ]
            )
            centers.append(np.round(center))
            cords.append(newPts)

        if centers:
            tracker.update(centers, cords)

            for foot in tracker.tracks:
                cords = foot.cords
                ret = cv2.perspectiveTransform(cords.reshape(4, 1, 2), inv)
                ret = ret.reshape(4, 2)

                cv2.rectangle(
                    frame,
                    (ret[0][0], ret[0][1]),
                    (ret[-1][0], ret[-1][1]),
                    (255, 0, 255),
                    3,
                )

                if len(foot.trace) > 1:
                    for j in range(len(foot.trace) - 1):
                        # Draw trace line
                        x1 = foot.trace[j][0][0]
                        y1 = foot.trace[j][1][0]
                        x2 = foot.trace[j + 1][0][0]
                        y2 = foot.trace[j + 1][1][0]
                        ret = cv2.perspectiveTransform(
                            np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2), inv
                        )
                        ret = ret.reshape(2, 2)
                        # print(ret.shape)
                        # print(ret)
                        cv2.line(
                            frame,
                            (int(ret[0][0]), int(ret[0][1])),
                            (int(ret[1][0]), int(ret[1][1])),
                            (0, 255, 255),
                            2,
                        )

                ball_x, ball_y, ball_v, ball_dx, ball_dy = dr.get_stats()
                print("ball\t", ball_x, ball_y, ball_v, ball_dx, ball_dy)
                # print('ball', ball_x, ball_y)
                if intersection_ball_object(foot.cords, [ball_x, ball_y], radius):
                    # if (
                    #     ball_v <= 5
                    #     or (ball_x == 75 and ball_y == 45)
                    #     or (ball_dx == 0 and ball_dy == 0)
                    # ):
                    #     ball_v = 0
                    #     ball_dx = ball_dy = np.random.random()
                    if fr_counter - fr_shot <= 2 and foot.track_id == fr_id:
                        continue
                    fr_shot = fr_counter
                    fr_id = foot.track_id

                    if len(foot.trace) <= 1:
                        d_x = ball_dx * np.random.uniform(-2, 2)
                        d_y = ball_dy * np.random.uniform(-2, 2)
                        print("sh   0")

                    elif len(foot.trace) == 2:
                        print("sh   1")
                        d_x = foot.trace[-1][0][0] - foot.trace[-2][0][0]
                        d_y = foot.trace[-1][1][0] - foot.trace[-2][1][0]

                    else:
                        print("sh   2\t %d" % len(foot.trace))
                        d_x = foot.trace[-1][0][0] - foot.trace[-3][0][0]
                        d_y = foot.trace[-1][1][0] - foot.trace[-3][1][0]

                    if d_x == 0 and d_y == 0:
                        d_x = d_y = np.random.random()
                    velocity = np.sqrt(d_x ** 2 + d_y ** 2) * 10
                    velocity = min(100, max(5, velocity))
                    res_unit, res_len = sum_vectors(
                        (d_x, d_y), velocity, (-ball_dx, -ball_dy), ball_v / 2
                    )

                    angle = atan2(*res_unit) * 180.0 / pi
                    if angle < 0:
                        angle += 360
                    new_velocity = min(100, max(20, res_len))
                    print("2-velo-angle\t", velocity, angle)
                    ds.send(new_velocity, angle)
        clock.tick(FPS)
        fr_counter +=1


        # Display all images
        ball_x, ball_y, ball_v, ball_dx, ball_dy = dr.get_stats()

        ball_mask = np.zeros(shape=(field_height, field_width), dtype=np.uint8)
        cv2.line(
            ball_mask,
            (field_width // 2, 0),
            (field_width // 2, field_height),
            (0, 255, 0),
            2,
        )
        cv2.circle(ball_mask, (ball_x, ball_y), 4, (255, 0, 0), -1)
        pitch = cv2.resize(
            ball_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST
        )

        cv2.imshow("ball", pitch)
        cv2.imshow("mask", im_bw)
        cv2.imshow("original", frame)
        # cv2.imshow ('opening/closing', closing)
        # cv2.imshow ('background subtraction', fgmask)

        # Quit when escape key pressed
        if cv2.waitKey(5) == 27:
            break

        # Sleep to keep video speed consistent
        # time.sleep(1.0 / FPS)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # remove all speeding_*.png images created in runtime
    # for file in glob.glob('speeding_*.png'):
    # os.remove(file)

