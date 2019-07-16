import cv2
import numpy as np
import time
import copy
import os
import glob
from datetime import datetime
import imutils

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from kalman_filter import KalmanFilter
from tracker import Tracker

import queue, threading
import matplotlib.pylab as plt
from skimage.transform import rescale, resize
from skimage.io import imread
import keras
import keras.backend as K
import matplotlib.pylab as plt
from math import atan


# y = model.predict([img[np.newaxis,...], lr[np.newaxis,...]])

# plt.imshow(y[0,...,-1])
# plt.show()


# bufferless VideoCapture
class VideoCapture:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()


  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


if __name__ == '__main__':
	# The one I first used for testing; after staring at it so much, I've grown attached to this road :3
	# the_og_base_url = 'http://wzmedia.dot.ca.gov:1935/D3/89_rampart.stream/'

	# BASE_URL = 'http://wzmedia.dot.ca.gov:1935/D3/80_whitmore_grade.stream/'
	# FPS = 30
	'''
		Distance to line in road: ~0.025 miles
	'''
	ROAD_DIST_MILES = 0.025

	'''
		Speed limit of urban freeways in California (50-65 MPH)
	'''
	HIGHWAY_SPEED_LIMIT = 65

	# Initial background subtractor and text font
	# fgbg = cv2.createBackgroundSubtractorMOG2()
	font = cv2.FONT_HERSHEY_PLAIN

	centers = [] 

	# y-cooridinate for speed detection line

	input_dim = (384, 512, 3)
	K.set_learning_phase(0) # 0 testing, 1 training mode


	Y_THRESH = 240

	blob_min_width_far = 1
	blob_min_height_far = 1

	# blob_min_width_near = 80
	# blob_min_height_near = 40

	frame_start_time = None

	# Create object tracker
	tracker = Tracker(200, 3, 3, 1)

	# Capture livestream
	# cap = cv2.VideoCapture (BASE_URL + 'playlist.m3u8')
	cap = VideoCapture("http://192.168.43.1:8080/video")
	# cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
	# cap = VideoCapture(0)

	print("[INFO] loading model...")
	# net = cv2.dnn.readNet('frozen_model.pb')
	# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
	# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	model = keras.models.load_model('weights.39-0.56.hdf5')

	# # serialize model to JSON
	# model_json = model.to_json()
	# with open("context.json", "w") as json_file:
	#     json_file.write(model_json)
	# # serialize weights to HDF5
	# model.save_weights("context.h5")
	# with open('context.json', 'r') as json_file:
	# 	loaded_model_json = json_file.read()
	# model = keras.models.model_from_json(loaded_model_json)
	# # load weights into new model
	# model.load_weights("context.h5")
	print("Loaded model from disk")


	while True:
		centers = []
		frame_start_time = datetime.utcnow()
		frame = cap.read()
		

		# pts1 = np.float32([[285, 330],[206, 460],[550, 330],[620, 460]])
		# pts2 = np.float32([[0,0],[0, 300],[400, 0],[300, 400]])
		# M = cv2.getPerspectiveTransform(pts1,pts2)
		# h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
		# dst = cv2.warpPerspective(img,h,(400, 300))

		# frame = dst

		orig_frame = copy.copy(frame)

		#  Draw line used for speed detection
		# cv2.line(frame,(0, Y_THRESH),(640, Y_THRESH),(255,0,0),2)


		# Convert frame to grayscale and perform background subtraction
		# gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
		# fgmask = fgbg.apply (gray)

		# Perform some Morphological operations to remove noise
		# kernel = np.ones((3,3),np.uint8)
		# kernel_dilate = np.ones((4,4),np.uint8)
		# opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		# dilation = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_dilate)

		# erosion = cv2.erode(fgmask, kernel, iterations=1)
		# dilation = cv2.dilate(erosion, kernel_dilate, iterations=1)
		# opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=1)
		# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)
		# start = time.time()
		new = cv2.cvtColor (orig_frame, cv2.COLOR_BGR2RGB)
		# end = time.time()
		# print("[INFO] convert took " + str((end-start)*1000) + " ms")
		# start = time.time()
		img = cv2.resize(new, (input_dim[1], input_dim[0]), cv2.INTER_AREA)
		lr = cv2.resize(img, (input_dim[1]//4, input_dim[0]//4), cv2.INTER_AREA)
		# end = time.time()
		# print("[INFO] resize took " + str((end-start)*1000) + " ms")

		
		start = time.time()
		heat_map = model.predict([img[np.newaxis,...], lr[np.newaxis,...]])[0]
		end = time.time()
		print("[INFO] segmentation took " + str((end-start)*1000) + " ms")
		
		shoe_mask = np.zeros(shape=(48, 64), dtype=np.uint8)
		in_max = np.argmax(heat_map, axis=-1)
		shoe_mask[in_max == 19] = 255
		shoe_mask[in_max == 18] = 255



		# orig_frame = copy.copy(frame)
		# print(frame.shape)
		# start = time.time()
		(thresh, im_bw) = cv2.threshold(shoe_mask, 128, 255, cv2.THRESH_BINARY)
		im_bw = cv2.resize(im_bw,(frame.shape[1],frame.shape[0]), cv2.INTER_NEAREST)
		# end = time.time()
		# print("[INFO] final resize took " + str((end-start)*1000) + " ms")
		plt.ion()
		plt.show()
		plt.imshow(im_bw)


		_, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# print(len(contours))
		# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		# cv2.drawContours(frame, contours, -1, (0,255,0), 3)
		# print(frame.shape)

		# _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Find centers of all detected objects
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			
			# if (w >= blob_min_width_far and h >= blob_min_height_far):
				# w <= blob_min_width_near and h <= blob_min_height_near):
			center = np.array ([[x+w/2], [y+h/2]])
			centers.append(np.round(center))
			# print(center)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)
			# print((x, y), (x+w, y+h))
			# cv2.line(frame,(0, 30),(640, 30),(255,0,0),2)

			# if w >= blob_min_width_far and h >= blob_min_height_far:
			# 	center = np.array ([[x+w/2], [y+h/2]])
			# 	centers.append(np.round(center))

			# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if centers:
			tracker.update(centers)

			for vehicle in tracker.tracks:
				if len(vehicle.trace) > 1:
					for j in range(len(vehicle.trace)-1):
                        # Draw trace line
						x1 = vehicle.trace[j][0][0]
						y1 = vehicle.trace[j][1][0]
						x2 = vehicle.trace[j+1][0][0]
						y2 = vehicle.trace[j+1][1][0]

						cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

					# try:
						'''
							TODO: account for load lag
						'''

					d_x = vehicle.trace[-1][0][0] - vehicle.trace[-2][0][0]
					d_y = vehicle.trace[-1][1][0] - vehicle.trace[-2][1][0]

					angle = atan(-d_y/d_x)
					velocity = np.sqrt(d_x**2 + d_y**2)
					# Check if tracked object has reached the speed detection line
					# if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
						# cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
						# vehicle.passed = True

					load_lag = (datetime.utcnow() - frame_start_time).total_seconds()

					time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
					time_dur /= 60
					time_dur /= 60

					
					vehicle.mph = ROAD_DIST_MILES / time_dur

							# If calculated speed exceeds speed limit, save an image of speeding car
							# if vehicle.mph > HIGHWAY_SPEED_LIMIT:
								# print ('UH OH, SPEEDING!')
								# cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
								# cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
								# cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
								# print ('FILE SAVED!')

						# if vehicle.passed:
							# Display speed if available
							# cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
						# else:
							# Otherwise, just show tracking id
							# cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
					# except:
					# 	pass


		# Display all images
		cv2.imshow ('original', frame)
		# box = cv2.selectROI("original", frame, fromCenter=False,
		# 	showCrosshair=True)
		# print(box)
		# print(type(shoe_mask))
		# cv2.imshow('mask', shoe_mask)
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
	for file in glob.glob('speeding_*.png'):
		os.remove(file)