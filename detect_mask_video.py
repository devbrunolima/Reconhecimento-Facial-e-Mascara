# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import NameFind
from twilio.rest import Client
from datetime import datetime, timedelta
from datetime import date

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml') # Classifier "frontal-face" Haar Cascade
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml') # Classifier "eye" Haar Cascade

recognise = cv2.face.EigenFaceRecognizer_create(15,4000)  # creating EIGEN FACE RECOGNISER
recognise.read("Recogniser/trainingDataEigan.xml")

valida = 'MASK'
nome = 'BRUNO'

ultima_chamada = datetime.today() - timedelta(days=1)

print (ultima_chamada)








                      # Load the training data

# -------------------------     START THE VIDEO FEED ------------------------------------------



def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)






# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1024)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		valida=label


		color = (255, 255, 255) if label == "Mask" else (255, 255, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame

		cv2.putText(frame, label, (startX, startY - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.99, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	#cap = cv2.imshow("Frame", frame)
    
	ID = 0
	#time.sleep(10.0)
	ret, img = ("Frame", frame)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB COLOR_BGR2GRAY
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))
		eyes = eye_cascade.detectMultiScale(gray_face)
		for (ex, ey, ew, eh) in eyes:
			ID, conf = recognise.predict(gray_face)
			NAME = NameFind.ID2Name(ID, conf)
			nome = NameFind.IDNameName(ID, conf)
			NameFind.DispID(x, y, w, h, NAME, gray)

	cv2.imshow('EigenFace Face Recognition System', gray)
	#cap = cv2.imshow("Frame", frame)


	'''if valida == "No Mask":
				data_e_hora_atuais = datetime.now()
				diferenca = data_e_hora_atuais - ultima_chamada


				if diferenca.total_seconds() >= 120:
					ultima_chamada = data_e_hora_atuais
					account_sid = 'AC**************************'
					auth_token = '3e************************8a'
					client = Client(account_sid, auth_token)


					message = client.messages.create(
						from_='whatsapp:+14155238886',
						body='Atenção!!! ' + nome + ' foi visto sem mascara em ' + str(data_e_hora_atuais),
						to='whatsapp:+5551********'
					)'''


    

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

'''
                                                          # Read the camera object
                                        # Convert the Camera to gray
                                 # Detect the faces and store the positions
        for (x, y, w, h) in faces:                                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
            # ------------ BY CONFIRMING THE EYES ARE INSIDE THE FACE BETTER FACE RECOGNITION IS GAINED ------------------
            gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))               # The Face is isolated and cropped
            eyes = eye_cascade.detectMultiScale(gray_face)
            for (ex, ey, ew, eh) in eyes:
                ID, conf = recognise.predict(gray_face)                              # Determine the ID of the photo
                NAME = NameFind.ID2Name(ID, conf)
                NameFind.DispID(x, y, w, h, NAME, gray)
        cv2.imshow('EigenFace Face Recognition System', gray)                       # Show the video
        if cv2.waitKey(1) & 0xFF == ord('q'):                                       # Quit if the key is Q
            break
    cap.release()
    cv2.destroyAllWindows()'''

