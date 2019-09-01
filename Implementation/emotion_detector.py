#Import necessary packages
import cv2, time, imutils, numpy as np, tensorflow as tf, dlib, os, joblib
from imutils.video import FileVideoStream, VideoStream
from util import FACIAL_LANDMARKS_IDXS
from util import rect_to_bb, shape_to_np, extract_head_rotation
from util import analytics

# Job loader
clf = joblib.load('models/svm_classifier_custom_50_13.joblib.pkl')
predictor = dlib.shape_predictor('util/shape_predictor_68_face_landmarks.dat')
# Set emotion names
emotions = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
# Initialize lists
dataToPredict = []
actualObjects = []
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('models/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)


# Set image as input tensor and defince output tensors
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Initialize multi tracker
trackers = cv2.MultiTracker_create()


def get_hog(image, rect):
    # extract 68 facial landmarks
    shape = predictor(image, rect)
    shape = shape_to_np(shape)
    (x, y, w, l) = rect_to_bb(rect)
    # Check if image is valid
    if rect.left() > 0 and rect.top() > 0 and x>0 and y > 0:
        actualObjects.append((x, y, w, l))
        hog = cv2.HOGDescriptor((96, 96), (32, 32), (16, 16), (8, 8), 6)
        hogVector = []
        # Get all ROI regions and create feature vector
        for (name, (i, j)) in list(FACIAL_LANDMARKS_IDXS.items()):
            for (x, y) in shape[i:j]:
                if x <= 0 or y <= 0:
                    continue
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            if x <= 0 or y <= 0:
                continue
            (x, y, w, l) = cv2.boundingRect(np.array([shape[i:j]]))
            (ymax, xmax) = frame.shape[:2]
            if x <= 0 or y <= 0 or y + l >= ymax or x + w >= xmax:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + l), (0, 255, 0), 3)
            roi = cv2.resize(image[y:y + l, x:x + w], (96, 96))
            h = hog.compute(roi)
            h = np.squeeze(h)
            h = list(h)
            hogVector += h
        if len(hogVector) > 0:
            dataToPredict.append(hogVector)


# Face detection with the loaded tensorflow model
def face_detect():
    # Expand frame
    frame_expanded = np.expand_dims(frame, axis=0)
    # Perform the detection by running the model
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: frame_expanded})
    index = -1
    (h, w) = frame.shape[:2]
    output = []
    # Iterate over all bounding boxes
    for confidence in np.nditer(scores):
        index += 1
        # Skip low-confidence detections
        if confidence < 0.4:
            continue
        # Scale bounding boxes to frame size
        box = boxes[0][index] * np.array([h, w, h, w])
        (Y1, X1, Y2, X2) = box.astype("int")
        # Append detected bounding box to return array
        output.append((X1, Y1, X2 - X1, Y2 - Y1))
    return output


# Motion tracking with KCF
def motion_tracker():
    # Run algorithm for current frame
    (success, boxesX) = trackers.update(frame)
    output = []
    # If all bounding boxes are detected successfully continue,
    # otherwise reinitialize face detection
    if success:
        for boxIter in boxesX:
            (X, Y, W, H) = [int(v) for v in boxIter]
            output.append((X, Y, W, H))
    else:
        refresh_trackers()
    return output


# Helper method for reinitializing the face detection
def refresh_trackers():
    faces = face_detect()
    global trackers
    trackers = cv2.MultiTracker_create()
    if faces is not None:
        for face in faces:
            trackers.add(cv2.TrackerKCF_create(), frame, face)


# Camera Videostream
vs = VideoStream(src=0).start()
time.sleep(1)
# vs = FileVideoStream('benchmark/4FrontFacing.mp4').start()
# out = cv2.VideoWriter('emotion_detector_KCF.avi',
# cv2.VideoWriter_fourcc('M','J','P','G'), 30, (400, 225))

# Set recording time
starttime = time.time()


# Main loop over frames
while True:
    start = time.time()
    read = vs.read()
    # Cancel if no more frames are detected (relevant for reading video files)
    if read is None:
        break
    # Set Resolution for tracking algorithm
    frame = imutils.resize(read, height=400, width=400)
    # Get new bounding boxes
    boxes = motion_tracker()
    if boxes is not None:
        # Make sure the face detector runs every 10 sec (max)
        if time.time() - starttime > 10 or len(boxes) == 0:
            refresh_trackers()
            starttime = time.time()
        for box in boxes:
            # Draw face bounding boxes on frame
            (startX, startY, width, height) = box
            get_hog(frame, dlib.rectangle(startX, startY, startX + width, startY + height))
    else:
        # Refresh trackers if no bounding box is detected
        # (This makes sure the tracker is not executed when there is no face detected)
        refresh_trackers()
        starttime = time.time()

    if len(dataToPredict) > 0:
        # Predict emotion
        results = clf.predict(np.array(dataToPredict))
        for i in range(0, len(dataToPredict)):
            (x, y, w, l) = actualObjects[i]
            cv2.putText(frame, emotions[results[i]], (x + w//2 + 10, (y + l) + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    dataToPredict = []
    actualObjects = []
    # Save measurement to database
    analytics.persist('emotion_detector_KCF', time.time() - start, len(boxes))
    cv2.imshow("Emotion_detector", frame)
    # out.write(frame)
    # Stop program when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Final cleanup
# out.release()
cv2.destroyAllWindows()
vs.stop()
analytics.close()
