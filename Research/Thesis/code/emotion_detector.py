#Import necessary packages
import time, cv2, imutils, numpy as np, tensorflow as tf
from imutils.video import FileVideoStream, VideoStream
from util import analytics


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


# Face detection with the loaded tensorflow model
def face_detect():
    # Expand frame
    frame_expanded = np.expand_dims(frame, axis=0)
    # Perform the detection by running the model
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
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

# Read video from camera
vs = VideoStream(src=0).start()
time.sleep(1)
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
            cv2.rectangle(frame, (startX, startY),
                          (startX+width,startY+height), (0, 0, 255), 2)
    else:
        # Refresh trackers if no bounding box is detected
        # (This makes sure the tracker is not executed when there is no face detected)
        refresh_trackers()
        starttime = time.time()
    # Save measurement to database
    analytics.persist('emotion_detector_KCF', time.time() - start, len(boxes))
    cv2.imshow("Emotion_detector", frame)
    # Stop program when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Final cleanup
cv2.destroyAllWindows()
vs.stop()
analytics.close()