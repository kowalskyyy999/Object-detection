import numpy as np 
import os 
import six.moves.urllib as urllib
import sys 
import tensorflow as tf 
import pathlib
import cv2

from collections import defaultdict
from io import StringIO

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils 

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

cap = cv2.VideoCapture('test_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('results/test_video.avi', fourcc, 20.0, (1080, 720))

MODEL_NAME = 'tom_and_jerry_ssd_mobilenet'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = 'D:/Python/KOWALSKYYY/object-detection/Tom and Jerry/annotations/object-detection.pbtxt'

NUM_CLASSES = 4

model_dir = pathlib.Path(MODEL_NAME)


detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT,'rb') as f:
		serialized_graph = f.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
	with tf.compat.v1.Session(graph=detection_graph) as sess:
		while True:
			ret, image_np = cap.read()
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections], feed_dict={image_tensor:image_np_expanded})
			vis_utils.visualize_boxes_and_labels_on_image_array(image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness = 9)

			out.write(cv2.resize(image_np,(1080, 720)))
			cv2.imshow('object_detection', cv2.resize(image_np,(1080, 720)))
			

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		out.release()
		cv2.destroyAllWindows()