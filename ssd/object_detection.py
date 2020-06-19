#!/usr/bin/env python
# coding: utf-8

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import cv2
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from imageio import imread
from PIL import Image
from imutils.video import FPS
import time
import os
import configparser
import warnings
warnings.filterwarnings("ignore")

class SSD:
    def __init__(self, config):
        # load the model name
        model_type = eval(config['inputs']['model'])  
        
        # load data set type
        self.dataset = eval(config['inputs']['dataset'])
    
        # path for the input image/video/webcam
        self.img_path = eval(config['inputs']['source_path'])

        # load the output directory path
        self.output_dir = eval(config['outputs']['output_dir'])
        
        # load the weights path of model
        self.weights_path = eval(config['ssd_model_params']['weights_path'])
        
        # get confidence threshold
        self.confidence_threshold = eval(config['ssd_model_params']['confidence_threshold'])
        
        # get image size of model
        self.img_height = eval(config['ssd_model_params']['img_height'])
        self.img_width = eval(config['ssd_model_params']['img_width'])
        
        if model_type == 'custom':
            # get model classes
            self.classes = eval(config['ssd_custom_model_params']['classes'])
        
        elif self.dataset == 'pascal_voc' and model_type =='ssd300':
            # get model classes
            self.classes = eval(config['ssd_pascal_model_params']['classes'])
            
        elif self.dataset == 'ms_coco' and model_type =='ssd300':
            # get model classes
            self.classes = eval(config['ssd_coco_model_params']['classes'])
            
    def load_model(self):
        if self.dataset == 'pascal_voc':
            # load pascal model
            self.load_model_pascal()
            
        elif self.dataset == 'ms_coco':
            # load coco model
            self.load_model_coco()
            
    def load_model_pascal(self):
        # 1: Build the Keras model
        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=len(self.classes)-1,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        self.model.load_weights(self.weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        
    def load_model_coco(self):
        # 1: Build the Keras model
        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=len(self.classes)-1,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        self.model.load_weights(self.weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def load_image(self):
        self.orig_images = [] # Store the images here.
        self.input_images = [] # Store resized versions of the images here.

        self.orig_images.append(imread(self.img_path))
        img = image.load_img(self.img_path, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img) 
        self.input_images.append(img)
        self.input_images = np.array(self.input_images)
        
    def predict_object(self):
        y_pred = self.model.predict(self.input_images)
        self.y_pred_thresh = [y_pred[k][y_pred[k,:,1] > self.confidence_threshold] for k in range(y_pred.shape[0])]
        
    def save_image(self):
        # Draw and save the predicted boxes.
        self.cv2_img = cv2.imread(self.img_path)
        for box in self.y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            self.xmin = int(box[2] * self.orig_images[0].shape[1] / self.img_width)
            self.ymin = int(box[3] * self.orig_images[0].shape[0] / self.img_height)
            self.xmax = int(box[4] * self.orig_images[0].shape[1] / self.img_width)
            self.ymax = int(box[5] * self.orig_images[0].shape[0] / self.img_height)
            
            self.label, self.conf = self.classes[int(box[0])], box[1]
        
            # draw labels on image
            self.draw_predict()
            
        # save image
        cv2.imwrite(os.path.join(self.output_dir,"prediction.jpg"), self.cv2_img)
        
    def save_video(self, frame):
        self.cv2_img = frame
        if self.flag_video_writer:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(os.path.join(self.output_dir, 'prediction.avi'), fourcc, 30, 
                                          (self.cv2_img.shape[1], self.cv2_img.shape[0]), True)
            self.flag_video_writer = False
                
        # Draw and save the predicted boxes.
        self.cv2_img = cv2.imread(self.img_path)
        for box in self.y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            self.xmin = int(box[2] * self.orig_images[0].shape[1] / self.img_width)
            self.ymin = int(box[3] * self.orig_images[0].shape[0] / self.img_height)
            self.xmax = int(box[4] * self.orig_images[0].shape[1] / self.img_width)
            self.ymax = int(box[5] * self.orig_images[0].shape[0] / self.img_height)
            
            self.label, self.conf = self.classes[int(box[0])], box[1]
        
            # draw labels on image
            self.draw_predict()
            
        # write the output frame to disk
        self.writer.write(self.cv2_img)
    
    def draw_predict(self):
        # Draw a bounding box.
        cv2.rectangle(self.cv2_img, (self.xmin, self.ymin), (self.xmax, self.ymax), COLOR_YELLOW, 2)

        text = '{}: {:.2f}'.format(self.label, self.conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(self.ymin, label_size[1])
        cv2.putText(self.cv2_img, text, (self.xmin, self.ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    COLOR_WHITE, 1)

            
if __name__=='__main__':
    
    # load configuration file
    config = configparser.ConfigParser()
    config.read('config.conf')
    
    # Default colors
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (0, 255, 255)
    
    # read input parameters
    # path for the input image/video/webcam
    src = eval(config['inputs']['source_path'])
    
    # load the model name
    model_type = eval(config['inputs']['model'])    
    
    # perform detections on image/video
    src_type = eval(config['inputs']['source_type'])
    
    print('Detector:', model_type)
    print('Source:', src_type)
    
    # initialize detector class
    detector = SSD(config)
    
    # load model
    detector.load_model()

    if src_type == 'image':
        # load image
        detector.load_image()
        
        # make predictions
        detector.predict_object()
        
        # save image
        detector.save_image()
        
        
    if src_type=='webcam' or src_type=='video':
        print("Press 'q' to exit")
        cap = cv2.VideoCapture(src)
        time.sleep(2.0)
        
        detector.flag_video_writer = True
        fps = FPS().start()

        while True:
            ret, frame = cap.read()
            if ret!=True:
                print('Not able to load image from webcam')
                break
            
            # read webcam image & save
            cv2.imwrite('frame.jpg', frame)

            # set image path
            detector.img_path = 'frame.jpg'

            # load image
            detector.load_image()

            # make predictions
            detector.predict_object()

            # save video frame
            detector.save_video(frame)

            cv2.imshow('SSD object detection', detector.cv2_img)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

            # update the FPS counter
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        detector.writer.release()
        cap.release()
        cv2.destroyAllWindows()
        