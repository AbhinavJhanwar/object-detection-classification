# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:27:11 2018

@author: abhinav.jhanwar
"""

import cv2
import dlib
import os
import glob
import time
import pickle

#########################################################################
############### training dlib for object/face detection #################
#########################################################################

images_folder = "images/objects"
objects = ['Axe Dark Temptation']

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 5 # Specify penalty parameter, too large - overfit, too small - underfit
options.num_threads = 2
options.be_verbose = True

for i in objects:
    print("Training model for {0}...".format(i))
    training_xml_path = images_folder+"/Axe Dark Temptation/training/compressed/training.xml"#"{0}/training/training.xml".format(i))
    #testing_xml_path = os.path.join(images_folder, "testing.xml")
    output_path = images_folder+"/Axe Dark Temptation/Axe Dark Temptation.svm"#"{0}/{1}.svm".format(i,i))
    dlib.train_simple_object_detector(training_xml_path, output_path, options)
    
    print("")  # Print blank line to create gap from previous output
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, os.path.join(images_folder, "{0}/{1}.svm".format(i,i)))))
    #print("Testing accuracy: {}".format(
        #dlib.test_simple_object_detector(testing_xml_path, os.path.join(images_folder, "{0}/{1}.svm".format(i,i)))))
    
#####################################################
############ object detection from images ###########
#####################################################
for i in objects:
    detector = dlib.simple_object_detector(os.path.join(images_folder, "{0}/{1}.svm".format(i,i)))
    #detector = dlib.fhog_object_detector(os.path.join(images_folder, "{0}/{1}.svm".format(i,i)))
    print("Showing detections on the images in the objects folder...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(images_folder, "{0}/training/*.jpg".format(i))):
        #f="images/objects\Axe Dark Temptation\IMG_20181210_112945.jpg"
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        #dets = dlib.fhog_object_detector.run(detector, img)[0]
        dets = detector(img)
        print("Number of objects detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
    
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        time.sleep(0.5)
        
########################################################################################
############## object detection from live video with single detector #############
######################################################################################

detector_url = "images/objects/Axe Dark Temptation/Axe Dark Temptation.svm"
detector = dlib.simple_object_detector(detector_url)
#detector = dlib.fhog_object_detector(detector_url)
#image = dlib.load_rgb_image(images_folder + '/impact mints.jpg')
#dlib.fhog_object_detector.run(detector, image)
camera = cv2.VideoCapture(0)

if camera.isOpened() is not True:
    print("Could not access camera")
    exit()
else:
    print("Camera ready")
    
width, height = 840, 640
fps = 10
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, fps)    

while True:
    # Capture frame-by-frame
    capture_success, camera_frame = camera.read()
    #camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    
    if capture_success:
       dets = detector(camera_frame)
        
       if len(dets) == 0:
            cv2.putText(camera_frame, "No known objects detected", (6, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
       else:
            for det in dets:
                p1 = (det.left(), det.top())
                p2 = (det.right(), det.bottom())
                color = (0, 0, 255) # Blue
                cv2.rectangle(camera_frame, p1, p2, color)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(camera_frame, "Object", (p1[0] + 6, p2[1]+15), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Object Detector', camera_frame)
    key = cv2.waitKey(1000 // fps)
    if key == 27: # Escape key
        break
    
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

#################################################################################
################### training for object recognition with mulitple detectors ##############
##################################################################################3
def trainObjects():
    detectors = []
    for xml in glob.glob(os.path.join(images_folder, "*.xml")):
        dlib.train_simple_object_detector(xml, "{0}.svm".format(xml.split('\\')[1].split('.')[0]), options)
        # save all detectors as list
        detectors.append("{0}.svm".format(xml.split('\\')[1].split('.')[0]))
        
        
    # Next, suppose you have trained multiple detectors and you want to run them
    # efficiently as a group. You can do this as follows
    detectorsModels = list()
    for detector in detectors:
        detectorsModels.append(dlib.fhog_object_detector(detector))
    
    # testing multiple detectors with image
    image = dlib.load_rgb_image(images_folder + '/head-and-shoulder-best-oily-hair-shampoo.jpg')
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectorsModels, image, upsample_num_times=1, adjust_threshold=0.5)
    for i in range(len(boxes)):
        print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
    return detectorsModels, detectors

#############################################
############## testing with live video ###########
##############################################

def recognizeObject():
    camera = cv2.VideoCapture(0)
    
    if camera.isOpened() is not True:
        print("Could not access camera")
        exit()
    else:
        print("Camera ready")
        
    width, height = 640, 420
    fps = 10
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)    
    
    while True:
        # Capture frame-by-frame
        capture_success, camera_frame = camera.read()
        #camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        
        if capture_success:
            [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectorsModels, camera_frame, upsample_num_times=0, adjust_threshold=0.22)
            for i in range(len(boxes)):
                print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
       
                if len(detector_idxs) == 0:
                    cv2.putText(camera_frame, "No known objects/faces detected", (6, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                else:
                    p1 = (boxes[i].left(), boxes[i].top())
                    p2 = (boxes[i].right(), boxes[i].bottom())
                    color = (0, 0, 255) # Blue
                    cv2.rectangle(camera_frame, p1, p2, color)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(camera_frame, detectors[detector_idxs[i]].split('.')[0], (p1[0] + 6, p2[1]+15), font, 0.5, (0, 0, 255), 1)
                
        # Display the resulting frame
        cv2.imshow('Object Recognizer', camera_frame)
        key = cv2.waitKey(1000 // fps)
        if key == 27: # Escape key
            break
        
    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()

        
detectorsModels, detectors = trainObjects()

# save models
pickle_out = open("detectorsModels.pickle","wb")
pickle.dump(detectorsModels, pickle_out)
pickle_out.close()

pickle_out = open("detectors.pickle","wb")
pickle.dump(detectors, pickle_out)
pickle_out.close()

# load models
pickle_in = open("C:/Users/abhinav.jhanwar/Documents/workspace-sts-3.8.4.RELEASE/Python_Tutorials/src/MachineLearning/AlgosImplemented/Computer Vision/Facial Recognition/dlib/detectorsModels.pickle","rb")
detectorsModels = pickle.load(pickle_in)

pickle_in = open("C:/Users/abhinav.jhanwar/Documents/workspace-sts-3.8.4.RELEASE/Python_Tutorials/src/MachineLearning/AlgosImplemented/Computer Vision/Facial Recognition/dlib/detectors.pickle","rb")
detectors = pickle.load(pickle_in)

# start recognition
recognizeObject()    
    
