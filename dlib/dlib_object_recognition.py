# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:49:38 2018

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

images_folder = "images"

# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# for faces set it true
#options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5 # Specify penalty parameter, too large - overfit, too small - underfit
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join(images_folder, "objects/training.xml")
#testing_xml_path = os.path.join(images_folder, "testing.xml")
# This function does the actual training.  It will save the final detector to
# detector.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# However, to get an idea if it really worked without overfitting we need to
# run it on images it wasn't trained on.  The next line does this.  Happily, we
# see that the object detector works perfectly on the testing images.
#print("Testing accuracy: {}".format(
#    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))


#####################################################
############ object detection from images ###########
#####################################################

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("detector.svm")


# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

# Now let's run the detector over the images in the faces folder and display the
# results.
print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(images_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img)
    print("Number of faces/objects detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    #dlib.hit_enter_to_continue()
    time.sleep(5)
    



########################################################################################
############## object detection from live video with single detector #############
######################################################################################

#detector = dlib.simple_object_detector("detector.svm")
detector = dlib.fhog_object_detector("detector.svm")
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
            cv2.putText(camera_frame, "No known objects/faces detected", (6, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
       else:
            for det in dets:
                p1 = (det.left(), det.top())
                p2 = (det.right(), det.bottom())
                color = (0, 0, 255) # Blue
                cv2.rectangle(camera_frame, p1, p2, color)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(camera_frame, "Face Detected", (p1[0] + 6, p2[1]+15), font, 0.5, (255, 255, 255), 1)

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
pickle_in = open("detectorsModels.pickle","rb")
detectorsModels = pickle.load(pickle_in)

pickle_in = open("detectors.pickle","rb")
detectors = pickle.load(pickle_in)

# start recognition
recognizeObject()    
    
