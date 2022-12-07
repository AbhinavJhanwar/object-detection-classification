<b>for pretrained model weights-</b>
ping me for the weights file and save it in the folder pretrained_weights

for more info check out https://github.com/pierluigiferrari/ssd_keras<br>

install the required libraries-
```
conda install -c anaconda keras-gpu #==2.2.4
pip install -r requirements.txt
```

Download dataset from- http://host.robots.ox.ac.uk/pascal/VOC/

Instructions to run-
1) modify config.conf file as following-
* model = 'your prefered detector'
* dataset = 'dataset on which model is trained on'
* source_type = 'webcam or video or image'
* source_path = 'camera id or video/image path'
* output_dir = 'directory to save detected video/image'
* weights_path = 'weights file that is being used for object detection'
* confidence_threshold = 'threshold to be used for detection'
* classes = 'path to classes.pickle'

2) run object_detection.py
``` 
python object_detection.py
```

For downloading data of open images, use code data_generator/downloading_open_images.ipynb file.
