# Setting up the code pipeline
```
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
conda create --name object_detection python=3.7
pip install -r requirements.txt
```

add "del loss, loss_items, imgs, pred, targets" in line 400 in train.py<br>
add "import gc" in the start of train.py

now add below two lines of code wherever del appears
```
torch.cuda.empty_cache()
gc.collect()
```

wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"

sample run>><br>
```
!gdown https://drive.google.com/u/1/uc?id=1pmlMhaOw9oUqIH7OZP8d0dBKVOLqAYcy&export=download<br>
!unzip /content/yolov7/data.zip -d dataset
```

update data.yaml for corresponding image folders, check download_open_images_v4.ipynb for sample

# training yolo model
python train.py --batch 64 --cfg cfg/training/yolov7.yaml --epochs 100 --data dataset/data/data.yaml --weights 'yolov7.pt' --device 0

# evaluation
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.5 --source dataset/test/images

# display inference on ALL test images
```
import glob
from IPython.display import Image, display
i = 0
limit = 10 # max images to print
for imageName in glob.glob('runs/detect/exp/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
    i = i + 1
```

# Preparing for open images v4
prepare dataset using download_open_images_v4.ipynb file<br>
for pretrained weights comment your email id and I will share.

## train
python train.py --name open_images_v4 --batch 16 --workers 8 --img 640 --epochs 20 --data open_images_v4/data.yaml --hyp data/hyp.scratch.p5.yaml --cfg cfg/training/yolov7.yaml  --weights 'yolov7.pt' --device 0

## evaluation
python detect.py --weights runs/train/open_images_v4/weights/best.pt --conf 0.5 --source open_images_v4/test/images

Final dataset folder looks like below before get into YOLOv7 training,
```
├── yolov7
## └── train
####└── images (folder including all training images)
####└── labels (folder including all training labels)
## └── test
####└── images (folder including all testing images)
####└── labels (folder including all testing labels)
## └── valid
####└── images (folder including all valid images)
####└── labels (folder including all valid labels)
```

Create Custom Config File for Training
We must now develop a customized configuration file. (Be sure to specify the proper directory), as the training process will be entirely dependent on that file.
Create a file with the name "custom.yaml" in the (yolov7/data) folder. In that file, paste the code below. Set the correct path to the dataset folder, alter the number of classes and their names, and then save it.

Make a file that specifies the training configuration. In custom.yaml file, write the following:

```
# path for images
train: /object-detection-classification/yolo/open_images_v4/train/images
val: /object-detection-classification/yolo/open_images_v4/valid/images
test: /object-detection-classification/yolo/open_images_v4/test/images

# number of classes
nc: 5
# list of all classes in sorted order
names: ['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football']
```