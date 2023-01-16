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
!gdown https://drive.google.com/u/1/uc?id=1pmlMhaOw9oUqIH7OZP8d0dBKVOLqAYcy&export=download<br>
!unzip /content/yolov7/data.zip -d dataset

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

