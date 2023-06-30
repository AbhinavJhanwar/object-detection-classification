I have implemented following object/person detection models here namely-
1) SSD
2) FRCNN
3) YOLO V7
4) Zero Shot Detection

# 1. SSD
* **Paper**- https://arxiv.org/pdf/1512.02325.pdf
* **Repo**- https://github.com/pierluigiferrari/ssd_keras
* **Methodology/technique**- ssd (single shot detection): dividing whole image into segments of fixed size then using various size of bounding boxes (6 per feature cell) and pass it through the network to classify. Here the difference is that this technique is applied not just on last conv layer but at multiple levels to handle scaling and then in the end NMS (non maximum suppression) is used to get final bounding box of objects
* **Architecture/base network**- pretrained vgg16 layers + few custom conv layers and classificatin layers from various layers
* **Dataset trained on**- PASCAL VOC and COCO datasets
* **Input image size**- 300x300, 512x512
* **Optimizer**- sgd in original implementation and adam in the current implementation
* **Anchor boxes**- 4/6 default boxes per cell
* **loss function**- confidence_loss (logloss) + alpha*location_loss (smooth L1 loss)

# 2. FRCNN
* **Paper**- https://arxiv.org/abs/1506.01497
* **Repo**- https://github.com/rbgirshick/py-faster-rcnn
* **Methodology/technique**- feature extraction using pretrained vgg16 network, region proposal network (RPN) to propose ROI, then few layers (fc) to end up in classification (object/background) through softmax and bounding box through regressor, then NMS based on object score and another few layers (fc) to decide which object class if objectness score from previous fc layers is positive (NMS)
* **Architecture/base network**-  Based on ZF/VGG16 layers + FC layers
* **Dataset trained on**- PASCAL VOC 2007, 2012, and MS COCO
* **Input image size**- 1000 x 600
* **Optimizer**- sgd
* **Anchor boxes**- 9 (3 aspect ratio * 3 scaled) anchors, for pascal 3 scales of box area 128², 256², 512² and 3 aspect ratios of 1:1, 1:2 and 2:1
* **loss function**- log loss for classification in RPN, absolute error for regression in RPN

# 3. YOLO V7
* **Paper**- https://arxiv.org/abs/2207.02696
* **Repo**- https://github.com/WongKinYiu/yolov7
* **Methodology/technique**- 
yolov3: dividing whole image into segments of fixed size then using various size of bounding boxes (3 per feature cell) and pass it through the network to classify. Here the difference is that this technique is applied not just on last conv layer but at multiple (3) levels (hence making total prediction per cell to 3x3 = 9) to handle scaling and then in the end nms (non maximum suppression) is used to get final bounding box of objects. Instead of softmax, sigmoid activation function is used, in this way a single bounding box may indicate multiple classes. In yolo, total no. of bounding boxes in a image detected or trained are also set to maximum value of 20.<br>
* **Architecture/base network**- DarkNet (contains residual blocks and upsampling), Extended-ELAN (E-ELAN) with reparameterized convolution (RepConv)
* **Dataset trained on**- MS COCO
* **Input image size**- 640*640
* **Optimizer**- Adam (v3)
* **Anchor boxes**- 3 default boxes per cell from 3 different layers (v3)
* **loss function**- confidence_loss (binary cross entropy) + alpha*location_loss (squared error loss) (v3)

# 4. Zero Shot Detection
* **Paper**- 
* **Repo**- https://github.com/openai/CLIP
* **Methodology/technique**- 
* **Architecture/base network**- 
* **Dataset trained on**- 
* **Input image size**- 
* **Optimizer**- 
* **Anchor boxes**- 
* **loss function**- 