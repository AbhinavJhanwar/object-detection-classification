# %%
# For running inference on the TF-Hub module.
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import json

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import imutils
import cv2

# For measuring the inference time.
#import time

from tqdm import tqdm
tqdm.pandas()

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 
detector = hub.load(module_handle).signatures['default']

# %%
# Helper functions for downloading images and for visualization.
# Visualization code adapted from TF object detection API for the simplest required functionality.
def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  
  open_cv_image = np.array(pil_image)
  h,w,c = open_cv_image.shape
  if h/480<w/640:
    resized = imutils.resize(open_cv_image, width=620)
    h,w,c = resized.shape
    top = (480-h)//2
    bottom = top
    left = 10
    right = 10
  else:
    resized = imutils.resize(open_cv_image, height=460)
    h,w,c = resized.shape
    top = 10
    bottom = 10
    left=(640-w)//2
    right=left

  image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
  pil_image = Image.fromarray(image)

  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  
  #print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename

def download_and_resize_image1(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
  
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  
  #print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=20, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img
     

# %%
def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  #start_time = time.time()
  result = detector(converted_img)
  #end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  #print("Found %d objects." % len(result["detection_scores"]))
  #print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  boxes = result['detection_boxes']
  min_score=0.1
  scores = result['detection_scores']
  class_names = result['detection_class_entities']

  objects_detected = pd.DataFrame()

  for i in range(boxes.shape[0]):
      if scores[i] >= min_score:
        #ymin, xmin, ymax, xmax = tuple(boxes[i])
        objects_detected = pd.concat([objects_detected, pd.DataFrame({'object':[class_names[i].decode("ascii")], 'score':[int(100 * scores[i])]})])
        
  # display_image(image_with_boxes)
  return result, objects_detected
     

# %%
image_urls = [
  # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
  "https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg",
  # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
  "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
  # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
  "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg",
  ]

def detect_img(image_url):
  #start_time = time.time()
  image_path = download_and_resize_image(image_url, 640, 480, True)
  result, df = run_detector(detector, image_path)
  #end_time = time.time()
  #print("Inference time:",end_time-start_time)
  try:
    return df['object'].unique()
  except:
    return None

def detect_img1(image_url):
  #start_time = time.time()
  image_path = download_and_resize_image1(image_url, 640, 480, False)
  result, df = run_detector(detector, image_path)
  #end_time = time.time()
  #print("Inference time:",end_time-start_time)
  try:
    return df['object'].unique()
  except:
    return None

# %%
# df1 = pd.read_pickle('final_df.pkl').reset_index(drop=True)
# df1['image_filename'] = df1['image_filename'].apply(lambda x: x.split('/')[-1])

# f = open('gcp_results.json')
# data = json.load(f)

# # set objects and labels
# df1['objects'] = None
# df1['labels'] = None
# for i in range(df1['image_filename'].shape[0]):
#   object_data = set([filename['name'] for filename in data[df1.iloc[i]['image_filename']]['Objects']])
#   labels_data = set(data[df1.iloc[i]['image_filename']]['Labels'])
#   df1.loc[i, 'objects'] = str(object_data)
#   df1.loc[i, 'labels'] = str(labels_data)

# %%
df = pd.read_pickle('final_objects3.pkl')

# %%
#df1['frcnn_objects'] = None
for i in tqdm(range(df['image_filename'].shape[0])):
  df.loc[i, 'frcnn_objects'] = str(detect_img(df['primaryImageURL'].iloc[i]))
  #df.to_pickle('final_objects4.pkl')

# %%
import pandas as pd
df = pd.read_pickle('final_objects4.pkl')
# %%
i=1
i=120
i=130
i=270
i=360
i=4298
i=6782
i=7590
i=8001
url = df['primaryImageURL'].iloc[i]
print('new frcnn:', detect_img(url))
print('old frcnn:', detect_img1(url))
print('Vision Objects:', df['objects'].iloc[i])
print('Vision Labels:', df['labels'].iloc[i], '\n')
print(df['frcnn_objects'].iloc[i])
