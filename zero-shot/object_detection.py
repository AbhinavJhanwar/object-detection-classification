
# %%
# https://huggingface.co/docs/transformers/v4.30.0/en/tasks/zero_shot_object_detection#zeroshot-object-detection-pipeline
# https://huggingface.co/docs/transformers/model_doc/owlvit
# object detection
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoProcessor

def draw_boxes(image, predictions):
    im = image.copy()

    draw = ImageDraw.Draw(im)
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black")
    return im


# https://huggingface.co/models?other=owlvit&sort=downloads
checkpoint = "google/owlvit-base-patch32"
image_to_classify = "images/test16.jpg"
image_to_classify = "images/test12.jpg"
candidate_labels=["human face", "rocket", "screen",
                "nasa badge", "star-spangled banner", 
                "toothbrush", "toothpaste", "text",
                "hat", "book", "sunglasses", "camera",
                "sofa", "lamp", "table", "plant", "pot", 
                "Air Conditioner", "human hand", "laptop"]

image = Image.open(image_to_classify)
plt.imshow(image)

# %%
############ method 1: using pipeline
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

predictions = detector(
    image,
    candidate_labels=candidate_labels,
)

print(*predictions, sep='\n')
predicted_image = draw_boxes(image, predictions)
plt.imshow(predicted_image)

# %%
############## method 2: by hand
checkpoint = "google/owlvit-base-patch32"
model = OwlViTForObjectDetection.from_pretrained(checkpoint)
processor = OwlViTProcessor.from_pretrained(checkpoint)

inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process(outputs, target_sizes=target_sizes)[0]

scores = results["scores"].tolist()
labels = results["labels"].tolist()
boxes = results["boxes"].tolist()

predictions = []
score_threshold = 0.1
for box, score, label in zip(boxes, scores, labels):
    if score >= score_threshold:
        predictions.append({'score':score,
                            'label':candidate_labels[int(label)],
                            'box':{'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3]}})

print(*predictions, sep='\n')  
predicted_image = draw_boxes(image, predictions)
plt.imshow(predicted_image)

# %%
############# method 3: batch

checkpoint = "google/owlvit-base-patch32"
model = OwlViTForObjectDetection.from_pretrained(checkpoint)
processor = OwlViTProcessor.from_pretrained(checkpoint)

image1 = Image.open("images/test18.jpg")
image2 = Image.open("images/test16.jpg")
images = [image1, image2]

text_queries = [
    ["sofa", "lamp", "table", "plant", "pot", "Air Conditioner", 
     "clock", "microwave oven", "tap", "human hand", "laptop", 
     "chair", "monitor", "tv", "pencil stand"],
    ["sofa", "lamp", "table", "plant", "pot", "Air Conditioner", 
     "clock", "microwave oven", "tap", "human hand", "laptop", 
     "chair", "monitor", "tv", "pencil stand"]
     ]

inputs = processor(text=text_queries, images=images, 
                   return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([x.size[::-1] for x in images])
    results = processor.post_process(outputs, target_sizes=target_sizes)

# %%
image_idx = 0

scores = results[image_idx]["scores"].tolist()
labels = results[image_idx]["labels"].tolist()
boxes = results[image_idx]["boxes"].tolist()

predictions = []
score_threshold = 0.2
for box, score, label in zip(boxes, scores, labels):
    if score >= score_threshold:
        predictions.append({'score':score,
                            'label':text_queries[image_idx][int(label)],
                            'box':{'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3]}})

print(*predictions, sep='\n')  
predicted_image = draw_boxes(images[image_idx], predictions)
plt.imshow(predicted_image)

# %%
############## method 4 : image vs image
import requests
import matplotlib.pyplot as plt

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_target = Image.open(requests.get(url, stream=True).raw)

query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(query_image)
ax[1].imshow(image_target)

inputs = processor(images=image_target, query_images=query_image, 
                   return_tensors="pt")

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)
    target_sizes = torch.tensor([image_target.size[::-1]])
    results = processor.post_process_image_guided_detection(
        outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes)[0]

draw = ImageDraw.Draw(image_target)

scores = results["scores"].tolist()
boxes = results["boxes"].tolist()

for box, score in zip(boxes, scores):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)
    draw.text((xmin, ymin), f"{round(score,2)}", fill="black")

image_target
