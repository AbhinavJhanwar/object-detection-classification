# %%
from transformers import pipeline 
# More models in the model hub.
# https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads
# https://www.pinecone.io/learn/zero-shot-image-classification-clip/

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from typing import Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_image_class(image:Union[str, list], model_name:str, labels:list, method:str):
    """ 
    method1 - using pipeline
    method2 - using text and image encoders

    Args:
        image (Union[str, list]): local address of image
        model_name (str): choose from "openai/clip-vit-large-patch14-336" or "openai/clip-vit-base-patch32"
        labels (list): list of various labels
        method (str): whether to run on method1 or method2

    Returns:
        _type_: returns dictionary of label and confidence
    """
    
    image = Image.open(image_to_classify)
    plt.imshow(image)

    if method=='method1':
        classifier = pipeline("zero-shot-image-classification", model=model_name)
        scores = classifier(image, candidate_labels=labels)
    
    if method=='method2':
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name) 

        # if you have CUDA set it to the active device like this
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # move the model to the device
        model.to(device)
        #print(device)

        # create label tokens
        label_tokens = processor(
            text=labels_for_classification,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to(device)
        #print(label_tokens['input_ids'][0][:10])

        # encode tokens to sentence embeddings
        label_emb = model.get_text_features(**label_tokens)
        # detach from pytorch gradient computation
        label_emb = label_emb.detach().cpu().numpy()
        #print(label_emb.shape)
        #print(label_emb.min(), label_emb.max())

        # normalize the weights to apply dot products
        # normalization
        #label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
        #print(label_emb.min(), label_emb.max())

        image = Image.open(image_to_classify)

        # process image
        image = processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(device)
        #print(image.shape)

        # generate embedding
        img_emb = model.get_image_features(image)
        #print(img_emb.shape)
        img_emb = img_emb.detach().cpu().numpy()

        # normalize
        #img_emb = img_emb / np.linalg.norm(img_emb, axis=0)

        # distance matrix
        #scores = np.dot(img_emb, label_emb.T)
        scores = cosine_similarity(img_emb, label_emb)*100
        #print(scores.shape)

        # convert to probability
        scores = softmax(scores).flatten()

        # to dictionary
        scores = dict(zip(labels_for_classification, scores))
        scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        scores = [{'label':item[0], 'score':item[1]} for item in scores_sorted]
    return scores[:2]

# %%
model_name = "openai/clip-vit-large-patch14-336"
model_name = "openai/clip-vit-base-patch32"
#model_name = "openai/clip-vit-large-patch14"

image_to_classify = "image classification images/comparison/3.jpg"
image_to_classify = "image classification images/composition/6.jpg"
image_to_classify = "image classification images/dimensions/5.jpg"
image_to_classify = "image classification images/features_benefits/15.jpg"
image_to_classify = "image classification images/instructions/3.jpg"
image_to_classify = "image classification images/lifestyle/33.jpg"

classification_map = {'lifestyle':['lifestyle', 'cafe', 'bar', 'pub', 'kitchen', 'natural scenary', 
                                   'park', 'garden', 'cabin', 'corner of a room',
                                   'study room', 'gym', 'drawing room', 'bedroom', 'office',
                                   'sitting area', 'indoors', 'living room', 
                                   'people using product', 'outdoor', 
                                   'a boy is smiling while using a product', 
                                   'a girl is smiling while using a product',
                                   'restaurant', 'girl is smiling with something in her hand', 
                                   'boy is smiling with something in his hand', 
                                   'people enjoying something', 'people eating', 
                                   'people playing', 'hand holding some product'], 
                      'dimensions':['scale of product', 'size specifications of product', 
                                    'measurement of product', 'height and width of product', 
                                    'dimensions of product'],
                      'features/benefits':['usage benefits of product',
                                            'product key features',
                                            'steps, benefits of usage of product',
                                            'serving details benefits of product'],
                      'instructions':['steps to serve product',
                                      'how to cook product', 
                                      'how to ready product to use',  
                                      'how to apply product to use', 
                                      'directions to use product'],
                      'comparison':['comparison of two products', 
                                    'old versus new'],
                      'composition':['composition', 'ingredients', 'nutrition facts'],
                      'usage':['usage']
                      }

image_to_classify = "image classification images/comparison/3.jpg"
image_to_classify = "image classification images/composition/6.jpg"
image_to_classify = "image classification images/dimensions/5.jpg"
image_to_classify = "image classification images/features_benefits/15.jpg"
image_to_classify = "image classification images/instructions/3.jpg"
image_to_classify = "image classification images/lifestyle/33.jpg"

labels_for_classification = []
for items in classification_map.items():
    labels_for_classification += items[1]

results = get_image_class(image_to_classify, model_name, labels_for_classification, 'method1')
print('method 1', *results, sep='\n')
for dict_key in classification_map.keys():
    for result in results:
        if result['label'] in classification_map[dict_key]:
            result['new_label']= dict_key
            if result['score']>0.1:
                print('final label:', dict_key)

results = get_image_class(image_to_classify, model_name, labels_for_classification, 'method2')
print('method 2', *results, sep='\n')
for dict_key in classification_map.keys():
    for result in results:
        if result['label'] in classification_map[dict_key]:
            result['new_label']= dict_key
            if result['score']>0.1:
                print('final label:', dict_key)

