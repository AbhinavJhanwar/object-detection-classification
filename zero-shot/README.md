# Installation
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

# Usage
image_classification.py <br>
I have added some labels to map e-commerce product images into various classes like- lifestyle, features, instructions, dimensions, comparison and composition.<br>
This helps to determine the variour kinds of images that are available for a particular product which also affects its rank.
```
output = get_image_class(image_to_classify, model_name, labels_for_classification, 'method1')
print(*output, sep='\n')
for dict_key in classification_map.keys():
    for result in output:
        if result['label'] in classification_map[dict_key]:
            if result['score']>0.1:
                print('final label:', dict_key)
```

# References
* https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads
* https://www.pinecone.io/learn/zero-shot-image-classification-clip/

