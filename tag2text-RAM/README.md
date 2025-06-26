## CLIP (Contrastive Language–Image Pretraining)
https://arxiv.org/abs/2103.00020.pdf<br>
It consists of two separate encoders which map their respective inputs—images and text—into a shared embedding space: 
1. an image encoder (typically a Vision Transformer or ResNet) and
2. a text encoder (a Transformer-based language model).<br><br>
During training, CLIP is fed batches of image–text pairs collected from the internet. It uses a contrastive loss to align each image with its corresponding caption and push apart mismatched pairs. Specifically, it computes the cosine similarity between every image and every text in a batch, and uses a symmetric cross-entropy loss to maximize the similarity of the correct pairs while minimizing incorrect ones. Here each batch will have a unique pair that matches with each other and no other text or image.<br><br>
At inference, CLIP performs zero-shot classification by comparing an image's embedding to the embeddings of text prompts like "a photo of a cat", "a photo of a dog", etc. The label with the highest similarity is selected. 
![image](https://github.com/user-attachments/assets/a858b856-5334-42e0-8562-2d38057c3571)


## TAG2TEXT- 
https://arxiv.org/pdf/2303.05657.pdf
<br>It consists of 3 modules-
1. Image tagging- it uses a Vision Transformer (ViT) based image encoder pre-trained using DINO architecture on ImageNet or Swin Transformer model trained on ImageNet to generates image encoding then generates tags (semantically rich text) for the image using a transfomer based decoder.
2. Image tag text generation (image captioning)- using given tags embedding and image encoding, it utilizes transformer (encoder and decoder) architecture to generate image caption or image text, which is the final output. At the time of training it uses tag parsed from the given text using text semantics while at the time of inference these tags are provided by module 1 and are matched with image to generate final image description.
3. Image text alignment- determines whether given text (tags) and generated text caption and image are aligned or not in a semantic space.
Overall using image tags from module 1 and image encoding, it generates image caption in module 2 and then using module 3 evaluates the result by calculating Image-Text Contrastive Loss (ITC) i.e. cosine similarity and Image-Text Matching Loss (ITM).<br>
<br>The image features also interact with tags by the cross-attention layers in the module 1 & 2.
<br>At the time of inference along with the tags from module 1, user can input his own tags to get relevant caption.
![image](https://github.com/user-attachments/assets/389789f8-1eb3-44ec-adba-710d23d04d46)
![image](https://github.com/user-attachments/assets/a1cc00e7-936c-446e-8610-9f2b251dd61b)



## RAM (Recognize Anything Model)- 
https://arxiv.org/pdf/2306.03514.pdf
<br>Improvements over Tag2Text-<br>
Open Vocabulary Recognition- Instead of just using tag embeddings a separate encoder is used to generate semantically rich text and that embedding is used which facilitates generalization to previously unseen categories in training stage. The encoder-decoder used for text generation are 12-layer transformers, and the tag recognition decoder is a 2-layer transformer. Off-the-shelf text encoder from CLIP is utilized to perform prompt ensembling to obtain textual label queries. CLIP image encoder is also used to distill image feature, which further improves the model’s recognition ability for unseen categories via image-text feature alignment.
![image](https://github.com/user-attachments/assets/313e8299-27e1-4ac7-b730-5a61725c488d)


<b>References:</b>
1. https://huggingface.co/microsoft/swin-base-patch4-window7-224
2. https://huggingface.co/google/vit-base-patch16-224
3. https://github.com/xinyu1205/recognize-anything
4. https://github.com/openai/CLIP
