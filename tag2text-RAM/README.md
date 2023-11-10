## TAG2TEXT- 
https://arxiv.org/pdf/2303.05657.pdf
<br>It consists of 3 modules-
1. Image tagging- using encoder generates image features then generates tags using decoder
2. Image tag text generation (image captioning)- using given tags and image features from the encoder, it utilizes transformer (encoder and decoder) architecture to generate image caption or image text. At the time of training it uses tag parsed from the given text using text semantics while at the time of inference these tags are provided by module 1.
3. Image text alignment- determines whether given text and image are aligned or not
Overall using image tags from module 1 and image features, it generates image caption in module 2 and then using module 3 evaluates the result by calculating Image-Text Contrastive Loss (ITC) and Image-Text Matching Loss (ITM).<br>
<br>The image features also interact with tags by the cross-attention layers in the module 1 & 2.
<br>At the time of inference along with the tags from module 1, user can input his own tags to get relevant caption.
<br>Encoder is based on Vision Transformer (ViT) model pre-trained on ImageNet or Swin Transformer model trained on ImageNet.


## RAM (Recognize Anything Model)- 
https://arxiv.org/pdf/2306.03514.pdf
<br>Improvements over Tag2Text-<br>
Open Vocabulary Recognition- Instead of just using tag embeddings a separate encoder is used to generate semantically rich text and that embedding is used which facilitates generalization to previously unseen categories in training stage. The encoder-decoder used for text generation are 12-layer transformers, and the tag recognition decoder is a 2-layer transformer. Off-the-shelf text encoder from CLIP is utilized to perform prompt ensembling to obtain textual label queries. CLIP image encoder is also used to distill image feature, which further improves the model’s recognition ability for unseen categories via image-text feature alignment.

<b>References:</b>
1. https://huggingface.co/microsoft/swin-base-patch4-window7-224
2. https://huggingface.co/google/vit-base-patch16-224
3. https://github.com/xinyu1205/recognize-anything