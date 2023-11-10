## TAG2TEXT- 
https://arxiv.org/pdf/2303.05657.pdf (https://github.com/xinyu1205/recognize-anything)
<br>It consists of 3 modules-
1. Image tagging- using encoder generates image features then generates tags using decoder
2. Image tag text generation (image captioning)- using given tags from module 1 and image features from the encoder, it utilizes transformer (encoder and decoder) architecture to generate image caption or image text
3. Image text alignment- determines whether given text and image are aligned or not
Overall using image tags from module 1 and image features, it generates image caption in module 2 and then using module 3 evaluates the result by calculating Image-Text Contrastive Loss (ITC) and Image-Text Matching Loss (ITM).
<br>At the time of inference along with the tags from module 1, user can input his own tags to get relevant caption.<br>
Encoder is based on Vision Transformer (ViT) model pre-trained on ImageNet or Swin Transformer model trained on ImageNet.<br>

<b>References:</b>
1. https://huggingface.co/microsoft/swin-base-patch4-window7-224
2. https://huggingface.co/google/vit-base-patch16-224
3. https://github.com/xinyu1205/recognize-anything