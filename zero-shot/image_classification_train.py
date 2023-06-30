# %%
#################################
#### inference
#################################
import torch
import clip
from PIL import Image
import tensorflow as tf 
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
img_height, img_width = 32, 32

label_definition = {
                        0:'airplane',
                        1:'automobile',
                        2:'bird',
                        3:'cat',
                        4:'deer',
                        5:'dog',
                        6:'frog',
                        7:'horse',
                        8:'ship',
                        9:'truck'
                        }
idx = 26
image = preprocess(Image.fromarray(x_train[idx])).unsqueeze(0).to(device)
text_options = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck']

text = clip.tokenize(text_options).to(device)

# can also extract image and text features as below-
image_features = model.encode_image(image)
text_features = model.encode_text(text)

with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = probs.flatten()

results = dict(zip(text_options, probs))
results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("Label probs:", *results, sep='\n')  # prints: [[0.9927937  0.00421068 0.00299572]]
print("Actual Value:", label_definition[y_train[idx][0]])
Image.fromarray(x_train[idx])

# %%
# check available models
clip.available_models()

# %%
####################################
###### training
####################################
# https://github.com/openai/CLIP/issues/83

##################################
############# prepare data
import tensorflow as tf 
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import clip
import pandas as pd
import numpy as np
from tqdm import tqdm 
tqdm.pandas()

# %%
# Load in the data from keras
cifar10 = tf.keras.datasets.cifar10

# we will have to convert cifar10 data to pytorch data
# steps-
# 1. load train and test data
# 2. map integers to labels
# 3. convert the image data to numpy array
# 4. save the data as df to be able to utilized by pytorch

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
img_height, img_width = 32, 32

label_definition = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}

# train_data
train_data = pd.DataFrame()
train_data['label'] = y_train[:, 0]

# update int to text labels
train_data['label'] = train_data['label'].apply(lambda x: label_definition[x])

# convert image array to PIL objects
images = []
for i in range(len(x_train)):
    images.append(x_train[i].flatten().tolist())
train_data['image'] = images

# now we have to update the sequence of data such that
# when I train the model I will have unique labels in a 
# particular batch
# reason behind is that CLIP model is designed such a way
# of cosine similarity that it tries to keep the distance
# between Image1 and text1 smallest while all other
# distances of Image1, text2... until Image1 textn wide

# sort train_data as per labels
train_data = train_data.sort_values('label').reset_index(drop=True)
# get count of the label which has smallest number of occurence
label_count = 5000
unique_number_of_labels = 10
df = pd.DataFrame()
for i in tqdm(range(label_count)):
    # get indices of all 10 labels in sequence
    indices = np.linspace(i, label_count*(unique_number_of_labels-1), unique_number_of_labels).astype(int)

    # add these values in dataframe
    df = pd.concat([df, train_data.iloc[indices]])

# final dataframe is such that every 10 rows has unique
# labels based on number of labels
df.head()

# save data
df.to_csv('cifar10_train.csv', index=False)

# test_data
test_data = pd.DataFrame()
test_data['label'] = y_test[:, 0]

# update int to text labels
test_data['label'] = test_data['label'].apply(lambda x: label_definition[x])

# convert image array to PIL objects
images = []
for i in range(len(x_test)):
    images.append(x_test[i].flatten().tolist())
test_data['image'] = images

# save data
test_data.to_csv('cifar10_test.csv', index=False)

# %%
##############################
########### load data
# columns - image, caption
train_data = pd.read_csv('cifar10_train.csv')
test_data = pd.read_csv('cifar10_test.csv')
img_height, img_width = 32, 32

def update_str_to_array(x):
    x = eval(x)
    x = np.array(x)
    return x

# convert str to numpy array to utilize as image later
train_data['image'] = train_data['image'].progress_apply(update_str_to_array)
test_data['image'] = test_data['image'].progress_apply(update_str_to_array)

# create ground truth for each class of image
# every image in batch can belong to 1 single class
ground_truths = {'airplane': torch.tensor([0], dtype=torch.long),
                'automobile': torch.tensor([1], dtype=torch.long),
                'bird': torch.tensor([2], dtype=torch.long),
                'cat': torch.tensor([3], dtype=torch.long),
                'deer': torch.tensor([4], dtype=torch.long),
                'dog': torch.tensor([5], dtype=torch.long),
                'frog': torch.tensor([6], dtype=torch.long),
                'horse': torch.tensor([7], dtype=torch.long),
                'ship': torch.tensor([8], dtype=torch.long),
                'truck': torch.tensor([9], dtype=torch.long)
}

train_data['ground_truth'] = train_data['label']
train_data['ground_truth'] = train_data['ground_truth'].apply(lambda x: ground_truths[x])

test_data['ground_truth'] = test_data['label']
test_data['ground_truth'] = test_data['ground_truth'].apply(lambda x: ground_truths[x])

# %%
######################################################
############ create your own dataset as below
class ImageCaptionDataset(Dataset):
    def __init__(self, df, preprocess):
        # list to path of images
        self.images = df["image"].tolist()
        # you can tokenize everything at once here, it will slow at beginning, or tokenize in the training loop
        self.caption = clip.tokenize(df["label"].tolist())
        self.ground_truth = df['ground_truth'].tolist()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        # load image using PIL and process using CLIP
        image = self.images[idx].reshape(img_height, img_width, 3)
        image = self.preprocess(Image.fromarray(np.uint8(image)))
        
        # instead of loading the image, directly preprocess
        caption = self.caption[idx]
        ground_truth = self.ground_truth[idx].squeeze()
        return image, caption, ground_truth

# get device and load pretrained model
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False) # Must set jit=False for training

# generate train dataset
train_dataset = ImageCaptionDataset(train_data, preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)

# generate test dataset
test_dataset = ImageCaptionDataset(test_data, preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# %%
####################################
############ peek into the dataset
image, label, gt = train_dataset[0]
print(label.shape, image.shape, gt)

images, labels, gt = next(iter(train_dataloader))
print(images[0].sum())
print(labels.shape, images.shape, gt.shape, gt)

# %%
#####################################
#### define training parameters
#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()
else:
    # precautional step, although unnecessary
    clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# mae = nn.L1Loss()

# Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = optim.Adam(model.parameters(), 
                       lr=5e-5,
                       #betas=(0.9,0.98),
                       #eps=1e-6,
                       #weight_decay=0.2
                       )

# %%
# check initial loss of model
images, labels, gt = next(iter(train_dataloader))
print(gt, images.sum())
images = images.to(device)
texts = labels.to(device)
gt = gt.to(device)

# get logits and convert to probabilities
logits_per_image, logits_per_text = model(images, texts)
#image_probs = logits_per_image.softmax(dim=-1)
#text_probs = logits_per_text.softmax(dim=-1)

# loss type 1
loss = (loss_img(logits_per_image, gt) + loss_txt(logits_per_text, gt))/2
print('loss 1:', loss)

# loss type 2
ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
print('loss 2:', loss)
# %%
######################################3
######### start model training
import logging

logging.basicConfig(filename='image_classification_.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('Initializing Script')
logging.warning('Logging enabled')

# define number of epochs
Epochs = 50
model.to(device)

df = pd.DataFrame()
df['epoch'] = -1
df['train_loss'] = float('inf')
df['val_loss'] = float('inf')

best_val_loss = float('inf')
# add your own code to track the training progress.
for epoch in tqdm(range(Epochs)):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        #torch.cuda.empty_cache()
        images, texts, gt = batch
        
        # transfer to device
        images = images.to(device)
        texts = texts.to(device)
        ground_truth = gt.to(device)
        # ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        #torch.cuda.empty_cache()
        # forward pass
        logits_per_image, logits_per_text = model(images, texts)
        loss = (loss_img(logits_per_image, ground_truth))# + loss_txt(logits_per_text, ground_truth))/2        
        
        ##############################
        ###############################
        ######################################start here to check how loss is going
        #print(loss, train_loss)
        print(loss)
        logging.info(str(epoch)+'\t'+str(loss)+'\t'+str(train_loss))
        train_loss += loss.item()

        # back propagation
        loss.backward()
        
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        optimizer.zero_grad()

    print(f'Training loss after Epoch - {epoch}: ', train_loss/len(train_dataloader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            torch.cuda.empty_cache()
            images, texts, gt = batch
        
            # transfer to device
            images = images.to(device)
            texts = texts.to(device)
            ground_truth = gt.to(device)

            torch.cuda.empty_cache()
            # forward pass
            logits_per_image, logits_per_text = model(images, texts)
            loss = (loss_img(logits_per_image, ground_truth))# + loss_txt(logits_per_text, ground_truth))/2        
            val_loss += loss.item()
    
        print(f"Validation loss after epoch {epoch}:", val_loss/len(test_dataloader))

    temp = pd.DataFrame()
    temp['epoch'] = epoch
    temp['train_loss'] = train_loss/len(train_dataloader)
    temp['val_loss'] = val_loss/len(test_dataloader)
    df = pd.concat(df, temp)
    df.to_csv('model_evaluation.csv', index=False)
    
    if val_loss<best_val_loss:
        best_val_loss = val_loss
        print('saving model.')
        # save the model
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, "image_classification/model_cifar.pt") #just change to your preferred folder/filename

# %%
# check loss of model after training
images, labels, gt = next(iter(train_dataloader))
images = images.to(device)
texts = labels.to(device)
gt = gt.to(device)

# get logits and convert to probabilities
logits_per_image, logits_per_text = model(images, texts)
#image_probs = logits_per_image.softmax(dim=-1)
#text_probs = logits_per_text.softmax(dim=-1)

# loss type 1
loss = (loss_img(logits_per_image, gt) + loss_txt(logits_per_text, gt))/2
print('loss 1:', loss)

# loss type 2
ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
print('loss 2:', loss)

# %%
##########################################
########## inference after training
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load("image_classification/model_cifar.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
#checkpoint['model_state_dict']["input_resolution"] = model.input_resolution # default is 224
#checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
#checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

#model.load_state_dict(checkpoint['model_state_dict'])

# run prediction on sample data
label_definition = {
                        0:'airplane',
                        1:'automobile',
                        2:'bird',
                        3:'cat',
                        4:'deer',
                        5:'dog',
                        6:'frog',
                        7:'horse',
                        8:'ship',
                        9:'truck'
                        }
idx = 54
image = preprocess(Image.fromarray(x_train[idx])).unsqueeze(0).to(device)
text_options = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck']

text = clip.tokenize(text_options).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = probs.flatten()

results = dict(zip(text_options, probs))
results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("Label probs:", *results, sep='\n')  # prints: [[0.9927937  0.00421068 0.00299572]]
print("Actual Value:", label_definition[y_train[idx][0]])
Image.fromarray(x_train[idx])


# %%
# images, text from data_loader
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
for i, prob in enumerate(probs):    
    results = dict(zip(text_options, prob))
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("Label probs:", *results, sep='\n')  # prints: [[0.9927937  0.00421068 0.00299572]]
    print(text_options)
    #print("Actual Value:", label_definition[y_train[idx][0]])

