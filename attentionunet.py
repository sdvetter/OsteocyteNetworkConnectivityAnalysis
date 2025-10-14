import os
import time
import monai
import numpy as np
import einops
import tifffile
import matplotlib.pyplot as plt
from PIL import Image
import glob
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
import tifffile
from monai.utils import first, set_determinism

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
import cv2
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import scipy
import torch
import torchvision
from torchvision import transforms

class LCNDataset(Dataset):
    def __init__(self,path_variables, transform, is_test = False) :
        self.label_folder = sorted(glob.glob(os.path.join( path_variables[0], path_variables[1])))
        self.images_folder = sorted(glob.glob(os.path.join( path_variables[2], path_variables[3])))
        self.transform = transform
        self.is_test = is_test
       
        
    def __len__(self):
        return len(self.images_folder)
    
    def transform_label(self, label):

        osteocyte_label = label[1,:,:]
        dendrite_label = label[0,:,:]
   

        ## Turn pixel values to ones
        osteocyte_label = np.clip(osteocyte_label, 0, 1)
        dendrite_label = np.clip(dendrite_label, 0, 1)
        background_label = np.zeros(osteocyte_label.shape) # to device

        ## If you want to train with dilated dendrites (make them thicker) uncomment 3 next lines
        kernel = np.ones((2,2))
        dendrite_label = cv2.dilate(np.expand_dims(dendrite_label, 0), kernel)
        dendrite_label = np.squeeze(dendrite_label, 0)

        ## Remove overlapping Ostoecyte and Dendrite labels, turn osteocyte = 0 wherever they overlap
        osteocyte_label = osteocyte_label - osteocyte_label * dendrite_label
    
        ## 0 = background, 1 = Osteocyte, 2 = Dendrite
        y = background_label + osteocyte_label + dendrite_label * 2
        return y

    def __getitem__(self, idx):
        image_path = self.images_folder[idx]
    
        # Load image and labels
        img = Image.open(image_path)

        # Load the next 3 labels
        labels = np.zeros((3,512,512))
        labels_idx = 0
        
        for i in range(3*idx, 3*idx+3):
            next_label_path = self.label_folder[i]
            next_label = Image.open(next_label_path)
            labels[labels_idx,:,:] = next_label
            labels_idx += 1

        label = self.transform_label(labels)
        label = torch.Tensor(label)
        img = transforms.functional.pil_to_tensor(img)
        
        img = torch.squeeze(img, axis=0)
        img = img.float()
        
        ## Perform transformations 
        if self.transform and self.is_test: 
            img,label = torch.unsqueeze(img, axis = 0),torch.unsqueeze(label, axis = 0)
            img = self.transform['test'](img)
        else: 
            img_lab = torch.stack((img, label), axis=0)
            
            img_lab = self.transform['train'](img_lab)
            img,label = img_lab[0], img_lab[1]
            
            img,label = torch.unsqueeze(img, axis = 0),torch.unsqueeze(label, axis = 0)
            img = self.transform['image'](img)
       
        
        return {"image": img, "label": label}
    

    
data_transforms = {
    'train': transforms.Compose([
        #transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomCrop(448),
        #torchvision.transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=30, translate=(0.1,0.2), scale=(0.9,1.1), shear=20)        
    ]),
    'image': transforms.Compose([
        #torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation =0.5, hue =0.4),
        torchvision.transforms.Normalize(0.5, 0.2),
      
    ]),
    'test': transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(0.5, 0.2)
    ])
}


def length_regularization_loss(logits: torch.Tensor, class_index=2, penalty_factor=0.25) -> torch.Tensor:
    """
    Computes the length regularization loss for dendrites by penalizing short, isolated components.
    :param logits: Model's raw output tensor of shape [batch_size, num_classes, height, width]
    :param class_index: Index of the class (dendrites) you want to segment
    :param penalty_factor: Controls how much smaller components are penalized.
    :return: A scalar tensor representing the loss.
    """
    # Apply softmax to logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Get the most probable class for each pixel
    preds = torch.argmax(probs, dim=1)
    
    # Create a binary mask for the specified class
    segmented_dendrites = (preds == class_index).float()
    
    # Move tensor to CPU and convert to numpy for processing
    segmented_np = segmented_dendrites.cpu().numpy()
    batch_size = segmented_np.shape[0]
    
    total_penalty = 0

    # Loop through batch and calculate penalty
    for i in range(batch_size):
        labeled, num_features = scipy.ndimage.label(segmented_np[i])
    
        # For each component, calculate its size
        for j in range(1, num_features + 1):  # Components are labeled from 1 to num_features
            component_size = scipy.ndimage.sum(segmented_np[i], labeled, j)
            
            # Apply an inverse penalty; smaller components lead to larger penalties
            total_penalty += (1 / (component_size ** penalty_factor))
    
    return torch.tensor(float(total_penalty/512), requires_grad=True)


def num_dendrite_loss(logits: torch.Tensor, class_index=2.0) -> torch.Tensor:
    """
    Computes the length regularization loss for dendrites by penalizing short, isolated components.
    :param logits: Model's raw output tensor of shape [batch_size, num_classes, height, width]
    :param class_index: Index of the class (dendrites) you want to segment
    :param penalty_factor: Controls how much smaller components are penalized.
    :return: A scalar tensor representing the loss.
    """
    # Apply softmax to logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Get the most probable class for each pixel
    preds = torch.argmax(probs, dim=1)
    
    # Create a binary mask for the specified class
    segmented_dendrites = (preds == class_index).float()
    
    # Move tensor to CPU and convert to numpy for processing
    segmented_np = segmented_dendrites.cpu().numpy()
    batch_size = segmented_np.shape[0]
    
    total_penalty = 0

    # Loop through batch and calculate penalty
    for i in range(batch_size):
        labeled, num_features = scipy.ndimage.label(segmented_np[i])
    
        # Apply an inverse penalty; smaller components lead to larger penalties
        if (num_features < 700): 
            total_penalty += num_features*0.5
        else:   
            total_penalty += num_features*0.1
    
    return torch.tensor(float(total_penalty), requires_grad=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path_var = [r"training_data/LCN_labels","*.ome.tiff", r"training_data/LCN_images", "*.tif"]
val_path_var = [r"training_data/LCN_labels_val", "*.ome.tiff", r"training_data/LCN_images_val", "*.tif"]

dataset = LCNDataset(path_variables= train_path_var, transform = data_transforms)
val_ds = LCNDataset(path_variables=val_path_var, transform = data_transforms, is_test = True)

train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=8, num_workers=3, shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_ds, batch_size=7, num_workers=2, shuffle = True)

####
## Model parameters
####
spatial_dims = 2
in_channels = 1       # Grayscale images have 1 channel
out_channels = 3      # You want to classify pixels into 3 classes
channels = (16, 32, 64, 128, 256)  # Number of channels at each depth level. You can modify this based on your requirement.
strides = (2, 2, 2, 2)  # Strides for each depth level. You can modify this based on your requirement.

# Increase the number of channels at each depth level
#channels = (32, 64, 128, 256, 512, 1024)
# You might want to maintain a pattern in strides, or adjust as needed
#strides = (2, 2, 2, 2, 2)

# Initialize the network
model = monai.networks.nets.AttentionUnet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=channels,
    strides=strides
)

model.to(device)
#wandb.watch(model, log="all")

# Define loss function

loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice =0.6, lambda_ce = 0.4)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)
max_epochs = 300
total_steps = max_epochs * len(train_loader)
warmup_steps = 0.10 * total_steps  

pct_start = warmup_steps / total_steps

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = warmup_steps, eta_min = 5e-6)

start_epoch = 0
val_interval= 1
dice_val_best = 0
dice_val = 0

epoch_loss_values = []
metric_values = []
for epoch in range(start_epoch, max_epochs):
    print(time.ctime(), "Epoch:", epoch)
    epoch_time = time.time()
    model.train()
    start_time = time.time()

    loss_epoch = 0
    for idx, batch in enumerate(train_loader):
        ## Training
        start_time = time.time()
        x, y = batch["image"], batch["label"]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y) # + length_regularization_loss(outputs) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_epoch += loss.item()
        
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(train_loader)),
            "loss: {:.4f}".format(loss),
            "time {:.2f}s".format(time.time() - start_time),
        )
        
        ## Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"], batch["label"]
                x, y = x.to(device), y.to(device)
        
                ## process validation images so they can be utilised to calculate dice score
                val_outputs = model(x)
                val_labels_list = decollate_batch(y)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                    
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                   
            dice_val = dice_metric.aggregate().item()
            dice_metric.reset()

            ## Saving model if the new dice score is better than previous highest
        if dice_val > dice_val_best:
            dice_val_best = dice_val
            torch.save(model.state_dict(),"attentionunet.pth")

            print( "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
        else:
            print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
 
    #wandb.log({"Training Loss": loss_epoch, "Validation Dice Value":dice_val})
    epoch_loss_values.append(loss_epoch)
    metric_values.append(dice_val)
   
    print(
        "Epoch training  {}/{}".format(epoch, max_epochs - 1),
        "loss: {:.4f}".format(loss_epoch),
        "time {:.2f}s".format(time.time() - epoch_time),
    )

epochs = range(1, max_epochs + 1)
wandb.finish()

plt.plot(epochs, epoch_loss_values, label='Loss per epoch')
plt.title('Loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("attentionunet_lcn_epoch_loss")
plt.clf()

plt.plot(epochs, metric_values, label='Average Dice score')
plt.title('Average Dice score')
plt.xlabel('Epochs')
plt.ylabel('Dice score')
plt.legend()
plt.savefig("attentionunet_lcn_dice_score")
