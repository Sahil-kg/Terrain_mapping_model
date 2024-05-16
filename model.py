import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import os
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.encoder2=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.encoder3=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.encoder4=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1), 
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        

        self.decoder1=nn.Sequential(
            nn.ConvTranspose2d(512,256,2,2),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.decoder2=nn.Sequential(
            nn.ConvTranspose2d(384,128,2,2),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.decoder3=nn.Sequential(
            nn.ConvTranspose2d(192,64,2,2),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.decoder4=nn.Sequential(
            nn.ConvTranspose2d(96,7,2,2),
            nn.Conv2d(7,7,3,1,1),
            nn.ReLU(inplace=True)
        )

        self.skip_connection1=nn.Sequential(
            nn.Conv2d(32,32,1,1),
            nn.ReLU(inplace=True)
        )
        self.skip_connection2=nn.Sequential(
            nn.Conv2d(64,64,1,1),
            nn.ReLU(inplace=True)
        )
        self.skip_connection3=nn.Sequential(
            nn.Conv2d(128,128,1,1),
            nn.ReLU(inplace=True)
        )
        self.skip_connection4=nn.Sequential(
            nn.Conv2d(256,256,1,1),
            nn.ReLU(inplace=True)
        )
    

    def forward(self,x):
        x1=self.encoder1(x)
        x2=self.encoder2(x1)
        x3=self.encoder3(x2)
        x4=self.encoder4(x3)

        x_skip1=self.skip_connection1(x1)
        x_skip2=self.skip_connection2(x2)
        x_skip3=self.skip_connection3(x3)
        x_skip4=self.skip_connection4(x4)

        x4_dec=torch.cat([x4,x_skip4],dim=1)
        x3_dec=self.decoder1(x4_dec)
        x3_dec=torch.cat([x3_dec,x_skip3],dim=1)
        x2_dec=self.decoder2(x3_dec)
        x2_dec=torch.cat([x2_dec,x_skip2],dim=1)
        x1_dec=self.decoder3(x2_dec)
        x1_dec=torch.cat([x1_dec,x_skip1],dim=1)
        output=self.decoder4(x1_dec)


        return F.softmax(output,dim=1)

def prepo(directory_path):
    preprocessed_images = []
    filenames=[]

    def preprocess_image(image_path, target_size=(512, 512)):
        image = Image.open(image_path)
        image = image.resize(target_size)
        image=image.convert("RGB")
        image = np.array(image)
        return image
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):  
            image_path = os.path.join(directory_path, filename)
            preprocessed_image = preprocess_image(image_path)
            preprocessed_images.append(preprocessed_image)
            filenames.append(filename)
    preprocessed_images=np.array(preprocessed_images)
    return preprocessed_images,filenames

def prepomask(directory_path,filenames_x_train):

    preprocessed_images = []
    labels = []

    def preprocess_image(image_path, target_size=(512, 512)):
        image = Image.open(image_path)
        image = image.resize(target_size)
        image=image.convert("RGB")
        image = np.array(image)
        return image
    
    for filename in filenames_x_train:
        if filename.endswith(".jpg"):  # Filter only image files
            name,extenson=os.path.splitext(filename)
            image_path = os.path.join(directory_path, name+'.png')
            preprocessed_image = preprocess_image(image_path)
            preprocessed_images.append(preprocessed_image)
            label = rgb_to_2D_label(preprocessed_image)
            labels.append(label) 

    preprocessed_images=np.array(preprocessed_images)  
    labels=np.array(labels)
    labels = np.expand_dims(labels, axis=3)
    return labels
    
class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train,self.filenames_x_train=prepo(x_train)
        self.y_train=prepomask(y_train,self.filenames_x_train)
        self.x_train = torch.from_numpy(self.x_train).permute(0, 3, 1, 2).float()
        self.y_train = torch.from_numpy(self.y_train).permute(0, 3, 1, 2).long()
        self.y_train = torch.squeeze(self.y_train, dim=1)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    
def train(model,x_train,y_train,criterion,optimizer,epochs):
    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        run_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader):
        
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {run_loss/len(train_loader)}")

def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    urban_land = [0,255,255]
    agricultural_land = [255,255,0]
    range_land = [255,0,255]
    water = [0,0,255]
    barren_land = [255,255,255]
    forest_land = [0,255,0]
    unknown=[0,0,0]
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == urban_land,axis=-1)] = 0
    label_seg [np.all(label==agricultural_land,axis=-1)] = 1
    label_seg [np.all(label==range_land,axis=-1)] = 2
    label_seg [np.all(label==water,axis=-1)] = 3
    label_seg [np.all(label==barren_land,axis=-1)] = 4
    label_seg [np.all(label==forest_land,axis=-1)] = 5
    label_seg [np.all(label==unknown,axis=-1)] = 6
    
    label_seg = label_seg[:,:,0]
    return label_seg

if __name__ == "__main__":

    mask_dataset=('/home/sahil/Desktop/training_data/masks')
    image=('/home/sahil/Desktop/training_data/images')

    model=UNET()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001)

    train(model,image,mask_dataset,criterion,optimizer,10)











