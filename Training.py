import os
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F1
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  
from PIL import Image
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1
device = torch.device('mps')

# Please modify the following paths to the correct paths on your machine!!!
print('Welcome to the Road Lane Detection from Noisy Satellite Imag1es Program.')
i = eval(input('Please input the training mode, Debug mode if i = 0, Full mode if i = 1, Half mode if i = 2.'))
output_dir = 'output'
val_dir = 'val_output'
val_img_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/pic/good1/images'
val_gt_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Train/Train/labels'
train_gt_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Train/Train/labels'
if(i != 1): test_dir = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Test_studentversion/images_test'
else: test_dir = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Test_studentversion/images'
if(i == 1): train_img_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Train/Train/images'
elif(i == 2): train_img_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Train/Train/images_half'
else: train_img_folder = '/Users/jerrykong/FilesCenter/kzr/1HKPOLYU/Projects/example/Train/Train/images_test'

# Custom dataset for image segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)#.replace('sat.jpg', 'mask.png'))
        mask_path = os.path.join(self.mask_dir, img_name.replace('sat.jpg', 'mask.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)

        return image, mask, img_name
    
class TestDataset:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        image = preprocess(image)
        
        return image, self.images[index]

# Use plt.imshow to visualize images and masks
def show_image_mask(num1, num2, num3, num4, label):
    plt.clf() # Clean the current figure
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.plot(num1, label='Accuracy')
    plt.plot(num2, label='Loss')
    plt.plot(num3, label='BER')
    plt.plot(num4, label='MAE')
    # plt.plot(num5.cpu(), label='BCE')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.show()

# Simple CNN model for segmentation
class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2(self.conv2(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv4(self.conv4(self.conv3(x)))
        x = self.maxpool(x)
        x = self.conv6(self.conv6(self.conv5(x)))
        x = self.maxpool(x)
        x = torch.tanh(x) # Add a non-linearity to the model
        x = self.sigmoid(x)
        x = self.conv8(self.conv8(self.conv7(x)))
        x = self.upsample(x)
        x = self.conv6(self.conv6(self.conv9(x)))
        x = self.upsample(x)
        x = self.conv4(self.conv4(self.conv10(x)))
        x = self.upsample(x)
        x = self.conv11(x)
        return x

# Define transformations for images and masks
transform = {
    'image': transforms.Compose([
        transforms.Resize((256,256)),  # Resize images
        transforms.ToTensor()
    ]),
    'mask': transforms.Compose([
        transforms.Resize((256,256)),  # Resize masks to match images
        transforms.ToTensor()
    ])
}

# Data augmentation and normalization for training
def calculate_accuracy(pred_mask, true_mask): # Accuracy
    pred_mask = pred_mask.flatten() # Flatten the mask
    true_mask = true_mask.flatten()
    # print('max:', max(pred_mask), max(true_mask))
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5
    correct = torch.sum(pred_mask == true_mask).item()
    total = pred_mask.shape[0]
    return correct / total

def calculate_mae(image1, image2): # Mean Absolute Error (MAE)
    img1 = np.array(image1, dtype=np.float32)
    img2 = np.array(image2, dtype=np.float32)
    mae = np.mean(np.abs(img1 - img2))
    return mae

def calculate_ber(mask, image): # Balanced Error Rate (BER)
    mask_arr = np.array(mask, dtype=np.bool_)
    image_arr = np.array(image, dtype=np.bool_)
    TP = np.sum(mask_arr & image_arr)
    TN = np.sum(~mask_arr & ~image_arr)
    FP = np.sum(~mask_arr & image_arr)
    FN = np.sum(mask_arr & ~image_arr)
    P = TP + FN
    N = TN + FP
    TPR = TP / P if P != 0 else 0
    TNR = TN / N if N != 0 else 0
    ber = 1 - 0.5 * (TPR + TNR)
    return ber

def test_model(model_path):
    test_data = TestDataset(test_dir)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    model = SimpleSegmentationModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    print('Testing model...')
    with torch.no_grad():
        for images, image_names in test_loader:
            t = 0
            images = images.to(device)
            outputs = model(images)
            for output, image_name in zip(outputs, image_names):
                t += 1
                output = output.squeeze().cpu().numpy()
                output = (output * 255).astype('uint8')
                output_image = Image.fromarray(output)
                output_path = os.path.join(output_dir, image_name)
                output_image.save(output_path)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def dice_loss(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def Validation_Model(model):
    print('Validating model...')
    val_data = SegmentationDataset(val_img_folder, val_gt_folder, transform=transform)
    val_loader = DataLoader(val_data, batch_size=16)
    total_ber = 0
    total_mae = 0
    total_loss = 0
    total_acc = 0
    total_num = len(val_loader)
    model.eval()
    for images, masks, image_names in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = dice_loss(outputs, masks) # Adjust mask dimensions if necessary
        total_loss += loss.item()
        total_ber += calculate_ber(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
        total_mae += calculate_mae(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
        total_acc += calculate_accuracy(torch.from_numpy(outputs.cpu().detach().numpy()), torch.from_numpy(masks.cpu().detach().numpy()))
    sleep(0.5)
    print(f'Validation Stage --> Validation_Loss: {total_loss/total_num:.4f}  Value_Accuracy: {total_acc/total_num:.4f}  Value_BER: {total_ber/total_num:.4f}  Value_MAE: {total_mae/total_num:.4f}')
    os.makedirs(val_dir, exist_ok=True)
    for output, image_name in zip(outputs, image_names):
        output = output.squeeze().cpu().detach().numpy()
        output = (output * 255).astype('uint8')
        val_image = Image.fromarray(output)
        val_path = os.path.join(val_dir, image_name)
        val_image.save(val_path)
    return total_loss/total_num

criterion1 = nn.BCELoss() # Binary Cross Entropy Loss
criterion2 = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits Loss
criterion3 = nn.CrossEntropyLoss() # Cross Entropy Loss
criterion4 = nn.MSELoss() # Mean Squared Error Loss
criterion5 = FocalLoss() # Focal Loss

# Function to save the model
def save_checkpoint(model, epoch, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

def main():
    if(not os.path.exists(train_img_folder)):
        print('Folder not exists')
    if(not os.path.exists(train_gt_folder)):
        print('Folder not exists')
    print("Start Training.....")
    print(f'Device: {device}')

    # Initialization
    train_data = SegmentationDataset(train_img_folder, train_gt_folder, transform=transform)
    train_loader = DataLoader(train_data, batch_size=16) # Adjust batch size
    model = SimpleSegmentationModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    checkpoint_path = ''
    arr_acc = []
    arr_loss = []
    arr_ber = []
    arr_mae = []
    total_val_loss = []
    total_train_loss = []
    num_epochs = 300
    for epoch in range(num_epochs):
        total_ber = 0
        total_mae = 0
        total_loss = 0
        total_acc = 0
        flag = 0
        total_num = len(train_loader)
        # print(len(train_loader))
        if (checkpoint_path != ''):
            model.load_state_dict(torch.load(checkpoint_path))
        model.train()
        progress_bar = tqdm(train_loader, unit='batch')
        progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        for images, masks, image_names in train_loader:
            # plt.imshow(images[0].permute(1, 2, 0))
            # plt.show()
            # plt.imshow(masks[0].squeeze(), cmap='gray')
            # plt.show()
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # print(max(outputs[0].squeeze().detach().cpu().numpy().flatten()))
            # plt.imshow(outputs[0].squeeze().detach().cpu(), cmap='gray')
            # plt.show()
            loss = dice_loss(outputs, masks)  # Adjust mask dimensions if necessary
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update()
            total_ber += calculate_ber(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
            total_mae += calculate_mae(outputs.cpu().detach().numpy(), masks.cpu().detach().numpy())
            total_acc += calculate_accuracy(torch.from_numpy(outputs.cpu().detach().numpy()), torch.from_numpy(masks.cpu().detach().numpy()))
        sleep(0.5)
        progress_bar.close()
        print(f'Training Stage --> Current_Learning_Rate: {scheduler.get_last_lr()[0]}  Training_Loss: {total_loss/total_num:.4f}  Value_Accuracy: {total_acc/total_num:.4f}  Value_BER: {total_ber/total_num:.4f}  Value_MAE: {total_mae/total_num:.4f}')
        # if(total_acc/total_num < 0.85):
            # torch.cuda.empty_cache()
            # if (epoch + 1) % 1 == 0:
            #     checkpoint_path = save_checkpoint(model, epoch + 1)
            # test_model(checkpoint_path)
            # checkpoint_path = ''
            # # restart training
            # model = SimpleSegmentationModel().to(device)
            # optimizer = optim.SGD(model.parameters(), lr=0.01)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        total_train_loss.append(total_loss/total_num)
        val_loss = Validation_Model(model)
        total_val_loss.append(val_loss)
        if(len(total_val_loss) > 5 and total_val_loss[-1] > total_val_loss[-2] and total_val_loss[-2] > total_val_loss[-3] and total_val_loss[-3] > total_val_loss[-4] and total_val_loss[-4] > total_val_loss[-5]):
            if(len(total_train_loss) > 5 and total_train_loss[-1] < total_train_loss[-2] and total_train_loss[-2] < total_train_loss[-3] and total_train_loss[-3] < total_train_loss[-4] and total_train_loss[-4] < total_train_loss[-5]):
                print('Overfitting happened!')
                checkpoint_path = f'checkpoint_{epoch - 5}.pth' # Load the model from 5 epochs ago
                flag = 1
        if(flag == 0):
            scheduler.step()
            if (epoch + 1) % 1 == 0:
                checkpoint_path = save_checkpoint(model, epoch + 1)
            arr_acc.append(total_acc/total_num)
            arr_loss.append(total_loss/total_num)
            arr_ber.append(total_ber/total_num)
            arr_mae.append(total_mae/total_num)
            #if(arr_acc.__len__() > 1):
                #show_image_mask(arr_acc, arr_loss, arr_ber, arr_mae, 'Multiple Line Plots')
            if(epoch + 1) % 10 == 0:
                test_model(checkpoint_path)
    print('Task completed!')

main()
