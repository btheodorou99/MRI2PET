import os
import ants
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from ..config import MRI2PETConfig

config = MRI2PETConfig()
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(style_img, content_img):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # normalization module
    normalization = Normalization().to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            continue

        model.add_module(name, layer)
        model = model.to(device)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(content_img, style_img, input_img, num_steps=250, 
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(style_img, content_img)
    
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    # print('Optimizing..')
    for step in tqdm(range(num_steps), desc='Steps', leave=False):

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            # if step % 50 == 0:
                # print("step {}:".format(step))
                # print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                #     style_score.item(), content_score.item()))
                # print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def run_style_transfer_on_slices(content_slices, style_slices, num_steps=250, style_weight=1000, content_weight=1):
    """Run style transfer on each slice."""
    content_img = content_slices.unsqueeze(1).repeat(1, 3, 1, 1)
    style_img = style_slices.unsqueeze(1).repeat(1, 3, 1, 1)
    input_img = content_img.clone()
    styled_img = run_style_transfer(content_img, style_img, input_img, num_steps, style_weight, content_weight)
    styled_img = styled_img.mean(dim=1)
    # styled_img = (styled_img - styled_img.min()) / (styled_img.max() - styled_img.min())
    return styled_img

def load_image(path):
    """Load an image as a torch tensor."""
    image = np.load(path)
    image = image.transpose((2, 0, 1)) # Convert to CxHxW
    image = torch.from_numpy(image).float() # Convert to torch tensor
    return image

def save_image(tensor, path):
    """Save a torch tensor as an image."""
    image = tensor.cpu().clone()       # Clone the tensor to not do changes on it
    image = image.numpy()              # Convert to numpy array
    image = image.transpose((1, 2, 0)) # Convert back to HxWxC
    np.save(path, image)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all PET images as style images
style_img = ants.image_read("./src/data/petTemplate.nii")
style_img = style_img.numpy()
style_img = (style_img - style_img.min()) / (style_img.max() - style_img.min())
style_img = style_img.transpose((2, 0, 1))
style_img = torch.from_numpy(style_img).float().to(device)

# Apply style transfer for each content image
content_dir = "/data/CARD_AA/data/ADNI/MRI_Pretrain/"
output_dir = "/data/CARD_AA/data/ADNI/MRI_StyleTransfer/"
os.makedirs(output_dir, exist_ok=True)

content_imgs = os.listdir(content_dir)
random.shuffle(content_imgs)

for content_file in tqdm(content_imgs, desc='MRI Images'):
    if os.path.exists(os.path.join(output_dir, content_file)):
        continue
    
    content_img = load_image(os.path.join(content_dir, content_file)).to(device)
    output = run_style_transfer_on_slices(content_img, style_img)
    save_image(output, os.path.join(output_dir, content_file))
