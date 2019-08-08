import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import torch
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def preprocess_image(pil_img, resize_img=True, img_size=512):
    # # mean and std list for channels (Imagenet)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_img:
        # pil_img.thumbnail((512, 512))
        pil_img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    im_as_arr = np.float32(pil_img)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        # im_as_arr[channel] -= mean[channel]
        # im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_img = copy.copy(im_as_var.data.numpy()[0])
    # for c in range(3):
    #     recreated_img[c] /= reverse_std[c]
    #     recreated_img[c] -= reverse_mean[c]
    recreated_img[recreated_img > 1] = 1
    recreated_img[recreated_img < 0] = 0
    recreated_img = np.round(recreated_img * 255)
    recreated_img = np.uint8(recreated_img).transpose(1, 2, 0)
    return recreated_img

def save_image(img, path):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            img = np.repeat(img, 3, axis=0)
        if img.shape[0] == 3:
            # Convert to values to range 1-255 and W,H,D
            img = img.transpose(1, 2, 0) * 255
        img = Image.fromarray(img.astype(np.uint8))
    img.save(path)

def save_gradient_images(gradient, save_path):
    # normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # save image
    path_to_file = os.path.join(save_path + '.jpg')
    save_image(gradient, path_to_file)

def cut_orig(img_path, masks):
    masks = convert_to_grayscale(masks)
    # mask = masks[0]
    # print(mask.shape)

    orig_img = Image.open(img_path).convert('RGBA')
    orig_img = orig_img.resize((512, 512), resample=Image.BILINEAR)

    def merge(img, mask):
        newdata = []
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r, g, b, a = img.getpixel((j, i))
                newdata.append((r, g, b, int(round(mask[i, j]*255))))
                # newdata.append((r, g, b, 255))
        newimg = Image.new(img.mode, img.size)
        newimg.putdata(newdata)
        return newimg
    out = merge(orig_img, masks[0])
    return out
