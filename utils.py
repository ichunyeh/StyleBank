import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import args

# use on DataLoader(): batch
class Resize(object):
    '''
    Resize with aspect ration preserved.
    '''
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        m = min(img.size)
        new_size = (int(img.size[0] / m * self.size), int(img.size[1] / m * self.size))
        return img.resize(new_size, resample=Image.BILINEAR)
img_transform = transforms.Compose([
    Resize(args.IMG_SIZE),
    transforms.RandomCrop(args.IMG_SIZE),
    transforms.ToTensor(),  # convert a PIL Image or numpy.ndarray 0-255 (h, w, ch) to tensor 0-1 (ch, h, w)

    # not needed, LossNetwork have used 'Normalization'
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) # normalize to -1-1
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize the image using the mean and variance computed on the ImageNet dataset.
])

# use on Image.open(): one
def preprocess_image(pil_img, resize_img=True, img_size=args.IMG_SIZE):
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


def saveimgs(file_path, imgs, titles):
    '''
    Put many pytorch image tensors on one figure and save it.
    '''
    dir = file_path[:file_path.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)
    num = len(imgs)
    row = 2
    col = (num+1) // row
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(row, col, i+1)
        plt.axis('off')
        plt.title(title)
        showimg(img)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
def showimg(img, isShow=False, isSave=False, isResize=False):
    '''
    Input a pytorch image tensor with size (channel, height, width) and display it.
    '''
    img = img.clamp(min=0, max=1)
    img = img.cpu().detach().numpy().transpose(1, 2, 0) # (ch, h, w) -> (h, w, ch)
    if isResize:
        from skimage import transform
        img = transform.resize(img, isResize)
    plt.imshow(img)
    if isShow:
        plt.axis('off')
        plt.show()
        plt.close()
    if isSave:
        plt.axis('off')
        fig = plt.gcf()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig(isSave, bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)
        plt.close()

def adjust_learning_rate(optimizer, step):
    '''
    learning rate decay
    '''
    lr = max(args.lr * (0.8 ** (step)), 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_sid_batch(style_id_seg, batch_size):
    ret = style_id_seg
    while len(ret) < batch_size:
        ret += style_id_seg
    return ret[:batch_size]