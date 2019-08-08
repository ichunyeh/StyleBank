import argparse
import sys
import os
import random
import math
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets

import args
from network import StyleBankNet
import utils
from cnn_vis.cnn_visualization import get_filter, get_featuremap, vis_filter, vis_featuremap, gmm

SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = args.device
if torch.cuda.is_available():
    MODEL_PATH = 'model.pth'
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    MODEL_PATH = 'model_cpu.pth'
# MODEL_PATH = ''


def menu():
    print('\nChoice...')
    print('1. stylize')
    print('2. stylize (style control)')
    print('3. get filter')
    print('4. get feature-map')
    print('5. visualize filter value')
    print('6. visualize feature-map value')
    print('7. finish!')
    print('8. save all images in one!')
    return input('Please choose the number: ')

def load_model():
    #################################
    # Load Dataset
    #################################
    style_dataset = datasets.ImageFolder(root=args.STYLE_IMG_DIR, transform=utils.img_transform)
    style_dataset = torch.cat([img[0].unsqueeze(0) for img in style_dataset], dim=0)
    style_dataset = style_dataset.to(device)
    print('dataloader done.')

    #################################
    # Define Model and Loss network (vgg16)
    #################################
    model = StyleBankNet(len(style_dataset)).to(device)

    if os.path.exists(args.GLOBAL_STEP_PATH):
        with open(args.GLOBAL_STEP_PATH, 'r') as f:
            global_step = int(f.read())
    else:
        raise Exception('cannot find global step file')
        # global_step = args.MAX_ITERATION
    if os.path.exists(args.MODEL_WEIGHT_PATH):
        model.load_state_dict(torch.load(os.path.join(args.MODEL_WEIGHT_DIR, MODEL_PATH)))
    else:
        raise Exception('cannot find model weights')
    print('network done. ({}, {}) '.format(args.MODEL_WEIGHT_DIR, global_step))
    model.eval()

    # # save as cpu version
    # m = deepcopy(model.cpu())
    # torch.save(m.state_dict(), os.path.join(args.MODEL_WEIGHT_DIR, 'model_cpu.pth'))

    return model

def stylized(img, model, save_path, style_id=[11, 14]):
    test_imgs = img.to(device)
    data = torch.zeros([args.batch_size, 3, args.IMG_SIZE, args.IMG_SIZE])

    for i, test_img in enumerate(test_imgs):
        output_ae_test = model(test_img.expand_as(data))[0]
        test_img_list = [test_img.cpu(), output_ae_test.cpu().detach()]
        outputs_test = model(test_img.expand_as(data), style_id)
        for j in range(len(style_id)):
            test_img_list.append(outputs_test[j].cpu().detach())
        utils.saveimgs(save_path, test_img_list, ['original', 'reconstructed'] + style_id)
    print('stylize finished')

def stylized2(img, model, save_path, style_id=[11, 14]):
    test_img = img.to(device)
    data = torch.zeros([args.batch_size, 3, args.IMG_SIZE, args.IMG_SIZE])
    output_test = model(test_img.expand_as(data), style_id)[0].cpu().detach()
    utils.showimg(output_test, False, save_path)
    print('stylize finished')

if __name__ == '__main__':
    orig_model = load_model()

    # load image
    img_path = 'data/test/husky.jpg'
    print(img_path)
    # img_path = 'data/style/style/14.jpg'
    img_name = img_path[img_path.rfind('/') + 1: img_path.rfind('.')]
    original_image = Image.open(img_path).convert('RGB')
    prep_img = utils.preprocess_image(original_image)

    save_dir = 'test/' + img_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_f_dir = 'test/' + '00_f'
    if not os.path.exists(save_f_dir):
        os.makedirs(save_f_dir)

    while True:
        choice = menu()
        if choice == '1':
            style_id_string = input('Enter the style image id(0~15) (maximun=4):')
            style_id = list(map(int, style_id_string.split(' ')))
            save_path = save_dir + '/stylized_id-' + style_id_string.replace(' ', '_')
            stylized(prep_img, orig_model, save_path, style_id)
        elif choice == '2':
            # filter clustering
            target_layer = input('Enter the target layer, e.g."style_bank.14.0": ')
            style_id = target_layer.split('.')[1]
            k = input('Enter the number of clusters, e.g."10": ')
            k = int(k)
            f_flatten = get_filter(orig_model, target_layer)
            # clustering
            centers, labels = gmm(f_flatten, k)
            masks = np.zeros(shape=(k, f_flatten.shape[0], f_flatten.shape[1]))
            for cid in range(k):
                for pid in range(f_flatten.shape[0]):
                    if labels[pid] == cid:
                        masks[cid][pid] = centers[cid]
            # plot
            total_mean = np.zeros_like(masks[0])
            for cid, mask in enumerate(masks):
                if cid == 2 or cid == 7 or cid == 11 or cid == 17 or cid == 3 or cid == 18:
                    total_mean = total_mean + mask
                # mask = mask.reshape(128, 128, 3, 3)
                # # mask = mask[np.newaxis, :]
                # mask = torch.FloatTensor(mask)
                # # print(mask.shape)
                # save_path = 'test2/' + img_name + '/' + target_layer.replace('.', '_') + '_k' + str(k) + '_c' + str(cid)

                # # stylized
                # new_model = deepcopy(orig_model)
                # new_model.state_dict()['style_bank.'+style_id+'.0.conv2d.weight'].copy_(mask)
                # # style_id = list(map(int, style_id.split(' ')))
                # stylized2(prep_img, new_model, save_path, [int(style_id)])
                # print('filter visualized')
                # del new_model
            total_mean = total_mean.reshape(128, 128, 3, 3)
            # mask = mask[np.newaxis, :]
            total_mean = torch.FloatTensor(total_mean)
            # print(mask.shape)
            save_path = 'test2/' + img_name + '/' + target_layer.replace('.', '_') + '_k' + str(k) + '2&3&7&11&17&18'
            new_model = deepcopy(orig_model)
            new_model.state_dict()['style_bank.'+style_id+'.0.conv2d.weight'].copy_(total_mean)
            stylized2(prep_img, new_model, save_path, [int(style_id)])
            print('filter visualized')

        elif choice == '3':
            target_layer = input('Enter the target layer, e.g."style_bank.14.0": ')
            save_path = save_f_dir + '/' + target_layer.replace('.', '_')
            get_filter(orig_model, target_layer, isSave=save_path)
        elif choice == '4':
            target_layer = input('Enter the target layer, e.g."style_bank.14.0": ')
            save_path = save_dir + '/' + target_layer.replace('.', '_') + '_fm'
            if target_layer.split('.')[0] == 'decoder_net':
                style_id = input('Enter the style id, enter \'None\' if not through style_bank, e.g."14": ')
                if style_id == 'None':
                    style_id = None
                else:
                    save_path = save_dir + '/' + target_layer + '_c' + style_id + '_fm'
            elif target_layer.split('.')[0] == 'style_bank':
                style_id = target_layer.split('.')[1]
            else:
                style_id = None
            get_featuremap(prep_img, orig_model, target_layer, style_id, isSave=save_path)
        elif choice == '5':
            target_layer = input('Enter the target layer, e.g."style_bank.14.0": ')
            target_filter = input('Enter the target filter, e.g."1": ')
            save_path = save_f_dir + '/' + target_layer.replace('.', '_') + '_' + target_filter
            vis_filter(orig_model, save_path, target_layer, int(target_filter))
        elif choice == '6':
            target_layer = input('Enter the target layer, e.g."style_bank.14.0": ')
            if target_layer.split('.')[0] == 'decoder_net':
                style_id = input('Enter the style id, enter \'None\' if not through style_bank, e.g."14": ')
                if style_id == 'None':
                    style_id = None
                else:
                    save_path = save_dir + '/' + target_layer + '_c' + style_id + '_fm'
            elif target_layer.split('.')[0] == 'style_bank':
                style_id = target_layer.split('.')[1]
            else:
                style_id = None
            mask = input('Enter the mask path or cluster number, enter \'None\' if not mask, e.g."test/japan/encoder_net_8_fm/gmm_5/c.txt": ')
            if mask == 'None':
                # mask = None
                save_path = save_dir + '/' + target_layer.replace('.', '_') + '_fm'
                vis_featuremap(img_path, prep_img, orig_model, save_path, target_layer, style_id)
            elif '.txt' in mask:  # loadtxt(clustering result)
                mask_path = mask
                mask = np.loadtxt(mask_path, dtype=np.float)
                print('size:', mask.shape)
                fm_flatten = get_featuremap(prep_img, orig_model, target_layer, style_id)
                if fm_flatten.shape != mask.shape:
                    print('wrong mask size!')
                    break
                k = int(mask_path[mask_path.rfind('gmm_')+4:mask_path.rfind('/c.txt')])
                # # 使用全部群的群中心作為mask
                # mask = mask.reshape(int(math.sqrt(mask.shape[0])), int(math.sqrt(mask.shape[0])), -1)
                # mask = mask.reshape(mask.shape[2], mask.shape[0], mask.shape[1])
                # mask = mask[np.newaxis, :]
                # mask = torch.FloatTensor(mask)
                # save_path = save_dir + '/' + target_layer.replace('.', '_') + '_k' + str(k)
                # vis_featuremap(img_path, prep_img, orig_model, save_path, target_layer, style_id, mask=mask)
                # 使用各群的群中心作為mask
                means = []
                for i in range(mask.shape[0]):
                    if not any(np.array_equal(mask[i, :], m) for m in means):
                        means.append(mask[i, :])
                print(len(means))
                masks = np.zeros(shape=(k, fm_flatten.shape[0], fm_flatten.shape[1]))
                for cid in range(k):
                    for pid in range(fm_flatten.shape[0]):
                        # print(mask[pid].shape)
                        # print(means[cid].shape)
                        if (mask[pid] == means[cid]).all():
                            masks[cid][pid] = means[cid]
                # plot
                for cid, mask in enumerate(masks):
                    mask = mask.reshape(int(math.sqrt(mask.shape[0])), int(math.sqrt(mask.shape[0])), -1)
                    mask = mask[np.newaxis, :]
                    mask = torch.FloatTensor(mask)
                    # print(mask.shape)
                    save_path = save_dir + '/' + target_layer.replace('.', '_') + '_k' + str(k) + '_c' + str(cid)
                    vis_featuremap(img_path, prep_img, orig_model, save_path, target_layer, style_id, mask=mask)

            else:  # clustering(sklearn)
                k = int(mask)
                fm_flatten = get_featuremap(prep_img, orig_model, target_layer, style_id)
                # clustering
                centers, labels = gmm(fm_flatten, k)
                masks = np.zeros(shape=(k, fm_flatten.shape[0], fm_flatten.shape[1]))
                for cid in range(k):
                    for pid in range(fm_flatten.shape[0]):
                        if labels[pid] == cid:
                            masks[cid][pid] = centers[cid]
                # plot
                for cid, mask in enumerate(masks):
                    mask = mask.reshape(int(math.sqrt(mask.shape[0])), int(math.sqrt(mask.shape[0])), -1)
                    mask = mask[np.newaxis, :]
                    mask = torch.FloatTensor(mask)
                    # print(mask.shape)
                    save_path = save_dir + '/' + target_layer.replace('.', '_') + '_k' + str(k) + '_c' + str(cid)
                    vis_featuremap(img_path, prep_img, orig_model, save_path, target_layer, style_id, mask=mask)

        elif choice == '7':
            sys.exit()
        elif choice == '8':
            # ordering
            # save
            imgs = [Image.open(save_f_dir+'/encoder_net.3_' + str(i) + '.jpg').convert('RGB') for i in range(64)]

            num = len(imgs)
            row = 8
            col = 8
            plt.figure(figsize=(8, 8))
            for i, img in enumerate(imgs):
                plt.subplot(row, col, i+1)
                plt.axis('off')
                plt.imshow(img)
            plt.tight_layout()
            plt.savefig(save_f_dir + '/encoder_net_3')
        else:
            print('wrong choice!')
