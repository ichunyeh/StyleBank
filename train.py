import time
import os
import random
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.utils as tvutils
from tensorboardX import SummaryWriter
import args
import utils
from network import LossNetwork, StyleBankNet, vgg16


########### Important ###########
args.continue_training = False
#################################
SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = args.device
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train():
    #################################
    # Load Dataset
    #################################
    content_dataset = datasets.ImageFolder(root=args.CONTENT_IMG_DIR, transform=utils.img_transform)
    content_dataloader = torch.utils.data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # test_imgs = datasets.ImageFolder(root=args.TEST_IMG_DIR, transform=utils.img_transform)
    # test_imgs = torch.cat([img[0].unsqueeze(0) for img in test_imgs], dim=0)
    # test_imgs = test_imgs.to(device)
    style_dataset = datasets.ImageFolder(root=args.STYLE_IMG_DIR, transform=utils.img_transform)
    style_dataset = torch.cat([img[0].unsqueeze(0) for img in style_dataset], dim=0)
    style_dataset = style_dataset.to(device)

    print('dataloader done')
    # # Display content images
    # for imgs, _ in content_dataloader:
    #     for i in range(args.batch_size):
    #         utils.showimg(imgs[i], True)
    #     break
    # # Display test images
    # for img in test_imgs:
    #     utils.showimg(img, True)
    # # Display style images
    # for img in style_dataset:
    #     utils.showimg(img, True)

    #################################
    # Define Model and Loss network (vgg16)
    #################################
    model = StyleBankNet(len(style_dataset)).to(device)

    if args.continue_training:
        if os.path.exists(args.GLOBAL_STEP_PATH):
            with open(args.GLOBAL_STEP_PATH, 'r') as f:
                global_step = int(f.read())
        else:
            raise Exception('cannot find global step file')
        if os.path.exists(args.MODEL_WEIGHT_PATH):
            model.load_state_dict(torch.load(args.MODEL_WEIGHT_PATH))
        else:
            raise Exception('cannot find model weights')
    else:
        if not os.path.exists(args.MODEL_WEIGHT_DIR):
            os.mkdir(args.MODEL_WEIGHT_DIR)
        if not os.path.exists(args.BANK_WEIGHT_DIR):
            os.mkdir(args.BANK_WEIGHT_DIR)
        global_step = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_ae = optim.Adam(model.parameters(), lr=args.lr)
    loss_network = LossNetwork().to(device)

    print('network done')
    # print(model.encoder_net)
    # from torchsummary import summary
    # summary(model, (3, 512, 512))
    # summary(vgg16, (3, 513, 513))

    #################################
    # Training
    #################################
    print(args.param_name)
    print('\n{}, Training({})...'.format(time.strftime('%H:%M:%S'), global_step))
    start_t = time.time()

    # writer = SummaryWriter(args.LOG_DIR)

    # [0, 1, 2, ... , N]
    style_id = list(range(len(style_dataset)))
    style_id_idx = 0
    style_id_seg = []
    for i in range(0, len(style_dataset), args.batch_size):
        style_id_seg.append(style_id[i: i+args.batch_size])

    l_sum = 0
    c_sum = 0
    s_sum = 0
    tv_sum = 0
    r_sum = 0
    while global_step <= args.MAX_ITERATION:
        for ii, (data,_) in enumerate(content_dataloader):
            global_step += 1
            data = data.to(device)
            # print(data.shape) # torch.Size([4, 3, 512, 512])
            # utils.showimg(data[0], True)
            batch_size = data.shape[0]
            if global_step % (args.T+1) != 0:
                style_id_idx += 1
                sid = utils.get_sid_batch(style_id_seg[style_id_idx % len(style_id_seg)], batch_size)

                optimizer.zero_grad()
                output_image = model(data, sid)
                content_score, style_score = loss_network(output_image, data, style_dataset[sid])
                content_loss = args.CONENT_WEIGHT * content_score
                style_loss = args.STYLE_WEIGHT * style_score

                diff_i = torch.sum(torch.abs(output_image[:, :, :, 1:] - output_image[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(output_image[:, :, 1:, :] - output_image[:, :, :-1, :]))
                tv_loss = args.REG_WEIGHT * (diff_i + diff_j)

                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward()
                optimizer.step()
                l_sum += total_loss.item()
                c_sum += content_loss.item()
                s_sum += style_loss.item()
                tv_sum += tv_loss.item()

            if global_step % (args.T+1) == 0: # auto-encoder
                optimizer_ae.zero_grad()
                output_image = model(data)
                r_loss = F.mse_loss(output_image, data)
                r_loss.backward()
                optimizer_ae.step()
                r_sum += r_loss.item()

            if global_step % 100 == 0:
                print('.', end='')
            if global_step % args.LOG_ITER == 0:
                print('gs: {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                    global_step / args.K,
                    time.strftime('%H:%M:%S'),
                    l_sum / 666, c_sum / 666, s_sum / 666, tv_sum / 666,
                    r_sum / 333
                ))
                # writer.add_scalar('c_loss', c_sum / 666, global_step)
                # writer.add_scalar('s_loss', s_sum / 666, global_step)
                # writer.add_scalar('tv_loss', tv_sum / 666, global_step)
                # writer.add_scalar('l_loss', l_sum / 666, global_step)
                # writer.add_scalar('r_loss', r_sum / 333, global_step)

                s_sum = 0
                c_sum = 0
                tv_sum = 0
                l_sum = 0
                r_sum = 0
                torch.cuda.empty_cache()
                # save whole model (including stylebank)
                torch.save(model.state_dict(), args.MODEL_WEIGHT_PATH)
                # save seperate part
                with open(args.GLOBAL_STEP_PATH, 'w') as f:
                    f.write(str(global_step))
                torch.save(model.encoder_net.state_dict(), args.ENCODER_WEIGHT_PATH)
                torch.save(model.decoder_net.state_dict(), args.DECODER_WEIGHT_PATH)
                for i in range(len(style_dataset)):
                    torch.save(model.style_bank[i].state_dict(), args.BANK_WEIGHT_PATH.format(i))

                #################################
                # Testing
                #################################
                x_test = data[0]
                y_test = model(x_test.expand_as(data))[0]
                imgs = [x_test.cpu(), y_test.cpu().detach()]
                ys_test = model(x_test.expand_as(data), sid)
                for i in range(len(sid)):
                    imgs.append(ys_test[i].cpu().detach())
                utils.saveimgs(args.OUTPUT_IMG_DIR + '/%05d.png' % global_step, imgs,
                               ['original', 'reconstructed'] + sid)
                # # tensorboard
                # for i, test_img in enumerate(test_imgs):
                #     test_img = test_img.to(device)
                #     y_test = model(test_img.expand_as(data))[0]
                #     imgs = [test_img.cpu(), y_test.cpu().detach()]
                #     ys_test = model(test_img.expand_as(data), style_id)
                #     for j in style_id:
                #         imgs.append(ys_test[j].cpu().detach())
                #     utils.saveimgs(args.OUTPUT_IMG_DIR + '/%06d_%02d.png' % (global_step, i), imgs, ['original', 'reconstructed'] + style_id)
                #     writer.add_image('Image_{}'.format(i), tvutils.make_grid(imgs), global_step)

            if global_step % args.ADJUST_LR_ITER == 0:
                lr_step = global_step / args.ADJUST_LR_ITER
                utils.adjust_learning_rate(optimizer, lr_step)
                new_lr = utils.adjust_learning_rate(optimizer_ae, lr_step)
                print('learning rate decay', new_lr)
    # writer.close()
    end_t = time.time()
    print('finished!')
    print('it spends {:.2f}s, ran {} epochs'.format(end_t-start_t, args.MAX_ITERATION))


if __name__ == '__main__':
    train()
                                                                                                                                                                                                                                                                                                                                                                                          