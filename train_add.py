import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import args
import utils
from network import LossNetwork, StyleBankNet


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
    style_dataset = datasets.ImageFolder(root=args.NEW_STYLE_IMG_DIR, transform=utils.img_transform)
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

    if os.path.exists(args.ENCODER_WEIGHT_PATH):
        model.encoder_net.load_state_dict(torch.load(args.ENCODER_WEIGHT_PATH))
    else:
        raise Exception('cannot find encoder weights')
    if os.path.exists(args.DECODER_WEIGHT_PATH):
        model.decoder_net.load_state_dict(torch.load(args.DECODER_WEIGHT_PATH))
    else:
        raise Exception('cannot find decoder weights')
    if not os.path.exists(args.NEW_BANK_WEIGHT_DIR):
        os.mkdir(args.NEW_BANK_WEIGHT_DIR)

    model.encoder_net.eval()
    model.decoder_net.eval()
    model.style_bank.train()
    # only update the bank
    optimizer = optim.Adam(model.style_bank.parameters(), lr=args.lr)
    loss_network = LossNetwork().to(device)

    print('network done')

    #################################
    # Training
    #################################
    print(args.param_name)
    print('\nTraining({})...'.format(time.strftime('%H:%M:%S')))
    start_t = time.time()

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

    global_step = 0
    LOG_ITER = int(args.LOG_ITER)

    while global_step <= args.MAX_ITERATION:
        for ii, (data, _) in enumerate(content_dataloader):
            global_step += 1
            data = data.to(device)
            batch_size = data.shape[0]

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

            if global_step % 100 == 0:
                print('.', end='')

            if global_step % LOG_ITER == 0:
                print('gs: {} {} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                    global_step / args.K,
                    time.strftime('%H:%M:%S'),
                    l_sum / LOG_ITER, c_sum / LOG_ITER, s_sum / LOG_ITER, tv_sum / LOG_ITER
                ))

                s_sum = 0
                c_sum = 0
                tv_sum = 0
                l_sum = 0
                torch.cuda.empty_cache()
                # save the bank
                for i in range(len(style_dataset)):
                    torch.save(model.style_bank[i].state_dict(), args.NEW_BANK_WEIGHT_PATH.format(i))

            if global_step % args.ADJUST_LR_ITER == 0:
                lr_step = global_step / args.ADJUST_LR_ITER
                new_lr = utils.adjust_learning_rate(optimizer, lr_step)
                print('learning rate decay', new_lr)
    end_t = time.time()
    print('finished!')
    print('it spends {:.2f}s, ran {} epochs'.format(end_t-start_t, args.MAX_ITERATION))


if __name__ == '__main__':
    train()
