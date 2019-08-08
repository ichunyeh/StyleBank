'''
CNN analysis
1) get filter/feature-map value
2) visualize filter/feature-map
3) clustering: gmm
'''
import numpy as np
import torch
from cnn_vis.utils import save_gradient_images, convert_to_grayscale, cut_orig


def get_filter(model, target_layer, isSave=False):
    model_l1 = target_layer.split('.')[0]
    model_l2 = int(target_layer.split('.')[1])
    if model_l1 == 'style_bank':
        f = model.style_bank[model_l2][0].conv2d.weight.data.numpy()
    else:
        f = model._modules.get(model_l1)[model_l2].conv2d.weight.data.numpy()  # ConvLayer
    print('size:', f.shape)  # shape: (ch_out, ch_in, h, w)
    f_flatten = f.reshape(f.shape[0], -1)
    print('flatten:', f_flatten.shape)
    if isSave:
        np.savetxt(isSave + '.txt', f_flatten)
        print('filter value saved')
    return f_flatten

def get_featuremap(img, model, target_layer, style_id=None, isSave=False):
    x = img

    # Forward pass layer by layer
    model_l1 = target_layer.split('.')[0]
    l1_list = ['encoder_net']
    if model_l1 == 'style_bank':
        l1_list.append('style_bank')
    elif model_l1 == 'decoder_net' and style_id == None: 
        l1_list.append('decoder_net')
    elif model_l1 == 'decoder_net' and style_id != None:
        l1_list.append('style_bank')
        l1_list.append('decoder_net')
    for l1 in l1_list:
        m = model._modules.get(l1)
        if type(m) == torch.nn.modules.container.ModuleList:  # style_bank
            mm = m._modules.get(style_id)  # style_id
            for index, layer in enumerate(mm):
                now_model = l1 + '.' + style_id + '.' + str(index)
                x = layer(x)
                if now_model == target_layer:
                    break
        else:
            for index, layer in enumerate(m):
                now_model = l1 + '.' + str(index)
                x = layer(x)
                if now_model == target_layer:
                    break
    fm = x.data.numpy()[0]

    print('size:', fm.shape)
    fm_flatten = fm.reshape(-1, fm.shape[0])
    print('flatten:', fm_flatten.shape)
    if isSave:
        np.savetxt(isSave + '.txt', fm_flatten)
        print('feature-map value saved')
    return fm_flatten

def gmm(points, k, type='diag'):  # points: out_flatten
    from sklearn.mixture import GaussianMixture
    import scipy.stats

    gmm = GaussianMixture(n_components=k, covariance_type=type, max_iter=100).fit(points)
    labels = gmm.predict(points)
    print('weight:', gmm.weights_)

    centers = np.empty(shape=(gmm.n_components, points.shape[1]))
    for i in range(gmm.n_components):
        desity = scipy.stats.multivariate_normal.pdf(points, mean=gmm.means_[i], cov=gmm.covariances_[i], allow_singular=True)
        centers[i, :] = points[np.argmax(desity)]
    return centers, labels

def vis_filter(model, save_path, target_layer, target_filter):
    from vis.random_learning_filter import CNNFilterVisualization
    filter_vis = CNNFilterVisualization(model, target_layer, target_filter)
    filter_vis.visualize_layer_with_hook(save_path)
    print(str(target_filter), 'filter visualized')

def vis_featuremap(img_path, img, model, save_path, target_layer, style_id, mask=None):
    from vis.guided_backprop import GuidedBackprop
    # Guided backprop
    GBP = GuidedBackprop(model, target_layer, style_id)
    # Get gradients
    guided_grads = GBP.generate_gradients(img, mask=mask)
    # Save colored gradients
    save_gradient_images(guided_grads, save_path + '_GBP_color.jpg')
    # Save grayscale gradients
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    save_gradient_images(grayscale_guided_grads, save_path + '_GBP_gray.jpg')

    # cut original image
    out = cut_orig(img_path, guided_grads)
    out.save(save_path + '_GBP_onIMG.png', 'PNG')
    print('feature-map visualized')
