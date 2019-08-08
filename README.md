# StyleBank
Implementation of the paper - [StyleBank: An Explicit Representation for Neural Image Style Transfer](https://arxiv.org/abs/1703.09210) <br>

Code Reference : [Stylebank](https://github.com/jxcodetw/Stylebank), [CNN Visualization](https://github.com/utkuozbulak/pytorch-cnn-visualizations)


## Requirements
Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages: <br>
* numpy
* torch >= 1.0.0
* torchvision

## Introducion
It is one method of the **Style Transfer** based on Convolutional Neural Network, also named as **Neural Style Transfer**. <br>
Compared to the first Neural Style Transfer ([Gatys, 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)), it's more conventient. <br>
1. TransferNet <br>
(no need to train every time if have trained this style before)
2. Incremental Learning
3. Style and Content is seperable

## Description
### Architecture
<img src="img/architecture.png" width="70%">
<br>

- **Input** <br>
using random 1000 pics from MS-COCO datasets as content image dataset, and 16 kinds of art painting from Wiki-Art as style image dataset.

- **Network and Training** <br>
having two part. Each part has each corresponding training branch. <br>
  1. Auto-encoder : encoder and decoder.<br>
  <img src="img/autoencoder_train.png" width="80%">


  2. StyleBank Layer: style filter.<br>
  <img src="img/stylebanklayer_train.png" width="80%">

- **Training Strategy** <br>
Employ a (T+1)-step alternative training strategy in order to balance the two branches (auto-encoder and stylizing). During training, for every T+1 iterations, we first train T iterations on the stylizing branch , then train one iteration for auto-encoder branch.


- **Parameters** <br>
Image size : 512 x 512 <br>
Content weight: 1, Style weight: 1000000 <br>
T: 2
Learning rate: 0.001, Learning rate decay: decayed by 0.8 at every 30k iterations.<br>
Iteration : 300,000



## Training
Prepare the content image dataset ([MS-COCO](http://cocodataset.org/#download)), and your style image dataset. <br>
`python train.py` <br>
It takes about 2 days to train on GeForce GTX 2080 Ti. <br>

Pre-trained weights are provied.  You can do **Incremental Learning**. Just load the pre-trained  encoder and decoder, then train the stylebank which style image you want.<br>
`python train_add.py`


## GUI
`python gui.py` <br>

### 1. Use whole style to do style transfer 
**Step1. Select Content Image** <br>
**Step2. Select Style Image** <br>
**Step3. Click 'Stylize...' Button** <br>
<text color='gray'>[ Result ]</text><br>
<img src="img/gui_result_1.png" width="80%">

### 2. Choose some style elements to do style transfer 
**Step1. Select Content Image** <br>
**Step2. Select Style Image** <br>
**Step3. Click 'gmm(10)' Button**<br>
Wait a monent, it will show 1~10 style elements.<br>Choose some style elements you like.<br>
**Step4. Click 'Stylize...' Button** <br>
<text color='gray'>[ Result ]</text><br>
<img src="img/gui_result_2.png" width="80%">
<br>

## Examples
Style | Content | Stylized
:--:|:--:|:--:
<img src="data/style/style/03.jpg" width="256"> | <img src="data/test/deer.jpg" width="512"> | <img src="data/output/deer_3.jpg" width="512">
<img src="data/style/style/00.jpg" width="256"> | <img src="data/test/mountain.jpg" width="512"> | <img src="data/output/mountain_0.jpg" width="512">
<img src="data/style/style/06.jpg" width="256"> | <img src="data/test/bridge.jpg" width="512"> | <img src="data/output/bridge_6.jpg" width="512">
<img src="data/style/style/10.jpg" width="256"> | <img src="data/test/bridge.jpg" width="512"> | <img src="data/output/bridge_10.jpg" width="512">


## Future
- [ ] content control
- [ ] mobile app
- [ ] show visualizing results
- [ ] explain test.py