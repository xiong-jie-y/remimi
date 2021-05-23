import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import cv2

import numpy as np
from PIL import Image
import glob

from remimi.segmentation.u2net.data_loader import RescaleT
from remimi.segmentation.u2net.data_loader import ToTensor
from remimi.segmentation.u2net.data_loader import ToTensorLab
from remimi.segmentation.u2net.data_loader import SalObjDataset2

from remimi.segmentation.u2net.model import U2NET # full size version 173.6 MB
from remimi.segmentation.u2net.model import U2NETP # small version u2net 4.7 MB


from remimi.utils.file import get_model_file_from_gdrive

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

import dataclasses
@dataclasses.dataclass
class U2MaskModelOption:
    model_name: str = "u2net"

class U2MaskModel:
    def __init__(self, option: U2MaskModelOption):
        model_name = option.model_name
        net = None
        file_path = None
        # --------- 3. model define ---------
        if(model_name=='u2net'):
            file_path = get_model_file_from_gdrive("u2net_pretrained.pth", "https://drive.google.com/u/0/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download")
            print("...load U2NET---173.6 MB")
            net = U2NET(3,1)
        elif(model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            net = U2NETP(3,1)
        elif(model_name=='u2net_anime'):
            file_path = get_model_file_from_gdrive("u2net_pretrained_anime.pth", "https://drive.google.com/u/0/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download")
            print("...load U2NET---173.6 MB")
            net = U2NET(3,1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(file_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(file_path, map_location='cpu'))
        net.eval()

        self.net = net

    def predict_mask(self, image_bgr):
        test_salobj_dataset = SalObjDataset2(images=[image_bgr],
                                            transform=transforms.Compose([RescaleT(320),
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)



        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):

            # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= self.net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            del d1,d2,d3,d4,d5,d6,d7

            return F.interpolate(pred.unsqueeze(0), size=(image_bgr.shape[0], image_bgr.shape[1])).squeeze(0)

    def convert_probability_mask_to_binary_mask(self, probability_mask_image):
        mask_numpy_u16 = cv2.normalize(probability_mask_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        thresh, binary_mask = cv2.threshold(
            mask_numpy_u16,
            0,
            255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        mask_u8 = np.zeros_like(binary_mask, dtype=np.uint8)
        mask_u8[binary_mask == 2 ** 16 - 1] = 255

        return binary_mask