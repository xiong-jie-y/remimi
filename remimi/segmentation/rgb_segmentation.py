from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2
import numpy as np

class SemanticSegmenter:
    def __init__(self):
        # config_file = '/home/yusuke/gitrepos/mmsegmentation/configs/ocrnet/ocrnet_hr18s_512x512_80k_ade20k.py'
        # checkpoint_file = '/home/yusuke/gitrepos/mmsegmentation/checkpoints/ocrnet_hr18s_512x512_80k_ade20k_20200615_055600-e80b62af.pth'


        config_file = '/home/yusuke/gitrepos/mmsegmentation/configs/dnlnet/dnl_r101-d8_512x512_80k_ade20k.py'
        checkpoint_file = '/home/yusuke/gitrepos/mmsegmentation/checkpoints/dnl_r101-d8_512x512_80k_ade20k_20200826_183354-d820d6ea.pth'
        ## config_file = '/home/yusuke/gitrepos/mmsegmentation/configs/sem_fpn/fpn_r50_512x512_160k_ade20k.py'
        # checkpoint_file = '/home/yusuke/gitrepos/mmsegmentation/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9.pth'

        # build the model from a config file and a checkpoint file
        model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

        model.PALETTE[12] = [0,0,0]
        model.PALETTE[33] = [0,0,255]
        model.PALETTE[15] = [0,255,0]
        model.PALETTE[7] = [255, 255, 255]

        self.model = model

    
    def convert_to_semantic_image(self, color_image_bgr):
        frame = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        result = inference_segmentor(self.model, frame)
        ret_img = np.zeros(frame.shape)

        # Get Pallete.
        palette = self.model.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.model.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        seg = result[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        return color_seg
