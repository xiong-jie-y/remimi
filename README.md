# remimi
This repository contains python library and examples to do R&D of computer vision ML efficiently. Especially this library focus on stream video or camera input processing.

This library is optimized for efficient R&D and might not fit some kind of products.

Feel free to post issue and PR when you find problem or new feature.

## Installation
Everything is tested on Ubuntu 20.04.
Most of the script requires GPU.

Please install pytorch, opencv (>4).

Example installation procedure for cuda11 is

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

```bash
git clone https://github.com/xiong-jie-y/remimi.git
pip install . -e
```

## Examples
### Human Elimination
![](./images/human_elim.gif)

This script will eliminate as many human as possible from the video.
To run this script,

```
python examples/remove_people.py --input-file ${INPUT_FILE_PATH} --output-file ${OUTPUT_FILE_PATH}
```

And please wait until finishes.

### Realtime Monocular Depth Estimation Examples with DPT
![](./images/monodepth_static.gif)

This script will show point cloud from the camera stream.

* This scripts uses prime sense default camera intrinsic as default intrinsic. 
* When you use realsense you can use correct intrinsic easily, or for web cam please find intrinsic by yourself.
* The argument of the run_dpt_monodepth.py is almost same as DPT's run_monodepth.py script.

```bash
# (1) Install dependencies https://github.com/xiong-jie-y/DPT#setup

# (2) Download model from https://github.com/xiong-jie-y/DPT#setup

# (3) Run web camera depth stream.
# You can use realsense intrinsic parameters if you uses realsense and add --use-realsense flag.
python examples/monodepth/run_dpt_monodepth.py -m ~/gitrepos/DPT/weights/dpt_hybrid-midas-501f0c75.pt

# You can also show video file.
python examples/monodepth/run_dpt_monodepth.py -m ~/gitrepos/DPT/weights/dpt_hybrid-midas-501f0c75.pt --input-file $VIDEO_FILE
```

## LICENSE
This library and scripts are [MIT License](LICENSE). Some components has different license because they are inported from other repos for better integration. LICENSE files are in each directory.