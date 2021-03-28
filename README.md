# remimi
This repository contains library and examples to do R&D of ML efficiently.

## Installation
Please install pytorch, opencv (>4).

```bash
git clone https://github.com/xiong-jie-y/remimi.git
pip install . -e
```

## Examples
### Realtime Monocular Depth Estimation Examples with DPT
![](./images/monodepth_static.gif)

```bash
# (1) Install dependencies https://github.com/xiong-jie-y/DPT#setup

# (2) Download model from https://github.com/xiong-jie-y/DPT#setup

# (3) Run web camera depth stream.
# You can use realsense intrinsic parameters if you uses realsense and add --use-realsense flag.
python examples/monodepth/run_dpt_monodepth.py -m ~/gitrepos/DPT/weights/dpt_hybrid-midas-501f0c75.pt
```