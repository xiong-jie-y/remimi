#!/usr/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "Please put video id to argument." 1>&2
    exit 1
fi

export VIDEO_ID=$1
export MASK_FOLDER=${VIDEO_ID}_mask
export CACHE_ROOT=$2

python examples/data_conversion/create_mask_image_dataset.py \
    --video-url $VIDEO_ID \
    --output-folder $MASK_FOLDER \
    --class-names person \
    --margin 0 --cache-root $CACHE_ROOT

export FRAME_RATE=`ffmpeg -i $CACHE_ROOT/videos/${VIDEO_ID}_quilt.mp4 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"`

ffmpeg -framerate $FRAME_RATE -i $MASK_FOLDER/masks/%05d.png -vcodec libx264 -s 820x460 -pix_fmt yuv420p -crf 18 $CACHE_ROOT/$MASK_FOLDER.mp4

python examples/monodepth/create_point_cloud_video.py \
    --video-url $VIDEO_ID \
    --mask-dir $CACHE_ROOT/$MASK_FOLDER.mp4 \
    --cache-root $CACHE_ROOT \
    --inpaint --create-looking-glass