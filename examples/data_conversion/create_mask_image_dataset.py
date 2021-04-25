from os.path import join
from remimi.sensors.file import MultipleImageStream
from remimi.sensors.edit_stream import SaveMaskAndFrameSink
from remimi.sensors.webcamera import SimpleWebcamera
import click
import cv2
import glob

from remimi.sensors import StreamFinished

@click.command()
@click.option("--video-file")
@click.option("--image-folder")
@click.option("--output-folder")
@click.option("--class-names", multiple=True)
@click.option("--margin", type=int, default=1)
def main(video_file, image_folder, output_folder, class_names, margin):
    if video_file:
        video_stream = SimpleWebcamera(video_file)
    elif image_folder:
        aa = sorted(list(glob.glob(join(image_folder, "*.jpg"))) + list(glob.glob(join(image_folder, "*.png"))))
        video_stream = MultipleImageStream(aa)
    else:
        raise RuntimeError("No input.")
    sink = SaveMaskAndFrameSink(video_stream, output_folder, class_names, margin)

    frame_no = 0
    wait_time = 1
    if image_folder is not None:
        wait_time = 4000
    while True:
        # if frame_no < 1000:
        #     frame_no += 1
        #     video_stream.get_color()
        #     continue

        try:
            sink.process(show=True)
        except StreamFinished:
            break

        key = cv2.waitKey(wait_time)
        if key == ord('a'):
            break

        frame_no += 1


if __name__ == "__main__":
    main()
