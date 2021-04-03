from remimi.sensors.edit_stream import SaveMaskAndFrameSink
from remimi.sensors.webcamera import SimpleWebcamera
import click
import cv2

from remimi.sensors import StreamFinished

@click.command()
@click.option("--input-file")
@click.option("--output-folder")
@click.option("--class-names", multiple=True)
@click.option("--margin", type=int, default=1)
def main(input_file, output_folder, class_names, margin):
    video_stream = SimpleWebcamera(input_file)
    sink = SaveMaskAndFrameSink(video_stream, output_folder, class_names, margin)

    while True:
        try:
            sink.process(show=True)
        except StreamFinished:
            break

        key = cv2.waitKey(1)
        if key == ord('a'):
            break


if __name__ == "__main__":
    main()
