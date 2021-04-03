import click
import cv2

from moviepy.editor import ImageSequenceClip
from remimi.sensors import StreamFinished
from remimi.sensors.edit_stream import HumanEliminatedStream
from remimi.sensors.webcamera import SimpleWebcamera


@click.command()
@click.option("--input-file")
@click.option("--output-file")
def main(input_file, output_file):
    sensor = SimpleWebcamera(input_file)
    human_edited_stream = HumanEliminatedStream(sensor)

    frames = []

    while True:
        try:
            color = human_edited_stream.get_color()
        except StreamFinished:
            break

        cv2.imshow("Human Removed", color)

        # RGB becase moviepy requires rgb frames.
        frames.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)
        if key == ord('a'):
            break

    ImageSequenceClip(sequence=frames, fps=30).write_videofile(output_file)

if __name__ == "__main__":
    main()
