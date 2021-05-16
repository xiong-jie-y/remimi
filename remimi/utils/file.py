import os
from os.path import join
import urllib.request

import progressbar
import urllib.request

import youtube_dl


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


MODEL_PATH_ROOT_ = os.path.expanduser("~/.cache/remimi/models")

def get_model_file(name, url):
    filepath = os.path.join(MODEL_PATH_ROOT_, name)
    if not os.path.exists(filepath):
        os.makedirs(MODEL_PATH_ROOT_, exist_ok=True)
        urllib.request.urlretrieve(url, filepath, show_progress)

    return filepath

import gdown

def get_model_file_from_gdrive(name, url):
    filepath = os.path.join(MODEL_PATH_ROOT_, name)
    if not os.path.exists(filepath):
        os.makedirs(MODEL_PATH_ROOT_, exist_ok=True)
        gdown.download(url, filepath)

    return filepath


def ensure_video(video_url, cache_root):
    ydl_opts = {
        "outtmpl": join(cache_root, "videos", "%(id)s.%(ext)s")
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        a = ydl.extract_info(video_url)
        original_video_file = join(cache_root, "videos", f"{a['id']}.{a['ext']}")
        video_file = join(cache_root, "videos", f"{a['id']}_quilt.mp4")
        
        if not os.path.exists(original_video_file):
            original_video_file = join(cache_root, "videos", f"{a['id']}.mkv")

        import subprocess
        if not os.path.exists(video_file):
            subprocess.run(f"ffmpeg -y -i {original_video_file} -s 820x460  {video_file}", shell=True, check=True)

    return video_file
