import os
from glob import glob

import imageio as imageio


def empty_torch_queue(q):
    while True:
        try:
            o = q.get_nowait()
            del o
        except:
            break
    q.close()


def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind('/') + 1:s.rfind('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")),
                         key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(output, images)
