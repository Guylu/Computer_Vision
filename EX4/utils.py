import io
import os
import tempfile

import PIL
import numpy as np
import skvideo.io

import ipywidgets as widgets
from IPython.display import display, Markdown


__all__ = ['imread', 'imwrite', 'imshow', 'vread', 'vwrite', 'vshow']


def _img_to_float32(image, bounds=(0, 1)):
    vmin, vmax = bounds
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.clip((vmax - vmin) * image + vmin, vmin, vmax)
    return image


def _img_to_uint8(image, bounds=(0, 1)):
    if image.dtype != np.uint8:
        vmin, vmax = bounds
        image = (image.astype(np.float32) - vmin) / (vmax - vmin)
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image


def _check_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def _title(title, level=4):
    if title is not None:
        # display(widgets.HTML('<h{lvl}>{title}</h{lvl}>'.format(lvl=level, title=title)))
        display(Markdown("%s %s" % ("#" * level, title)))


## Image
def imread(fname, bounds=(0, 1)):
    image = PIL.Image.open(fname).convert(mode='RGB')
    image = _img_to_float32(image, bounds)
    return image


def imwrite(fname, image, bounds=(0, 1), **kwargs):
    image = _img_to_uint8(image, bounds)

    if isinstance(fname, io.BytesIO):
        kwargs['format'] = kwargs.get('format', 'png')
    else:
        fname = _check_path(fname)
    
    image = PIL.Image.fromarray(image)
    image.save(fname, **kwargs)

    
# def imshow(image, title=None, *, fname=None, bounds=(0, 1)):
#     output = widgets.Output()
#     with output:
#         if fname is None:
#             f = io.BytesIO()
#             imwrite(f, image, bounds)
#             display(widgets.Image(value=f.getvalue())
#             f.close()
#         else:
#             imwrite(fname, image, bounds)
#             display(widgets.Image.from_file(fname))

#     display(output)


def imshow(image, bounds=(0, 1), title=None, *, fname=None):
    fd = None
    if fname is None:
        fd, fname = tempfile.mkstemp(suffix='.png')
    
    imwrite(fname, image, bounds)
    
    output = widgets.Output()
    with output:
        _title(title)
        img_output = widgets.Image.from_file(fname)
        display(img_output)

    if fd is not None:
        os.close(fd)
                    
    display(output)


## Video
def vread(fname, bounds=(0,1), **kwargs):
    video = skvideo.io.vread(fname)
    video = _img_to_float32(video, bounds)
    meta = skvideo.io.ffprobe(fname).get('video', None)
    return video, meta


def _meta_to_inputdict(meta):
    inputdict = {}
    if '@r_frame_rate' in meta:
        inputdict['-r'] = meta['@r_frame_rate']
    return inputdict


def vwrite(fname, video, bounds=(0, 1), *, meta=None, **kwargs):
    video = _img_to_uint8(video, bounds)
    
    inputdict = kwargs.pop('inputdict', None)
    if meta is not None and inputdict is None:
        inputdict = _meta_to_inputdict(meta)
    
    if not isinstance(fname, io.BytesIO):
        fname = _check_path(fname)
    
    skvideo.io.vwrite(fname, video, inputdict=inputdict)


def vshow(video, bounds=(0, 1), title=None, *, fname=None, autoplay=False, loop=False, **kwargs):
    fd = None
    if fname is None:
        fd, fname = tempfile.mkstemp(suffix='.mp4')
    
    vwrite(fname, video, bounds, **kwargs)
    
    output = widgets.Output()
    with output:
        _title(title)
        vid_output = widgets.Video.from_file(fname)
        vid_output.autoplay = autoplay
        vid_output.loop = loop
        display(vid_output)

    if fd is not None:
        os.close(fd)
                    
    display(output)
