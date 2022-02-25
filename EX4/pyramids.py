import numpy as np
from scipy.ndimage import convolve


__all__ = ['pyramid_kernel', 'pyramid_reduce', 'pyramid_expand',
           'gaussian_reduce', 'laplacian_reduce', 'laplacian_expand',
           'DEFAULT_DEPTH', 'DEFAULT_KERNEL', 'conv2d']


def conv2d(image, kernel, mode='mirror'):
    """Convolves image with 2D kernel.
    
    Args:
        image (np.ndarray): base image.
        kernel (np.ndarray): kernel to convolve with `image`.
    
    Returns:
        np.ndarray: `image` convolved with `kernel`.
    """
    if image.ndim == 3 and kernel.ndim == 2:
        kernel = np.expand_dims(kernel, -1)
    
    return convolve(image, kernel, mode=mode)


def pyramid_kernel(a):
    """Returns the 5-by-5 generating kernel, given parameter `a`.
    
    Args:
        a (float): the kernel parameter.
    
    Returns:
        np.ndarray: 5-by-5 generating kernel.
    """
    w = np.array([1/4 - a/2, 1/4, a, 1/4, 1/4 - a/2], dtype=np.float32)
    return np.outer(w, w)

DEFAULT_DEPTH = 5
KERNEL_PARAM = 0.375
DEFAULT_KERNEL = pyramid_kernel(a=KERNEL_PARAM)


def pyramid_reduce(image, kernel=DEFAULT_KERNEL):
    """Reduces an image given a kernel.
    
    See:
        "The Laplacian Pyramid as a Compact Image Code" by Burt & Adelson (1983)
    
    Args:
        image (np.ndarray): image to reduce, of size (2*h, 2*w).
        kernel (np.ndarray): kernel to use for the reduction (usually 5x5).
    
    Returns:
        np.ndarray: reduced image, of size (h, w).
    """
    new_image = conv2d(image, kernel)
    return new_image[::2, ::2, ...].astype(np.float32)


def pyramid_expand(image, kernel=DEFAULT_KERNEL, out_size=None):
    """Expands an image given a kernel.
        
    Args:
        image (np.ndarray): image to expand, of size (h, w).
        kernel (np.ndarray): kernel to use for the expansion (usually 5x5).
        out_size (Tuple[int, int], optional): the expected shape of the returned image.
        
    Returns:
        np.ndarray: expanded image, of size (2*h, 2*w).
    """
    h, w, *c = image.shape
    outh, outw = (2 * h, 2 * w) if out_size is None else out_size
    assert abs(outh - 2 * h) <= 1 and abs(outw - 2 * w) <= 1, "`out_size` should be close to `(2*h, 2*w)`."
    new_image = np.zeros((outh, outw, *c), dtype=np.float32)    
    new_image[::2, ::2, ...] = image
    return conv2d(new_image, 4 * kernel)


def gaussian_reduce(image, kernel=DEFAULT_KERNEL, depth=DEFAULT_DEPTH):
    """Generates a Gaussian Pyramid from `image`, using `kernel` for reduction.
    
    The pyramid includes the original image (level 0), followed by `depth` reductions.
    
    Args:
        image (np.ndarray): image to reduce.
        kernel (np.ndarray): kernel to use for the reduction (usually 5x5).
        depth (int): how many reductions to apply.
    
    Returns:
        List[np.ndarray]: List of `depth + 1` pyramid levels, where the first level is the original image.
    """
    pyramid = [None] * (depth + 1)
    pyramid[0] = image.copy()
    
    for i in range(depth):
        pyramid[i + 1] = pyramid_reduce(pyramid[i], kernel)
            
    return pyramid


def laplacian_reduce(image, kernel=DEFAULT_KERNEL, depth=DEFAULT_DEPTH):
    """Generates a Laplacian Pyramid from `image`, using `kernel` for reduction.
    
    Args:
        image (np.ndarray): image to reduce.
        kernel (np.ndarray): kernel to use for the reduction (usually 5x5).
        depth (int): how many reductions to apply.
    
    Returns:
        List[np.ndarray]: List of `depth + 1` pyramid levels.
    """
    pyramid = gaussian_reduce(image, kernel, depth)
    for i in range(depth):
        pyramid[i] -= pyramid_expand(pyramid[i + 1], kernel, out_size=pyramid[i].shape[:2])

    return pyramid


def laplacian_expand(pyramid, kernel=DEFAULT_KERNEL):
    """Reconstructs an image from a Laplacian Pyramid.
    
    Args:
        pyramid (List[np.ndarray]): list of `depth + 1` pyramid levels.
        kernel (np.ndarray): kernel that was used to create `pyramid`.
    
    Returns:
        np.ndarray: reconstructed image.
    """
    expanded = pyramid[-1].copy()
    for curr_level in pyramid[-2::-1]:
        expanded = pyramid_expand(expanded, kernel, out_size=curr_level.shape[:2]) + curr_level
    
    return np.clip(expanded, 0, 1)