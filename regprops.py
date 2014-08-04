import numpy as np
from scipy import ndimage as nd
from skimage import filter as imfilter, measure, io
from line_profiler import LineProfiler


# threshold and labeling number of objects, statistics about object size and
# shape
def intensity_object_features(im, sample_size=None):
    """Segment objects based on intensity threshold and compute properties.

    Parameters
    ----------
    im : 2D np.ndarray of float or uint8.
        The input image.
    adaptive_t_radius : int, optional
        The radius to calculate background with adaptive threshold.
    sample_size : int, optional
        Sample this many objects randomly, rather than measuring all
        objects.

    Returns
    -------
    f : 1D np.ndarray of float
        The feature vector.
    names : list of string
        The list of feature names.
    """
    tim1 = im > imfilter.threshold_otsu(im)
    f, names = object_features(tim1, im, sample_size=sample_size)
    return f, names


def object_features(bin_im, im, erode=2, sample_size=None):
    """Compute features about objects in a binary image.

    Parameters
    ----------
    bin_im : 2D np.ndarray of bool
        The image of objects.
    im : 2D np.ndarray of float or uint8
        The actual image.
    erode : int, optional
        Radius of erosion of objects.
    sample_size : int, optional
        Sample this many objects randomly, rather than measuring all
        objects.

    Returns
    -------
    fs : 1D np.ndarray of float
        The feature vector.
    names : list of string
        The names of each feature.
    """
    lab_im, n_objs = nd.label(bin_im)
    if sample_size is None:
        sample_size = n_objs
        sample_indices = np.arange(n_objs)
    else:
        sample_indices = np.random.randint(0, n_objs, size=sample_size)
    objects = measure.regionprops(lab_im, intensity_image=im)
    prop_names = measure._regionprops.PROPS.values()
    properties = []
    for i, j in enumerate(sample_indices):
        properties.append([])
        properties[i].append(objects[j].area)
        properties[i].append(objects[j].bbox)
        properties[i].append(objects[j].moments_central)
        properties[i].append(objects[j].centroid)
        properties[i].append(objects[j].convex_area)
        properties[i].append(objects[j].convex_image)
        properties[i].append(objects[j].coords)
        properties[i].append(objects[j].eccentricity)
        properties[i].append(objects[j].equivalent_diameter)
        properties[i].append(objects[j].euler_number)
        properties[i].append(objects[j].extent)
        properties[i].append(objects[j].filled_area)
        properties[i].append(objects[j].filled_image)
        properties[i].append(objects[j].moments_hu)
        properties[i].append(objects[j].image)
        properties[i].append(objects[j].label)
        properties[i].append(objects[j].major_axis_length)
        properties[i].append(objects[j].max_intensity)
        properties[i].append(objects[j].mean_intensity)
        properties[i].append(objects[j].min_intensity)
        properties[i].append(objects[j].minor_axis_length)
        properties[i].append(objects[j].moments)
        properties[i].append(objects[j].moments_normalized)
        properties[i].append(objects[j].orientation)
        properties[i].append(objects[j].perimeter)
        properties[i].append(objects[j].solidity)
        properties[i].append(objects[j].weighted_moments_central)
        properties[i].append(objects[j].weighted_centroid)
        properties[i].append(objects[j].weighted_moments_hu)
        properties[i].append(objects[j].weighted_moments)
        properties[i].append(objects[j].weighted_moments_normalized)
    return properties, prop_names


if __name__ == '__main__':
    image = io.imread('test-image.png')
    green = image[..., 1].copy()
    lp = LineProfiler()
    lp.add_function(object_features)
    lp.run('intensity_object_features(green, 100)')
    lp.print_stats()
    lp.dump_stats('profile.lprof')
    print(__file__)
