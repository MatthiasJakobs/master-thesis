import numpy as np

from skimage.transform import rotate

from deephar.utils import transform, transform_2d_point, translate

def center_crop(image, center, size, trans_matrix):
    half_width = int(size[0] / 2)
    half_height = int(size[1] / 2)

    new_image = np.zeros((half_height * 2, half_width * 2, 3)) # due to rounding errors

    image_width = image.shape[1]
    image_height = image.shape[0]

    start_x = int(center[0] - half_width)
    end_x = int(center[0] + half_width)
    
    start_y = int(center[1] - half_height)
    end_y = int(center[1] + half_height)

    trans_matrix = translate(trans_matrix, -start_x, -start_y)

    if start_x < 0:
        x_offset = abs(start_x)
    else:
        x_offset = 0

    if start_y < 0:
        y_offset = abs(start_y)
    else:
        y_offset = 0

    if (end_x + x_offset) > image_width:
        x_after_offset = (end_x + x_offset) - image_width
    else:
        x_after_offset = 0

    if (end_y + y_offset) > image_height:
        y_after_offset = (end_y + y_offset) - image_height
    else:
        y_after_offset = 0

    padded_image = np.zeros((image_height + y_offset + y_after_offset, image_width + x_offset + x_after_offset, 3))
    padded_image[y_offset : y_offset + image_height, x_offset : x_offset + image_width, :] = image[:,:,:]

    start_slice_x = 0 if x_offset > 0 else start_x
    start_slice_y = 0 if y_offset > 0 else start_y

    end_slice_x = half_width * 2 + start_slice_x
    end_slice_y = half_width * 2 + start_slice_y

    new_image[:,:,:] = padded_image[start_slice_y : end_slice_y, start_slice_x: end_slice_x, :]

    return trans_matrix, new_image

def rotate_and_crop(image, angle, center, window_size):
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    #image = rotate(image, angle, resize=True, preserve_range=True)
    image = rotate(image, angle, resize=True)

    # find new center points
    afmat = np.eye(3)

    # translate so that top left is origin (i thnk?)
    afmat = translate(afmat, -center[0], -center[1])

    # build rotation matrix
    rot_mat = np.eye(3)
    angle *= np.pi / 180
    a = np.cos(angle)
    b = np.sin(angle)
    rot_mat[0,0] = a
    rot_mat[0,1] = b
    rot_mat[1,1] = a
    rot_mat[1,0] = -b

    afmat = np.dot(rot_mat, afmat)

    # get back to original
    afmat = translate(afmat, center[0], center[1])

    # get rotated corners
    corners = np.array([
        [0, 0],
        [image_width, 0],
        [0, image_height],
        [image_width, image_height]
        ]).transpose()

    rotated_corners = transform(afmat, corners)

    # dont know what that does yet, but they do it
    afmat = translate(afmat, -min(rotated_corners[0,:]), -min(rotated_corners[1,:]))

    # get rotated_center
    rotated_center = transform_2d_point(afmat, np.array([center[0], center[1]]))

    return center_crop(image, rotated_center, window_size, afmat)

def normalize_channels(input_image, power_factors=None):
    # power factors = vector of factors for each channel, i.e. (0.01, 0.001, 0.1)
    return input_image.copy() # until I figure out why this is distorted
    image = input_image.copy()    
    if power_factors is not None:
        assert len(power_factors) == 3
        for c in range(3):
            image[:,:,c] = np.power(image[:,:,c], power_factors[c])

    # equivalent to 1/127 * image - 1 from project group
    return 2.0 * (image - 0.5)
