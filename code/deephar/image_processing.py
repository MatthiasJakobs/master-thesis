import numpy as np

from skimage.transform import rotate

from deephar.utils import transform, transform_2d_point, translate

def center_crop(image, center, size, trans_matrix):
    print(size)
    half_width = size[0] / 2
    half_height = size[1] / 2

    image_width = image.shape[1]
    image_height = image.shape[0]

    start_x = int(max(0, center[0] - half_width))
    end_x = int(min(image_width, center[0] + half_width))
    
    start_y = int(max(0, center[1] - half_height))
    end_y = int(min(image_height, center[1] + half_height))

    # Concern: Maybe clipping between (0, width) etc. only for when actually slicing and not before matrix translate? could f things up
    trans_matrix = translate(trans_matrix, -start_x, -start_y)

    return trans_matrix, image[start_y : end_y, start_x : end_x, :]

def rotate_and_crop(image, angle, center, window_size):
    image_width = image.shape[1]
    image_height = image.shape[0]
    
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
    image = input_image.copy()
    print(image.shape)
    if power_factors is not None:
        print(power_factors.shape)      
    image /= 255
    if power_factors is not None:
        assert len(power_factors) == 3
        for c in range(3):
            image[:,:,c] = np.power(image[:,:,c], power_factors[c])

    # equivalent to 1/127 * image - 1 from project group
    return 2 * (image - 0.5)
