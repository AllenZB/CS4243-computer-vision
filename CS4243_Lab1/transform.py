import numpy as np
from skimage import io
import os.path as osp

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = np.asarray(io.imread(file_name))
    if len(image.shape)==3 and image.shape[2]>3:
        image = image[:, :, :3]
    # print(image.shape) #should be (x, x, 3)
    return image

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def cs4243_resize(image, new_width, new_height):
    """
    5 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    Please round down the value to its nearest interger, 
    and take care of the order of image dimension.
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    ###Your code here####
    height, width = image.shape[:2]
    x_ratio = width / new_width
    y_ratio = height / new_height
    for i in range(new_height):
        for j in range(new_width):
            x = int(np.round(j * x_ratio))
            y = int(np.round(i * y_ratio))
            if x >= width:
                x = width - 1
            if y >= height:
                y = height - 1
            new_image[i][j] = image[y][x]
    ###
    return new_image

def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function, use the
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    ###Your code here####
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    image = (0.299 * r + 0.587 * g + 0.114 * b)
    ###

    return image / 255

def cs4243_histnorm(image, grey_level=256):
    """
    5 points 
    Stretch the intensity value to [0, 255]
    :param image : ndarray
    :param grey_level
    :return res_image: hist-normed image
    Tips: use linear normalization here https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    res_image = image.copy()
    flat = res_image.flatten()
    ##your code here ###
    low = min(flat)
    large = max(flat)
    mul = (grey_level - 1) / (large - low)
    f = lambda x: (x - low) * mul
    res_image = np.array(list(map(f, flat))).reshape(res_image.shape)
    ####
    return res_image



def cs4243_histequ(image, grey_level=256):
    """
    10 points
    Apply histogram equalization to enhance the image.
    the cumulative histogram will aso be returned and used in the subsequent histogram matching function.
    :param image: numpy.ndarray(float64)
    :return: ori_hist: histogram of original image
    :return: cum_hist: cumulated hist of original image, pls normalize it with image size.
    :return: res_image: image after being applied histogram equalization.
    :return: uni_hist: histogram of the enhanced image.
    Tips: use numpy buildin funcs to ease your work on image statistics
    """
    ###your code here####
    ###
    flat = image.flatten()
    ori_hist = np.bincount(flat, minlength=grey_level)
    cum_hist = np.array([sum(ori_hist[:i + 1]) / image.size for i in range(len(ori_hist))])
    uniform_hist = [int(cum_hist[i] * (grey_level - 1)) for i in range(len(cum_hist))]
    # Set the intensity of the pixel in the raw image to its corresponding new intensity 
    height, width = image.shape
    res_image = np.zeros(image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = uniform_hist[image[i,j]]
    
    uni_hist = np.bincount(res_image.flatten(), minlength=grey_level)
    return ori_hist, cum_hist, res_image, uni_hist
 
def cs4243_histmatch(ori_image, refer_image):
    """
    10 points
    Map value according to the difference between cumulative histogram.
    Note that the cum_hists of the two images can be very different. It is possible
    that a given value cum_hist[i] != cum_hist[j] for all j in [0,255]. In this case, please
    map to the closest value instead. if there are multiple intensities meet the requirement,
    choose the smallest one.
    :param ori_image #image to be processed
    :param refer_image #image of target gray histogram 
    :return: ori_hist: histogram of original image
    :return: ref_hist: histogram of reference image
    :return: res_image: image after being applied histogram matching.
    :return: res_hist: histogram of the enhanced image.
    Tips: use cs4243_histequ to help you
    """
    ##your code here ###
    ori_hist = np.bincount(ori_image.flatten(), minlength=256)
    ori_cum_hist = np.cumsum(ori_hist) / ori_image.size
    ref_hist = np.bincount(refer_image.flatten(), minlength=256)
    ref_cum_hist = np.cumsum(ref_hist) / refer_image.size
    map_value = [np.argmax(ref_cum_hist == min(ref_cum_hist, key=lambda x: np.abs(ori_cum_hist[i] - x))) for i in range(256)]
#
#    map_value = []
#    for i in range(256):
#        ori = sum(ori_hist[:i+1]) * refer_image.size
#        cum = 0
##        print('i --> ' + str(i))
#        for j in range(256):
#            ref = sum(ref_hist[:j+1]) * ori_image.size
#            pre_ref = sum(ref_hist[:j]) * ori_image.size
#            if ref == ori:
#                map_value.append(j)
##                print('j --> ' + str(j))
#                break
#            elif ref > ori:
#                if ori - pre_ref <= ref - ori:
#                    map_value.append(j - 1)
##                    print('j --> ' + str(j - 1))
#                else:
#                    map_value.append(j)
##                    print('j --> ' + str(j))
#                break
#    print(map_value)
    ##
    # Set the intensity of the pixel in the raw image to its corresponding new intensity
    height, width = ori_image.shape
    res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = map_value[ori_image[i,j]]
    
    res_hist = np.bincount(res_image.flatten(), minlength=256)
    
    return ori_hist, ref_hist, res_image, res_hist


def cs4243_rotate180(kernel):
    """
    Rotate the matrix by 180. 
    Can utilize build-in Funcs in numpy to ease your work
    :param kernel:
    :return:
    """
    kernel = np.flip(np.flip(kernel, 0),1)
    return kernel

def cs4243_gaussian_kernel(ksize, sigma):
    """
    5 points
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize))
    ###Your code here####
    mean = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            kernel[i][j] = np.exp(((i - mean)**2 + (j - mean)**2)/(-2 * sigma**2))
    ###
    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    10 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))
    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    flipped_kernel = cs4243_rotate180(kernel)
    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    filtered_image[i][j] += flipped_kernel[m][n] * padded_image[i + m][j + n]
    ###Your code here####

    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    10 points
    Implement a fast version of filtering algorithm.
    take advantage of matrix operation in python to replace the 
    inner 2-nested for loops in filter function.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))
    
    ###Your code here####
    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    flipped_kernel = cs4243_rotate180(kernel)
    for i in range(Hi):
        for j in range(Wi):
            filtered_image[i][j] = (flipped_kernel * padded_image[i : i + Hk, j : j + Wk]).flatten().sum()
    ###

    return filtered_image
#    return cs4243_filter(image, kernel)

def cs4243_filter_faster(image, kernel):
    """
    10 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk),also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and rehshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    map_image = np.zeros((Hi * Wi, Hk * Wk))
    for i in range(Hi):
        for j in range(Wi):
            map_image[i * Wi + j] = padded_image[i : i + Hk, j : j + Wk].flatten()
    flipped_kernel = cs4243_rotate180(kernel).flatten()
#    print(flipped_kernel)
#    print(np.matmul(map_image, flipped_kernel))
    filtered_image = np.matmul(map_image, flipped_kernel).reshape((Hi, Wi))
    ###
    return filtered_image

def cs4243_downsample(image, ratio):
    """
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """
    width, height = image.shape[1], image.shape[0]
    return image[0:height:ratio, 0:width:ratio]

def cs4243_upsample(image, ratio):
    """
    upsample the image to its 2^ratio, 
    :param image: image to be upsampled
    :param kernel: use same kernel to get approximate value for additional pixels
    :param ratio: which means upsample the width to ratio*width, and height to ratio*height
    :return res_image: upsampled image
    """
    width, height = image.shape[1], image.shape[0]
    new_width, new_height = width*ratio, height*ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = image
    return res_image


def cs4243_gauss_pyramid(image, n=3):
    """
    10 points
    build a Gaussian Pyramid of level n
    :param image: original grey scaled image
    :param n: level of pyramid
    :return pyramid: list, with list[0] corresponding to blurred image at level 0 
    Tips: you may need to call cs4243_gaussian_kernel() and cs4243_filter_faster()
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []
    ## your code here####
    filtered_image = image
    for i in range(n + 1):
        pyramid.append(filtered_image)
        filtered_image = cs4243_filter_fast(filtered_image, kernel)
        filtered_image = cs4243_downsample(filtered_image, 2)
    ##
    return pyramid
    
def test(image, n=4):
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []
    ## your code here####
    filtered_image = image
    for i in range(n + 1):
        blurred = cs4243_filter_fast(filtered_image, kernel)
        h = filtered_image - blurred
        pyramid.append(h)
        filtered_image = cs4243_downsample(blurred, 2)
    ##
    return pyramid

def cs4243_lap_pyramid(gauss_pyramid):
    """
    10 points
    build a Laplacian Pyramid from the corresponding Gaussian Pyramid
    :param gauss_pyramid: list, results of cs4243_gauss_pyramid
    :return lap_pyramid: list, with list[0] corresponding to image at level n-1
    """
    #use same Gaussian kernel
    kernel = cs4243_gaussian_kernel(7, 1)
    n = len(gauss_pyramid)
    lap_pyramid = [gauss_pyramid[n-1]] # the top layer is same as Gaussian Pyramid
    ## your code here####
    for i in range(n - 1):
        diff = gauss_pyramid[i] - cs4243_filter_faster(cs4243_upsample(gauss_pyramid[i + 1], 2), 4 * kernel)
        lap_pyramid.insert(1, diff)
    ##
    return lap_pyramid
    
def cs4243_Lap_blend(A, B, mask):
    """
    10 points
    blend image with Laplacian pyramid
    :param A: image on left
    :param B: image on right
    :param mask: mask [0, 1]
    :return blended_image: same size as input image
    Tips: use cs4243_gauss_pyramid() & cs4243_lap_pyramid() to help you
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    n = 3
    blended_image = None
    ## your code here####
    lap_pyramid_a = cs4243_lap_pyramid(cs4243_gauss_pyramid(A, n))
    lap_pyramid_b = cs4243_lap_pyramid(cs4243_gauss_pyramid(B, n))
    gauss_pyramid_r = cs4243_gauss_pyramid(mask, n)
    gauss_pyramid_r.reverse()
    lap_pyramid_c = [gauss_pyramid_r[i] * lap_pyramid_a[i] + (1.0 - gauss_pyramid_r[i]) * lap_pyramid_b[i] for i in range(len(lap_pyramid_a))]
    blended_image = lap_pyramid_c[0]
    for i in range(1, len(lap_pyramid_c)):
        blended_image = cs4243_filter_faster(cs4243_upsample(blended_image, 2), kernel * 4) + lap_pyramid_c[i]
    ##
    
    return blended_image
