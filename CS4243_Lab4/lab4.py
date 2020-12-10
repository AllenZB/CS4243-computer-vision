""" CS4243 Lab 4: Tracking
Please read accompanying Jupyter notebook (lab4.ipynb) and PDF (lab4.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time


# Part 1 

def meanShift(dst, track_window, max_iter=100,stop_thresh=1):
    """Use mean shift algorithm to find an object on a back projection image.

    Args:
        dst (np.ndarray)            : Back projection of the object histogram of shape (H, W).
        track_window (tuple)        : Initial search window. (x,y,w,h)
        max_iter (int)              : Max iteration for mean shift.
        stop_thresh(float)          : Threshold for convergence.
    
    Returns:
        track_window (tuple)        : Final tracking result. (x,y,w,h)


    """

    completed_iterations = 0
    
    """ YOUR CODE STARTS HERE """
    H, W = dst.shape
    x, y, w, h = track_window
    x_max = W - w
    y_max = H - h
    # print('--------')
    # print(x, y, w, h)
    left = max_iter
    # left = 1
    while left > 0:
        left -= 1
        prob = dst[y:y+h+1, x:x+w+1]
        prob_x = np.average(prob, axis=0)
        prob_y = np.average(prob, axis=1)
        if sum(prob_x) == 0:
            mean_x = max(0, x - w)
        else:
            mean_x = sum(prob_x * range(x, min(x + w + 1, W))) / sum(prob_x)
        if sum(prob_y) == 0:
            mean_y = max(0, y - h)
        else:
            mean_y = sum(prob_y * range(y, min(y + h + 1, H))) / sum(prob_y)
        mean_x = max(0, int(mean_x - w * 0.5))
        mean_y = max(0, int(mean_y - h * 0.5))
        # print(x, mean_x, y, mean_y)
        if abs(mean_x - x) + abs(mean_y - y) < stop_thresh:
            break
        x = int(mean_x)
        y = int(mean_y)
    track_window = (x, y, w, h) 
    """ YOUR CODE ENDS HERE """
    
    return track_window
    
    
    
        

def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes.

    Args:
        bbox1 (tuple)               : First bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
        bbox2 (tuple)               : Second bounding box position (x, y, w, h) where (x, y) is the top left corner of
                                      the bounding box, and (w, h) are width and height of the box.
    Returns:
        score (float)               : computed IoU score.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    """ YOUR CODE STARTS HERE """

    total_area = w1 * h1 + w2 * h2
    if x1 < x2:
        common_x = x1 + w1 - x2
    else:
        common_x = x2 + w2 - x1
    if y1 < y2:
        common_y = y1 + h1 - y2
    else:
        common_y = y2 + h2 - y1
    common_area = common_x * common_y
    common_area = common_area if common_area > 0 else 0
    score = common_area / (total_area - common_area) 
    """ YOUR CODE ENDS HERE """
    print(score, common_area, total_area)
    return score


# Part 2:
def lucas_kanade(img1, img2, keypoints, window_size=9):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix.
    """
    assert window_size % 2 == 1, "window_size must be an odd number"
    
    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be loacated between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        """ YOUR CODE STARTS HERE """
        H,W=img1.shape
        A= np.zeros((window_size*window_size,2))
        B= np.zeros((2,1))
        C= np.zeros((window_size*window_size,))
        m=0
        n=0
        for i in range(y-w,y+w+1):
            k=i
            if(i<0):
                k=0
            if(i>=H):
                k=H-1
            n=0
            for j in range(x-w,x+w+1):
                t=j
                if (i < 0):
                    t=0
                if (i >= W):
                    t=W-1
                A[m*window_size+n][0]=Iy[k][t]
                A[m*window_size+n][1]=Ix[k][t]
                C[m*window_size+n]=-It[k][t]
                n+=1
            m+=1

        # A1 = Ix[y - w: y + w + 1, x - w: x + w + 1]
        # A2 = Iy[y - w: y + w + 1, x - w: x + w + 1]
        # A = np.c_[A1.reshape(-1, 1), A2.reshape(-1, 1)]
        # b = -It[y - w: y + w + 1, x - w: x + w + 1].reshape(-1, 1)
        # d = np.dot(np.linalg.inv(A.T.dot(A)), A.T.dot(b))
        # flow_vectors.append(d.flatten())
        print(A.shape)
        print(C.shape)
        B=np.linalg.lstsq(A,C,rcond=None)[0]
        flow_vectors.append(B)


        """ YOUR CODE ENDS HERE """

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2
        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 (np.ndarray)         : Grayscale image patch1 of shape (patch_size, patch_size)
        patch2 (np.ndarray)         : Grayscale image patch2 of shape (patch_size, patch_size)
    Returns:
        error (float)               : Number representing mismatch between patch1 and patch2.
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0

    """ YOUR CODE STARTS HERE """

    std_p1=np.std(patch1)
    std_p2=np.std(patch2)
    if std_p1==0:
        std_p1=1
    if std_p2==0:
        std_p2=1
    std1=((patch1-patch1.mean())/std_p1)
    std2=((patch2-patch2.mean())/std_p2)
    from sklearn.metrics import mean_squared_error
    error= mean_squared_error(std2,std1)

    """ YOUR CODE ENDS HERE """

    return error



def iterative_lucas_kanade(img1, img2, keypoints,
                           window_size=9,
                           num_iters=5,
                           g=None):
    """ Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to update flow vector.
        g (np.ndarray)              : Flow vector guessed from previous pyramid level.
                                      Array of shape (N, 2).
    Returns:
        flow_vectors (np.ndarray)   : Estimated flow vectors for keypoints. flow_vectors[i] is
                                      the flow vector for keypoint[i]. Array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2
    
   
    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2) # Initialize flow vector as zero vector
        y1 = int(round(y))
        x1 = int(round(x))
        
        """ YOUR CODE STARTS HERE """
        #G=np.zeros((2,2))
        #H,W=img1.shape
        # for i in range(int(y-w),int(y+w+1)):
        #     k=i
        #     if(i<0):
        #         k=0
        #     if(i>=H):
        #         k=H-1
        #     for j in range(int(x-w),int(x+w+1)):
        #         t=j
        #         if (i < 0):
        #             t=0
        #         if (i >= W):
        #             t=W-1
        #         G[0][0]=np.dot(Ix[k][t],Ix[k][t])+G[0][0]
        #         G[0][1]=np.dot(Ix[k][t],Iy[k][t])+G[0][1]
        #         G[1][0]=np.dot(Ix[k][t],Iy[k][t])+G[1][0]
        #         G[1][1]=np.dot(Iy[k][t],Iy[k][t])+G[1][1]
        ty = Iy[y1 - w:y1 + w + 1, x1 - w:x1 + w + 1]
        tx = Ix[y1 - w:y1 + w + 1, x1 - w:x1 + w + 1]
        G = np.array([[(tx ** 2).sum(), (tx * ty).sum()], [(tx * ty).sum(), (ty ** 2).sum()]])
        for i in range(num_iters):
            vx,vy=v
            y2 = int(round(y + gy + vy))
            x2 = int(round(x + gx + vx))
            # I_k=np.zeros((2,2))   #I_k= I(x,y) − J(x + g x + v x ,y + g y + v y )
            before = img1[y1 - w:y1 + w + 1, x1 - w:x1 + w + 1]
            if (x2 - w < 0):
                after = np.ndarray((2 * w + 1, 2 * w + 1))
                for i in range(2 * w + 1):
                    for j in range(abs(x2-w)):
                        after[i][j] = img2[y1 - w + i][0]
                    for j in range(abs(x2-w), 2 * w + 1):
                        after[i][j] = img2[y1 - w + i][j - abs(x2-w)]
                # after = img2[y2 - w:y2 + w + 1, 0:x2 + w + 1]
                # for i in range(len(after)):
                #     after[i] = np.concatenate([[img2[y2 - w + i][0]] * abs(x2 - w), after[i]])
                # print(after)

            else:
                after = img2[y2 - w:y2 + w + 1, x2 - w:x2 + w + 1]
            # print(y1 - w, y1 + w + 1, x1 - w, x1 + w + 1)
            # print(y2 - w, y2 + w + 1, x2 - w, x2 + w + 1)
            I_k = before - after
            # b_k=np.zeros((2,1))
            # for i in range(int(y - w), int(y + w + 1)):
            #     k = i
            #     if (i < 0):
            #         k = 0
            #     if (i >= H):
            #         k = H - 1
            #     for j in range(int(x - w), int(x + w + 1)):
            #         t = j
            #         if (i < 0):
            #             t = 0
            #         if (i >= W):
            #             t = W - 1
            #         b_k[0][0]=b_k[0][0]+np.dot(I_k[k][t],Ix[k][t])
            #         b_k[1][0]=b_k[1][0]+np.dot(I_k[k][t],Iy[k][t])
            b_k = np.array([(I_k * tx).sum(), (I_k * ty).sum()])
            v_k=np.dot(np.linalg.inv(G),b_k)
            v=np.add(v,v_k)
        """ YOUR CODE ENDS HERE """

        vx, vy = v
        flow_vectors.append([vy, vx])
        
    return np.array(flow_vectors)
        

def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=5,
                         level=2, scale=2):

    """ Pyramidal Lucas Kanade method

    Args:
        img1 (np.ndarray)           : Grayscale image of the current frame. 
                                      Flow vectors are computed with respect to this frame.
        img2 (np.ndarray)           : Grayscale image of the next frame.
        keypoints (np.ndarray)      : Coordinates of keypoints to track of shape (N, 2).
        window_size (int)           : Window size to determine the neighborhood of each keypoint.
                                      A window is centered around the current keypoint location.
                                      You may assume that window_size is always an odd number.

        num_iters (int)             : Number of iterations to run iterative LK method
        level (int)                 : Max level in image pyramid. Original image is at level 0 of
                                      the pyramid.
        scale (float)               : Scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    """ YOUR CODE STARTS HERE """
    for L in range(level, -1, -1):
        I_L = pyramid1[L]
        J_L = pyramid2[L] 
        p_L = keypoints / scale ** L
        d = iterative_lucas_kanade(I_L, J_L, p_L, window_size=window_size, num_iters=num_iters, g=g)
        g = scale * (g + d) 

    """ YOUR CODE ENDS HERE """

    d = g + d
    return d























"""Helper functions: You should not have to touch the following functions.
"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from skimage import filters, img_as_float
from skimage.io import imread
from skimage.transform import pyramid_gaussian

def load_frames_rgb(imgs_dir):

    frames = [cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, frame)), cv2.COLOR_BGR2RGB) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_frames_as_float_gray(imgs_dir):
    frames = [img_as_float(imread(os.path.join(imgs_dir, frame), 
                                               as_gray=True)) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_bboxes(gt_path):
    bboxes = []
    with open(gt_path) as f:
        for line in f:
          
            x, y, w, h = line.split(',')
            #x, y, w, h = line.split()
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def animated_frames(frames, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_bbox(frames, bboxes, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    x, y, w, h = bboxes[0]
    bbox = ax.add_patch(Rectangle((x,y),w,h, linewidth=3,
                                  edgecolor='r', facecolor='none'))

    def animate(i):
        im.set_array(frames[i])
        bbox.set_bounds(*bboxes[i])
        return [im, bbox,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def animated_scatter(frames, trajs, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    scat = ax.scatter(trajs[0][:,1], trajs[0][:,0],
                      facecolors='none', edgecolors='r')

    def animate(i):
        im.set_array(frames[i])
        if len(trajs[i]) > 0:
            scat.set_offsets(trajs[i][:,[1,0]])
        else: # If no trajs to draw
            scat.set_offsets([]) # clear the scatter plot

        return [im, scat,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):

    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3 # Take 3x3 patches to compute error
    w = patch_size // 2 # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi)); xi = int(round(xi))
            yj = int(round(yj)); xj = int(round(xj))
            # Point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            # Compute error between patches in image I and J
            patchI = I[yi-w:yi+w+1, xi-w:xi+w+1]
            patchJ = J[yj-w:yj+w+1, xj-w:xj+w+1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs