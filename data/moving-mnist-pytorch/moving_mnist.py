# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from PIL import Image
import imageio
import sys
import os
import math
import numpy as np
import random
import scipy.misc

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
# usage: python3 moving_mnist.py --dest ~/dataset/moving-mnist/moving-mnist-val-new1.npz --filetype npz --seq_len 20 --n_seq 3000 --nums_per_image 2
###########################################################################################

# image_size = 64
# digit_size = 28
step_length = 0.15

# helper functions
def arr_from_img(im,shift=0):
    w,h=im.size
    arr=im.getdata()
    c = np.int(np.product(arr.size) / (w*h))
    return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255. - shift

def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
    if ch == 1:
        ret=ret.reshape(h,w)
    return ret


def load_dataset():
  # Load MNIST dataset for generating training data.
  import gzip
  # path = os.path.join(root, 'train-images-idx3-ubyte.gz')
  filename = 'train-images-idx3-ubyte.gz'
  with gzip.open(filename, 'rb') as f:
    mnist = np.frombuffer(f.read(), np.uint8, offset=16)
    mnist = mnist.reshape(-1, 28, 28)
  return mnist

def get_random_trajectory(seq_length=30, image_size=64, digit_size=28):
    
    ''' Generate a random sequence of a MNIST digit '''
    canvas_size = image_size - digit_size
    x = random.random()
    y = random.random()
    theta = random.random() * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros(seq_length)
    start_x = np.zeros(seq_length)
    for i in range(seq_length):
      # Take a step along velocity.
      y += v_y * step_length
      x += v_x * step_length

      # Bounce off edges.
      if x <= 0:
        x = 0
        v_x = -v_x
      if x >= 1.0:
        x = 1.0
        v_x = -v_x
      if y <= 0:
        y = 0
        v_y = -v_y
      if y >= 1.0:
        y = 1.0
        v_y = -v_y
      start_y[i] = y
      start_x[i] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

def generate_moving_mnist(num_digits=2, n_frames_total=30, n_seq=10000, image_size=64, digit_size=28):
    '''
    Get random trajectories for the digits and generate a video.
    '''
    mnist = load_dataset()
    data = np.zeros((n_seq, n_frames_total, image_size, image_size), dtype=np.float32)
    for seq_idx in range(n_seq):
        canvas = np.zeros((n_frames_total, image_size, image_size), dtype=np.float32)
        for n in range(num_digits):
          # Trajectory
          start_y, start_x = get_random_trajectory(n_frames_total, image_size, digit_size)
          ind = random.randint(0, mnist.shape[0] - 1)
          digit_image = mnist[ind]
          if digit_image.shape[0] != digit_size:
            digit_image = np.resize(digit_image, (digit_size, digit_size))
          print("digit_image shape", digit_image.shape, digit_image.max(), digit_image.min())
          for frame_idx in range(n_frames_total):
            top    = start_y[frame_idx]
            left   = start_x[frame_idx]
            bottom = top + digit_size
            right  = left + digit_size
            # Draw digit
            canvas[frame_idx, top:bottom, left:right] = np.maximum(canvas[frame_idx, top:bottom, left:right], digit_image)
        if seq_idx == 0:
            for frame_idx in range(n_frames_total):
                imageio.imwrite('tmp/out_%d.jpg'%(frame_idx), canvas[frame_idx])
                #scipy.misc.imsave('tmp/out_%d.jpg'%(frame_idx), canvas[frame_idx])
        data[seq_idx] = canvas
        print(seq_idx, data[seq_idx].max())
        # for frame_idx in range(n_frames_total):
        #     print(seq_idx, frame_idx, data[seq_idx, frame_idx].shape, data[seq_idx, frame_idx].max())
        #     if frame_idx == 0:
        #         save_img = data[seq_idx, frame_idx]
        #     else:
        #         save_img = np.concatenate([save_img, data[seq_idx, frame_idx]], axis=1)
        # print(save_img.shape, save_img.max())
        # img = Image.fromarray(save_img[:,:].astype(np.int8), 'L')
        # img.save('temp/%d.png'%(seq_idx))

    data = data[..., np.newaxis]#.astype(np.int8)
    print(data.shape)
    return data

def main(dest, filetype='npz', seq_len=30, n_seq=100, nums_per_image=2, 
                        image_size=64, digit_size=28):
    dat = generate_moving_mnist(num_digits=nums_per_image, n_frames_total=seq_len, 
                                n_seq=n_seq, image_size=image_size, digit_size=digit_size)
    if filetype == 'hdf5':
        n = n_seq * seq_len
        import h5py
        from fuel.datasets.hdf5 import H5PYDataset
        def save_hd5py(dataset, destfile, indices_dict):
            f = h5py.File(destfile, mode='w')
            images = f.create_dataset('images', dataset.shape, dtype='uint8')
            images[...] = dataset
            split_dict = dict((k, {'images':v}) for k,v in indices_dict.iteritems())
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()
        indices_dict = {'train': (0, n*9/10), 'test': (n*9/10, n)}
        save_hd5py(dat, dest, indices_dict)
    elif filetype == 'npz':
        np.savez(dest, data=dat)
        print(dest)
    elif filetype == 'jpg':
        for i in range(dat.shape[0]):
            Image.fromarray(get_picture_array(dat, i, shift=0)).save(os.path.join(dest, '{}.jpg'.format(i)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--image_size', type=int, dest='image_size')
    parser.add_argument('--dest', type=str, dest='dest')
    parser.add_argument('--filetype', type=str, dest='filetype')
    # parser.add_argument('--frame_size', type=int, dest='frame_size')
    parser.add_argument('--seq_len', type=int, dest='seq_len') # length of each sequence
    parser.add_argument('--n_seq', type=int, dest='n_seq') # number of sequences to generate
    parser.add_argument('--digit_size', type=int, dest='digit_size') # size of mnist digit within frame
    parser.add_argument('--nums_per_image', type=int, dest='nums_per_image') # number of digits in each frame
    # parser.add_argument('--step_length', type=float, default=0.1, dest='step_length')
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
