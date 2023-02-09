#!/usr/bin/env python

from numpy import *
import matplotlib.pyplot as plt
from imageio import imread, imsave
from glob import glob

if __name__ == '__main__':
    files = sort(glob('./figures/lens_crop*.jpg'))
    #files = sort(glob('./figures/fringe_phi*.png'))

    ## Make a list of the images to use, and then convert the list into a Numpy array 3D image stack.
    imagestack = []
    for file in files:
        imagestack.append(float32(imread(file)))
    imagestack = dstack(imagestack)

    (Nx,Ny,num_images) = imagestack.shape
    print('imagestack shape =', imagestack.shape)

    deltas = array([0.0, pi/2.0, pi, 3.0*pi/2.0])
    #(phi4_image, contrast4_image, bias4_image) = fpp_4_uniform_frames(imagestack)
    #(phi_imageN, contrast_imageN, bias_imageN) = fpp_N_uniform_frames(imagestack)
    #(phi_image) = fpp_4_nonuniform_frames(imagestack, deltas)
    (phi_image) = fpp_N_nonuniform_frames(imagestack, deltas)

    #(phi_image, contrast_image, bias_image, deltas) = fpp_estimate_deltas_and_phi(imagestack)
    deltas -= deltas[0]

    print(f"est. deltas [deg] = {array2string(deltas*180.0/pi, formatter={'float': lambda x:f'{x:.2f}'}, separator=', ')}")

    plt.figure('img0')
    plt.imshow(imagestack[:,:,0])

    plt.figure('phi')
    plt.imshow(phi_image)

    ## In the unwrapped phase image, make sure that the smallest phase is zero. If the phase is increasing, then we subtract the
    ## smallest value. If decreasing, then we add the smallest value.
    unwrapped_phi = unwrap(phi_image)
    avg_gradient = mean(gradient(unwrapped_phi[:,Ny//2]))

    if (avg_gradient > 0.0):
        unwrapped_phi += amax(unwrapped_phi)
    else:
        unwrapped_phi -= amax(unwrapped_phi)

    plt.figure('phi_unwrapped')
    plt.imshow(unwrapped_phi)

    plt.show()
