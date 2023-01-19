from numpy import *
from numpy.random import default_rng
from numpy.linalg import inv
import matplotlib.pyplot as plt
from imageio import imread, imsave

## ==============================================================================================
def fpp_estimate_phi_4frames(imagestack):
    image_1minus3 = imagestack[:,:,1] - imagestack[:,:,3]
    image_0minus2 = imagestack[:,:,0] - imagestack[:,:,2]

    phi_image = zeros_like(image_1minus3)
    contrast_image = zeros_like(image_1minus3)

    okay = (abs(image_0minus2) > 0.0)
    phi_image[okay] = arctan2(image_1minus3[okay], image_0minus2[okay])
    contrast_image[okay] = 0.5 * sqrt(image_1minus3[okay]**2 + image_0minus2[okay]**2)
    bias_image = 0.25 * sum(imagestack, axis=2)

    return(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_estimate_phi(imagestack, deltas):
    (Nx,Ny,num_images) = imagestack.shape
    phi_image = zeros((Nx,Ny), 'float32')
    A = zeros((3,3), 'float32')
    B = zeros((3,Ny), 'float32')

    c = cos(deltas)
    s = sin(deltas)
    cs = sum(c * s)
    cc = sum(c**2)
    ss = sum(s**2)

    A[0,0] = num_images
    A[0,1] = sum(c)
    A[0,2] = sum(s)

    A[1,0] = A[0,1]
    A[1,1] = cc
    A[1,2] = cs

    A[2,0] = A[0,2]
    A[2,1] = cs
    A[2,2] = ss

    B0 = sum(imagestack, axis=2)
    B1 = sum(imagestack * c, axis=2)
    B2 = sum(imagestack * s, axis=2)
    B = dstack([B0, B1, B2])[:,:,:,newaxis]
    X = squeeze(dot(inv(A),B))
    phi_image = arctan2(-X[2,:], X[1,:])
    bias_image = X[0,:]
    contrast_image = sqrt(X[1,:]**2 + X[2,:]**2)

    return(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_estimate_deltas(imagestack, phi_image, nrows=10):
    ## If you want to use only a small portion of the image to estimate the deltas, then make "xmax" a small number.
    ## If you want to use the entire image to estimate the deltas, then use xmax=None
    (Nx,Ny,num_images) = imagestack.shape
    Aprime = zeros((3,3), 'float32')
    Bprime = zeros((3,num_images), 'float32')
    delta_vectors = zeros((num_images,nrows))

    Aprime[0,0] = Ny

    c = cos(phi_image)
    s = sin(phi_image)
    cs = c * s
    cc = c**2
    ss = s**2

    Bprime[0,:] = sum(imagestack, axis=(0,1))
    Bprime[1,:] = sum(imagestack * c[:,:,newaxis], axis=(0,1))
    Bprime[2,:] = sum(imagestack * s[:,:,newaxis], axis=(0,1))

    ## Do the estimate separately for each row in the image, for a total of nrows. Then take the average.
    for x in range(nrows):
        cx = sum(c[x,:])
        sx = sum(s[x,:])
        cx_cx = sum(cc[x,:])
        cx_sx = sum(cs[x,:])
        sx_sx = sum(ss[x,:])

        Aprime[0,1] = cx
        Aprime[0,2] = sx

        Aprime[1,0] = cx
        Aprime[1,1] = cx_cx
        Aprime[1,2] = cx_sx

        Aprime[2,0] = sx
        Aprime[2,1] = cx_sx
        Aprime[2,2] = sx_sx

        Xprime = dot(inv(Aprime), Bprime)
        deltas_estimated = arctan2(-Xprime[2,:], Xprime[1,:])
        delta_vectors[:,x] = deltas_estimated

    deltas = mean(delta_vectors, axis=1)

    return(deltas)

## ==============================================================================================
def fpp_estimate_deltas_and_phi(imagestack, eps=1.0E-3):
    (Nx,Ny,num_images) = imagestack.shape
    deltas = default_rng(seed=0).uniform(0.0, 2.0*pi, num_images)
    #deltas_estimated = arange(num_images) * 2.0 * pi / (num_images - 1.0)
    niter = 10

    for k in range(niter):
        deltas_new = array(deltas)
        (phi_image, contrast_image, bias_image) = fpp_estimate_phi(imagestack, deltas_new)
        deltas = fpp_estimate_deltas(imagestack, phi_image)
        epsilons = array(deltas - deltas_new)
        mean_epsilon = mean(abs(epsilons))

        if (mean_epsilon < eps):
            break

    return(phi_image, contrast_image, bias_image, deltas)

## ==============================================================================================
## ==============================================================================================

img0 = float32(imread('./figures/lens_crop_000.jpg'))
img1 = float32(imread('./figures/lens_crop_090.jpg'))
img2 = float32(imread('./figures/lens_crop_180.jpg'))
img3 = float32(imread('./figures/lens_crop_270.jpg'))
imagestack = dstack([img0, img1, img2, img3])
(Nx,Ny,num_images) = imagestack.shape

#(phi_image4, contrast4_image, bias4_image) = fpp_estimate_phi_4frames(imagestack)
(phi_image, contrast_image, bias_image, deltas) = fpp_estimate_deltas_and_phi(imagestack)
deltas -= deltas[0]

print('true deltas [deg] =', array([0.0, 90.0, 180.0, 270.0]))
print('est. deltas [deg] =', deltas*180.0/pi)

plt.figure('img0')
plt.imshow(img0)

plt.figure('phi')
plt.imshow(phi_image)

plt.show()
