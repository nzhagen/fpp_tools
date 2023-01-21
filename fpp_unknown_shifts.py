from numpy import *
from numpy.random import default_rng
from numpy.linalg import inv
import matplotlib.pyplot as plt
from imageio import imread, imsave
from glob import glob

## ==============================================================================================
def generate_and_save_fringe_patterns(filebase, Nx, Ny, phases, num_fringes, gamma=1.0):
    '''
    filebase = the base of the filenames to save to
    Nx = the height dimension in pixels
    Ny = the width dimension in pixels
    phases = a list of the phases (in radians) of the patterns to generate
    num_fringes = how many fringes should appear inside each projection frame
    gamma = the projector gamma value to use

    Example
    -------
    generate_and_save_fringe_pattern(480, 640, 4, 12)
    '''

    for phase in phases:
        (proj_xcoord,proj_ycoord) = indices((Nx,Ny))
        k = 2.0 * pi * num_fringes_across_image / Ny
        fringe_pattern = pow(0.5 + 0.5*cos(k*proj_ycoord + phase), gamma)
        fringe_pattern_8bit = uint8(rint(255.0 * fringe_pattern / amax(fringe_pattern)))

        if (gamma == 1.0):
            filename = f'fringe_phi{int(phase*180.0/pi):03}.png'
        else:
            filename = f'fringe_phi{int(phase*180.0/pi):03}_gamma{int(gamma):1}.png'
        imsave(filename, fringe_pattern_8bit)

    return

## ==============================================================================================
def fpp_4_uniform_frames(imagestack):
    image_1minus3 = imagestack[:,:,1] - imagestack[:,:,3]
    image_0minus2 = imagestack[:,:,0] - imagestack[:,:,2]

    bias_image = 0.25 * sum(imagestack, axis=2)

    contrast_image = zeros_like(image_1minus3)
    contrast_image = 0.5 * sqrt(image_1minus3**2 + image_0minus2**2)

    phi_image = full(image_1minus3.shape, NaN)  ## create an array of NaNs
    phi_image = arctan2(image_1minus3, image_0minus2)

    return(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_N_uniform_frames(imagestack):
    (Nx,Ny,num_images) = imagestack.shape
    bias_image = (1.0 / num_images) * sum(imagestack, axis=2)

    ## The first and second terms are used to calculate both the contrast and phi images.
    first_term = 0.0
    second_term = imagestack[:,:,0]
    for n in range(1,num_images):
        first_term += imagestack[:,:,n] * sin(2.0 * pi * n / num_images)
        second_term += imagestack[:,:,n] * cos(2.0 * pi * n / num_images)

    contrast_image = (2.0 / num_images) * sqrt(first_term**2 + second_term**2)
    phi_image = arctan2(first_term, second_term)

    return(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_estimate_phi(imagestack, deltas):
    (Nx,Ny,num_images) = imagestack.shape
    phi_image = zeros((Nx,Ny), 'float32')
    H = zeros((3,3), 'float32')
    G = zeros((3,Ny), 'float32')

    c = cos(deltas)
    s = sin(deltas)
    cs = sum(c * s)
    cc = sum(c**2)
    ss = sum(s**2)

    H[0,0] = num_images
    H[0,1] = sum(c)
    H[0,2] = sum(s)

    H[1,0] = H[0,1]
    H[1,1] = cc
    H[1,2] = cs

    H[2,0] = H[0,2]
    H[2,1] = cs
    H[2,2] = ss

    G0 = sum(imagestack, axis=2)
    G1 = sum(imagestack * c, axis=2)
    G2 = sum(imagestack * s, axis=2)
    G = dstack([G0, G1, G2])[:,:,:,newaxis]
    X = squeeze(dot(inv(H),G))
    phi_image = arctan2(-X[2,:], X[1,:])
    bias_image = X[0,:]
    contrast_image = sqrt(X[1,:]**2 + X[2,:]**2)

    return(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_estimate_deltas(imagestack, phi_image, nrows=10):
    ## If you want to use only a small portion of the image to estimate the deltas, then make "xmax" a small number.
    ## If you want to use the entire image to estimate the deltas, then use xmax=None
    (Nx,Ny,num_images) = imagestack.shape
    Hprime = zeros((3,3), 'float32')
    Gprime = zeros((3,num_images), 'float32')
    delta_vectors = zeros((num_images,nrows))

    Hprime[0,0] = Ny

    c = cos(phi_image)
    s = sin(phi_image)
    cs = c * s
    cc = c**2
    ss = s**2

    Gprime[0,:] = sum(imagestack, axis=(0,1))
    Gprime[1,:] = sum(imagestack * c[:,:,newaxis], axis=(0,1))
    Gprime[2,:] = sum(imagestack * s[:,:,newaxis], axis=(0,1))

    ## Do the estimate separately for each row in the image, for a total of nrows. Then take the average.
    ## After a good deal of poking around, I still can't find out why the algorithm behaves so much better
    ## if I keep the Hprime matrix inside the x-loop and update the estimate while traversing few rows,
    ## rather than getting an Hprime matrix for the entire image. It's only 10 rows, and so it is not slow,
    ## but not understanding the issue is irritating.
    for x in range(nrows):
        cx = sum(c[x,:])
        sx = sum(s[x,:])
        cx_cx = sum(cc[x,:])
        cx_sx = sum(cs[x,:])
        sx_sx = sum(ss[x,:])

        Hprime[0,1] = cx
        Hprime[0,2] = sx

        Hprime[1,0] = cx
        Hprime[1,1] = cx_cx
        Hprime[1,2] = cx_sx

        Hprime[2,0] = sx
        Hprime[2,1] = cx_sx
        Hprime[2,2] = sx_sx

        Xprime = dot(inv(Hprime), Gprime)
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

#files = sort(glob('./figures/lens_crop*.jpg'))
files = sort(glob('./figures/fringe_phi*.png'))

## Make a list of the images to use, and then convert the list into a Numpy array 3D image stack.
imagestack = []
for file in files:
    imagestack.append(float32(imread(file)))
imagestack = dstack(imagestack)

(Nx,Ny,num_images) = imagestack.shape
print('imagestack shape =', imagestack.shape)

#deltas = array([0.0, pi/2.0, pi, 3.0*pi/2.0])
#(phi4_image, contrast4_image, bias4_image) = fpp_4_uniform_frames(imagestack)
#(phi_imageN, contrast_imageN, bias_imageN) = fpp_N_uniform_frames(imagestack)
(phi_image, contrast_image, bias_image, deltas) = fpp_estimate_deltas_and_phi(imagestack)
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
