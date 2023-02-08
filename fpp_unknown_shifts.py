from numpy import *
from numpy.random import default_rng
from numpy.linalg import inv
import matplotlib.pyplot as plt
from imageio import imread, imsave
from glob import glob

## ==============================================================================================
def generate_and_save_fringe_patterns(filebase, Nx, Ny, phases, num_fringes=10, gamma=1.0):
    '''
    Create images containing sinusoidal profiles, and saves them as PNG files. The files will be saved with the
    filename pattern [filebase]###.png if using gamma=1, or [filebase]###_gamma##.png, where the two-digit number
    after 'gamma' is the gamma value times 10.

    Parameters
    ----------
    filebase : str
        The base (folder + beginning of the filename) of the filenames to save to.
    Nx : int
        The image height dimension in pixels.
    Ny : int
        The image width dimension in pixels.
    phases : list or array of floats
        The list of the phases (in radians) of the patterns to generate.
    num_fringes : float
        How many fringes should appear inside each projection frame.
    gamma : float
        The gamma value to use for nonlinear conversion of the sinusoid profile.

    Example
    -------
    generate_and_save_fringe_pattern('fringe_phi', 480, 640, 4, 12)
    '''

    for phase in phases:
        (proj_xcoord,proj_ycoord) = indices((Nx,Ny))
        k = 2.0 * pi * num_fringes_across_image / Ny
        fringe_pattern = pow(0.5 + 0.5*cos(k*proj_ycoord + phase), gamma)
        fringe_pattern_8bit = uint8(rint(255.0 * fringe_pattern / amax(fringe_pattern)))

        if (gamma == 1.0):
            filename = f'{filebase}{int(phase*180.0/pi):03}.png'
        else:
            filename = f'{filebase}{int(phase*180.0/pi):03}_gamma{int(10.0*gamma):2}.png'
        imsave(filename, fringe_pattern_8bit)

    return

## ==============================================================================================
def fpp_4_uniform_frames(imagestack):
    '''
    Using four fringe-projection images, with the fringe phase steps in 90deg increments, estimate the phase of the
    underlying object at each pixel in the image.

    Parameters
    ----------
    imagestack : list of images
        The four images to use for estimating the phase phi.

    Returns
    -------
    phi : 2D array of float32
        The phases of the underlying object at each pixel in the images.
    '''

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
    '''
    Using N fringe-projection images, with the fringe phase steps in equal increments of 360deg / N, estimate the
    phase of the underlying object at each pixel in the image.

    Parameters
    ----------
    imagestack : list of images
        The N images to use for estimating the phase phi.

    Returns
    -------
    phi : 2D array of float32
        The phases of the underlying object at each pixel in the images.
    '''

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
def fpp_4_nonuniform_frames(imagestack, deltas):
    (Nx,Ny,num_images) = imagestack.shape
    if (len(deltas) != num_images):
        raise ValueError('The number of phase shift deltas ({len(deltas)}) must equal the number of images ({num_images})!')

    upper_term = (imagestack[:,:,0] - imagestack[:,:,2]) * (cos(deltas[1]) - cos(deltas[3]))
    upper_term -= (imagestack[:,:,3] - imagestack[:,:,1]) * (cos(deltas[0]) - cos(deltas[2]))
    lower_term = (imagestack[:,:,0] - imagestack[:,:,2]) * (sin(deltas[1]) - sin(deltas[3]))
    lower_term -= (imagestack[:,:,3] - imagestack[:,:,1]) * (sin(deltas[0]) - sin(deltas[2]))

    phi_image = arctan2(upper_term, lower_term)

    ## At some point, I should sit down an calculate the corresponding formuylas for the bias and contrast.
    return(phi_image) #(phi_image, contrast_image, bias_image)

## ==============================================================================================
def fpp_N_nonuniform_frames(imagestack, deltas):
    (Nx,Ny,num_images) = imagestack.shape
    if (len(deltas) != num_images):
        raise ValueError('The number of phase shift deltas ({len(deltas)}) must equal the number of images ({num_images})!')

    Iterm1 = zeros((Nx,Ny))
    Iterm2 = zeros((Nx,Ny))
    delta_term1 = 0.0
    delta_term2 = 0.0
    delta_term3 = 0.0
    delta_term4 = 0.0

    for n in range(num_images):
        Iterm1 += imagestack[:,:,n] * sin(2.0 * pi * n / num_images)
        Iterm2 += imagestack[:,:,n] * cos(2.0 * pi * n / num_images)
        delta_term1 += cos(2.0 * pi * n / num_images) * cos(deltas[n])
        delta_term2 += cos(2.0 * pi * n / num_images) * sin(deltas[n])
        delta_term3 += sin(2.0 * pi * n / num_images) * cos(deltas[n])
        delta_term4 += sin(2.0 * pi * n / num_images) * sin(deltas[n])

    phi_image = arctan2((Iterm1 * delta_term1) - (Iterm2 * delta_term3),
                        (Iterm1 * delta_term2) - (Iterm2 * delta_term4))

    ## At some point, I should sit down an calculate the corresponding formuylas for the bias and contrast.
    return(phi_image) #(phi_image, contrast_image, bias_image)

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
