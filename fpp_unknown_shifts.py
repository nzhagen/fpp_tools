from numpy import *
from numpy.random import default_rng
from numpy.linalg import inv
import matplotlib.pyplot as plt
from imageio import imread, imsave

## ==============================================================================================
def fpp_estimate_deltas_and_phi(list_of_images, eps=5.0E-4):
    num_images = len(list_of_images)
    img0 = list_of_images[0]
    (Nx,Ny) = img0.shape
    deltas_estimated = default_rng(seed=0).uniform(0.0, 2.0*pi, num_images)
    #deltas_estimated = arange(num_images) * 2.0 * pi / (num_images - 1.0)
    niter = 10

    phi_image = zeros((Nx,Ny), 'float32')
    delta_vectors = zeros((num_images,Nx))
    A = zeros((3,3), 'float32')
    B = zeros((3,Ny), 'float32')
    Aprime = zeros((3,3), 'float32')
    Bprime = zeros((3,num_images), 'float32')

    for x in range(Nx):
        list_of_lines = [image[x,:] for image in list_of_images]

        for k in range(niter):
            deltas = array(deltas_estimated)

            A[0,0] = num_images
            A[0,1] = sum(cos(deltas))
            A[0,2] = sum(sin(deltas))

            A[1,0] = sum(cos(deltas))
            A[1,1] = sum(cos(deltas) * cos(deltas))
            A[1,2] = sum(cos(deltas) * sin(deltas))

            A[2,0] = sum(sin(deltas))
            A[2,1] = sum(cos(deltas) * sin(deltas))
            A[2,2] = sum(sin(deltas) * sin(deltas))

            B[0,:] = sum(list_of_lines, axis=0)
            B[1,:] = list_of_lines[0] * cos(deltas[0])
            B[2,:] = list_of_lines[0] * sin(deltas[0])
            for n in range(1, num_images):
                B[1,:] += list_of_lines[n] * cos(deltas[n])
                B[2,:] += list_of_lines[n] * sin(deltas[n])

            X = dot(inv(A),B)
            #a = X[0,:]
            b = X[1,:]
            c = X[2,:]
            phi = arctan2(-c, b)

            Aprime[0,0] = Ny
            Aprime[0,1] = sum(cos(phi))
            Aprime[0,2] = sum(sin(phi))

            Aprime[1,0] = sum(cos(phi))
            Aprime[1,1] = sum(cos(phi) * cos(phi))
            Aprime[1,2] = sum(cos(phi) * sin(phi))

            Aprime[2,0] = sum(sin(phi))
            Aprime[2,1] = sum(cos(phi) * sin(phi))
            Aprime[2,2] = sum(sin(phi) * sin(phi))

            Bprime[0,:] = array([sum(line) for line in list_of_lines])
            Bprime[1,:] = array([sum(line * cos(phi)) for line in list_of_lines])
            Bprime[2,:] = array([sum(line * sin(phi)) for line in list_of_lines])

            Xprime = dot(inv(Aprime), Bprime)
            #a_prime = Xprime[0,:]
            b_prime = Xprime[1,:]
            c_prime = Xprime[2,:]
            deltas_estimated = arctan2(-c_prime, b_prime)

            epsilons = array(deltas_estimated - deltas)
            mean_epsilon = mean(abs(epsilons))

            #print(f'x={x}, k={k}, mean_epsilon={mean_epsilon:.3e}, deltas=', deltas*180.0/pi)
            #plt.figure()
            #plt.plot(unwrap(phi))
            #plt.show()

            if (mean_epsilon < eps):
                break

        phi_image[x,:] = phi
        delta_vectors[:,x] = deltas

    return(phi_image, delta_vectors)

## ==============================================================================================
## ==============================================================================================

xmin = 232
xmax = 744
ymin = 192
ymax = 850
xloc = 250

img0 = float32(imread('./figures/lens_crop_000.jpg'))[xmin:xmax,ymin:ymax]
img1 = float32(imread('./figures/lens_crop_090.jpg'))[xmin:xmax,ymin:ymax]
img2 = float32(imread('./figures/lens_crop_180.jpg'))[xmin:xmax,ymin:ymax]
img3 = float32(imread('./figures/lens_crop_270.jpg'))[xmin:xmax,ymin:ymax]
list_of_images = [img0, img1, img2, img3]

(phi_image, delta_vectors) = fpp_estimate_deltas_and_phi(list_of_images)
mean_deltas = mean(delta_vectors, axis=1)
mean_deltas -= mean_deltas[0]
print('mean deltas [deg] =', mean_deltas*180.0/pi)

plt.figure('img0')
plt.imshow(img0)

plt.figure('phi')
plt.imshow(phi_image)
plt.show()
