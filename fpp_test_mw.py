from numpy import *
from numpy.random import normal
import matplotlib.pyplot as plt
import fpp_tools as fpp

## ====================================================================================
## ====================================================================================

order = 2       ## do the fitting to what order multiple of the main modulation frequency (use 1, 2, 3, or 4)
noise = 0.04

nphases = 6
phases = arange(nphases) * 2.0 * pi / nphases
imagestack = fpp.generate_fringe_patterns(256, 256, phases, 15, gamma=1.5)

## Normalize the intensities so that the maximum is 1.
imagestack /= amax(imagestack)
(Nx,Ny,num_images) = imagestack.shape
print('imagestack shape =', imagestack.shape)

if (noise > 0.0):
    imagestack += normal(0.0, noise, imagestack.shape)
    imagestack -= amin(imagestack)
    imagestack /= amax(imagestack)

plt.figure('raw img0')
plt.imshow(imagestack[:,:,0])
plt.colorbar()

if (order == 1):
    (phi_img, amplitude_img, bias_img, deltas) = fpp.estimate_deltas_and_phi_lsq(imagestack, order=order)
    gamma_img = None
elif (order > 1):
    (phi_img, amplitude_img, bias_img, gamma_img, deltas) = fpp.estimate_deltas_and_phi_lsq(imagestack, order=2)

deltas -= amin(deltas)
deltas[deltas > 360.0] = deltas[deltas > 360.0] - 360.0
deltas = sort(deltas)
print(f"est. deltas [deg] = {array2string(deltas*180.0/pi, formatter={'float': lambda x:f'{x:.2f}'}, separator=', ')}")

contrast_img = amplitude_img / bias_img

plt.figure('contrast')
plt.imshow(contrast_img)
plt.title(f'mean(contrast)= {mean(contrast_img):.2f}')
plt.colorbar()

plt.figure('bias')
plt.imshow(bias_img)
plt.title(f'mean(bias)= {mean(bias_img):.2f}')
plt.colorbar()

if gamma_img is not None:
    plt.figure('gamma')
    plt.imshow(gamma_img)
    plt.title(f'mean(gamma)= {mean(gamma_img):.2f}')
    plt.colorbar()
    print(f'Average gamma value: {mean(gamma_img):.2f}')

plt.figure('phi')
plt.imshow(phi_img)

## In the unwrapped phase image, make sure that the smallest phase is zero. If the phase is increasing, then we subtract the
## smallest value. If decreasing, then we add the smallest value.
unwrapped_phi = unwrap(phi_img)
avg_gradient = mean(gradient(unwrapped_phi[:,Ny//2]))
if (avg_gradient > 0.0):
    unwrapped_phi += amax(unwrapped_phi)
else:
    unwrapped_phi -= amax(unwrapped_phi)

unwrapped_phi -= unwrapped_phi[0,0]

plt.figure('phi_unwrapped')
plt.imshow(unwrapped_phi)
plt.colorbar()

phi_curve = unwrapped_phi[100,:]

plt.figure('phi_unwrapped[100,:]')
plt.plot(unwrapped_phi[100,:])
plt.plot(unwrapped_phi[101,:])

## Subtract a linear fit to the phi curve.
x = arange(len(unwrapped_phi[100,:]))
phi_curve = zeros_like(unwrapped_phi[100,:])
for i in range(100,200):
    fit_x = polyfit(x, unwrapped_phi[i,:], 1)
    linear_baseline = poly1d(fit_x) # create the linear baseline function
    phi_curve += unwrapped_phi[i,:] - linear_baseline(x) # subtract the baseline from μ_Y

phi_curve_avg = phi_curve / 100.0

plt.figure('nonlinear error: phi_curve - baseline')
plt.plot(phi_curve_avg)

plt.show()
