import numpy as onp
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
import jax.numpy as np
import optax
import jax
import functools
from jax.scipy import ndimage as jndimage

def sum_gradsq_image(image,epsilon=0):
    jgrid_x, jgrid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    def_grid_x = jgrid_x + 1.0
    def_grid_y = jgrid_y + 1.0

    shifted_im = jndimage.map_coordinates(image, [def_grid_y, def_grid_x], order=1)
    excl_border = 1
    diff_im = shifted_im[excl_border:-excl_border,excl_border:-excl_border]-image[excl_border:-excl_border,excl_border:-excl_border]

    sum_gradsq = np.sum(diff_im**2+epsilon)
    return sum_gradsq

#model
def evaluate_model(F,x):
    return np.dot(F,x)*1e7

#cost functions

def lsq_loss(x,F,sigma, y):
    y_pred = evaluate_model(F, x)
    loss = np.sum(np.square((y_pred-y)))/2
    return loss
    
def weighted_lsq_loss(x,F, sigma, y):
    y_pred = evaluate_model(F, x)
    loss = np.sum(np.square((y_pred-y)/sigma))/2
    return loss
   
def l1_loss(x,F, sigma,y):
    y_pred = evaluate_model(F, x)
    loss = np.sum(np.square((y_pred-y)/sigma))/2+np.sum(np.abs(x))*1e7*1e-1
    return loss
    
def tv_loss(x,F,sigma,y):
    y_pred = evaluate_model(F, x)
    x_reshaped = np.reshape(x,(128,128))
    reg = np.sqrt(sum_gradsq_image(x_reshaped,epsilon=1e-12))
    loss = np.sum(np.square((y_pred-y)/sigma))/2+reg*1e10
    return loss
    
def tsv_loss(x,F,sigma,y):
    y_pred = evaluate_model(F, x)
    x_reshaped = np.reshape(x,(128,128))
    reg = sum_gradsq_image(x_reshaped,epsilon=0)
    loss = np.sum(np.square((y_pred-y)/sigma))/2+reg*1e14
    return loss

#TODO - need to define entropy relative to something    
def maxent_loss(x,F, y):

    return NotImplementedError()
    

    
ipsflibfn = '/home/nfiudev/dev/yxin/JWST/miri_fqpm_ipsfs_220825.npy'
F = np.load(ipsflibfn)
data = np.load('../data_avg_bright.npy').ravel()
stds = np.load('../data_std_bright.npy').ravel()

ngrid = 128
test_input = np.ones((ngrid,ngrid))*1e-6
test_input = np.ravel(test_input)

test_output = evaluate_model(F,test_input)
npix = 108

start_learning_rate = 1e-8
optimizer = optax.chain(optax.adam(start_learning_rate),optax.keep_params_nonnegative())

# Initialize parameters of the model + optimizer.
params = test_input
opt_state = optimizer.init(params)

nsteps=300
loss_vals = onp.zeros(nsteps)

#test_loss = tv_loss(params+onp.random.normal(np.median(params),onp.shape(params)),F, stds,data)
#print(test_loss)

## A simple update loop.
for n in range(nsteps):
    print(n)
    loss_vals[n],grads = jax.value_and_grad(tsv_loss)(params, F, stds, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
  
plt.figure()
plt.imshow(np.reshape(params,(ngrid,ngrid)),cmap='magma')
plt.colorbar()
plt.savefig('../figures/recovered_intensity_tsv.pdf')

plt.figure()
plt.plot(loss_vals)
plt.savefig('../figures/loss_vals_tsv.pdf')
