import numpy as onp
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
import jax.numpy as np
import optax
import jax
import functools

#model
def evaluate_model(F,x):
    return np.dot(F,x)*1e7

#cost function

def lsq_loss(x,F, y):
    y_pred = evaluate_model(F, x)
    loss = np.sum(np.square((y_pred-y)))/2
    return loss
    
#actually doesn't work as well as lsq_loss - I think there's something funny happening with std calculation
def weighted_lsq_loss(x,F, sigma, y):
    y_pred = evaluate_model(F, x)
    loss = np.sum(np.square((y_pred-y)/sigma))/2
    return loss
    
ipsflibfn = '/home/nfiudev/dev/yxin/JWST/miri_fqpm_ipsfs_220825.npy'
F = np.load(ipsflibfn)
data = np.load('../data_avg_bright.npy').ravel()
stds = np.load('../data_avg_dim.npy').ravel()

ngrid = 128
test_input = np.ones((ngrid,ngrid))*1e-6
test_input = np.ravel(test_input)

test_output = evaluate_model(F,test_input)
npix = 108

loss = lsq_loss(test_input,F,data)

#print(loss)

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
params = test_input
opt_state = optimizer.init(params)

## A simple update loop.
for n in range(200):
    print(n)
    grads = jax.grad(lsq_loss)(params, F, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
  
plt.figure()
plt.imshow(np.reshape(params,(ngrid,ngrid)))
plt.colorbar()
plt.savefig('../figures/recovered_intensity.pdf')
