'In this file you can find and change the parameters used in the network'

from numpy import sqrt
import numpy as np
seed = 362945635
#np.random.seed()
#seed = np.random.randint(1000000000)
#artifical network
N = 1000 #networksize
sampling_rate = 5
dt = 1./sampling_rate
tau_exponential_kernel = 5
mu = 1
alpha = sqrt(1/8)*mu
factor_ron_roff = 2
tau = 250
ron = 1./(tau*(1+factor_ron_roff))
roff = factor_ron_roff*ron


duration = 10*1000
qon_qoff_type = 'balanced'
scale = 17

mean_firing_rate = (0.1)/1000
stdq = mean_firing_rate*alpha
kernel = 'exponential'
regime = 1


#genereric parameters
baseline = 0
theta = 0