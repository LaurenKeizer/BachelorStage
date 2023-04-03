

#This file contains the parameters you want to use in your simulation.
#There are different types of initialisation which you can switch between giving the regime or calling it
from numpy import sqrt


class Parameters:

    def __init__(self, regime):
        self.netwoksize = 1000
        self.sampling_rate = 5
        self.dt = 1..sampling_rate
        self.tau_exponential_kernel =5
        self.mu = 1
        self.alpha = sqrt(1/8)*self.mu
        self.factor_ron_roff = 2
        self.tau = 250
        self.ron = 1./(self.tau*(1+self.factor_ron_roff))
        self.roff = self.factor_ron_roff*self.ron
        self.mean_firing_rate = (0.1)/1000
        self.duration = 10*1000
        self.qon_qoff_type = 'balanced'
        self.scale = 17

        self.stdq = self.mean_firing_rate*self.alpha
        self.kernel = 'exponential'
        self.regime = 1


        #genereric parameters
        self.baseline = 0
        self.theta = 0


    def make_inhibitory(self):
        self.tau = 50
        self.ron = 1./(self.tau*(1+self.factor_ron_roff))
        self.roff = self.factor_ron_roff*self.ron
        self.mean_firing_rate = (0.5)/1000
        self.duration = 20000

    def S(self):
        self.ron = 6.7
        self.roff = 13.3
        self.tau = 50



'''#Parameters for the input in the PC Setting
#networksize = 1000
sampling_rate = 5
dt = 1./sampling_rate
tau_exponentioal_kernel = 5
alpha = sqrt(1/8)
factor_ron_roff = 2

#PC settings

tau = 250
ron = 
roff =
mean_firing_rate = (0.1)/1000
duration = 10*1000
qon_qoff_type = 'balanced'
scale = 17

#IN Settings
tau = 50
ron = 1./(tau_IN*(1+factor_ron_roff))
roff = factor_ron_roff*ron_IN
mean_firing_rate = (0.5)/1000
duration = 20000

stdq = mean_firing_rate*alpha
kernel = 'exponential'
regime = 1


#genereric parameters
baseline = 0
theta = 0


#The parameters for an inhibithory neuron 
tau = 50
ron = 1./(tau_IN*(1+factor_ron_roff))
roff = factor_ron_roff*ron_IN
mean_firing_rate = (0.5)/1000
duration = 20000


target_PC = 1.4233
target_IN = 6.6397
on_off_ratio = 1.5
scales = {'CC_PC':19, 'DC_PC':30, 'CC_IN':17, 'DC_IN':6}
N_runs = 1
scale_PC = 19
scale_IN = 17
'''

#%%
