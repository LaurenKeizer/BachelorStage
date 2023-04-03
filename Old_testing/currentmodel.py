''' models.py

    Python file containing different neuron models used in simulations. Modelled in Brian2

    The Wang Buszaki model is described in:
    Xiao-Jing Wang & György Buzsáki, (1996). Gamma Oscillation
    by synaptic inhibition in a hippocampal interneuronal network
    model. doi: https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996
    Original from: https://brian2.readthedocs.io/en/stable/examples/frompapers.Wang_Buszaki_1996.html

    Fitting of the parameters used in Barrel_PC and Barrel_IN classes are described in:
    22.	Sterl, X. and Zeldenrust, F. (2020) Dopamine modulates ﬁring rates and information transfer
    in inhibitory and excitatory neurons of rat barrel cortex, but shows no clear inﬂuence on neuronal
    parameters. Bsc. University of Amsterdam. Available at: https://scripties.uba.uva.nl/search?id=715234.
'''


import os
os.chdir("C:\\Users\\laure\\DataspellProjects\\BacherlorStage\\Old_testing")
import brian2 as b2
import numpy as np

class Barrel_PC:
    ''' Hodgkin-Huxley model of a Pyramidal Cell in the rat barrel cortex.

        INPUT:
        clamp_type (str): type of input, ['current' or 'dynamic']
        dt (float): time step of the simulation in miliseconds.

        OUTPUT:
        Neurongroup of the Neuron
        StateMonitor, SpikeMonitor: Brian2 StateMonitor with recorded fields
        ['v', 'input' or 'conductance'] and SpikeMonitor which records spikes

    '''

    def __init__(self, dt,  inj_input):
        self.dt = dt
        self.stored = False
        self.inj_input = inj_input
        self.create_namespace()
        self.make_model()


    def make_model(self):

        #Setting the input
        eqs_input = '''I_inj = inj_input(t) : amp'''

        # Model the neuron with differential equations
        eqs = '''
            Vh_m = 3.583881 * k_m - 53.294454*mV : volt
            m = 1 / (1 + exp(-(v - Vh_m) / k_m)) : 1
            h = 1 / (1 + exp((v - Vh_h) / k_h)) : 1

            alpha_n = (0.032 * 5. / exprel((15. -v/mV + VT/mV) / 5.))/ms : Hz
            beta_n = (0.5 * exp((10. - v/mV + VT/mV) / 40.))/ms : Hz
            dn/dt = alpha_n * (1 - n) - beta_n * n : 1

            I_leak = -gL * (v - EL) : amp
            I_Na = -gNa * m**3 * h * (v - ENa) : amp
            I_K = -gK * n**4 * (v - EK) : amp

            dv/dt = (I_leak + I_Na + I_K + I_inj) / Cm : volt
            '''

        # Neuron & parameter initialization
        neuron = b2.NeuronGroup(2, model=eqs + eqs_input, method='exponential_euler',
                                threshold='m > 0.5', refractory=2 * b2.ms, reset=None, dt=self.dt * b2.ms,
                                namespace=self.namespace)
        neuron.v = -65 * b2.mV
        self.neuron = neuron

    def create_namespace(self):
        area = 20000 * b2.umetre ** 2
        parameters = np.loadtxt('PC_parameters.csv', delimiter=',')
        # Ni = np.random.randint(np.shape(parameters)[1])
        Ni = 54

        namespace = {'area' : 20000 * b2.umetre ** 2,
                     'param': np.log(10),
                     'Cm' : 203.94 * b2.pfarad / area * b2.cm ** 2,
                     'gL' : 18.92 * b2.nsiemens / area * b2.cm ** 2,
                     'gNa' : 92.22* b2.usiemens / area * b2.cm ** 2,
                     'gK' : 57.56 * b2.usiemens / area * b2.cm ** 2,
                     'EL' : -65 * b2.mV,
                     'ENa' : 50 * b2.mV,
                     'EK' : -90 * b2.mV,
                     'Er_e' : 0 * b2.mV,
                     'Er_i' : -75 * b2.mV,
                     'k_m' : parameters[4][Ni] * b2.volt,
                     'k_h' : parameters[5][Ni] * b2.volt,
                     'Vh_h' : parameters[6][Ni] * b2.volt,
                     'VT' : -63 * b2.mV,
                     'inj_input': self.inj_input}

        self.namespace = namespace


    def getNeurongroup(self):
        return self.neuron

    def getNamespace(self):
        return self.namespace


class Barrel_IN:
    ''' Hodgkin-Huxley model of an Inter neuron in the rat barrel cortex.

        INPUT:
            clamp_type (str): type of input, ['current' or 'dynamic']
            dt (float): time step of the simulation in miliseconds.

        OUTPUT:
            StateMonitor, SpikeMonitor: Brian2 StateMonitor with recorded fields
            ['v', 'input' or 'conductance'] and SpikeMonitor which records spikes

        The parameters used in this model have been fitted by Xenia Sterl under
        the supervision of Fleur Zeldenrust. Full description can be found at:
        Xenia Sterl, Fleur Zeldenrust, (2020). Dopamine modulates firing rates and information
        transfer in inhibitory and excitatory neurons of rat barrel cortex, but shows no clear
        influence on neuronal parameters. (Unpublished bachelor's thesis)
    '''

    def __init__(self, dt,  inj_input, Input=True):
        self.dt = dt
        self.stored = False
        self.inj_input = inj_input
        self.create_namespace()
        self.Input = Input
        self.make_model()


    def make_model(self):
        # Determine the simulation
        if self.Input:
            eqs_input = '''I_inj = inj_input(t) : amp'''
        else:
            eqs_input = '''I_inj = 0*amp : amp'''


        # Model the neuron with differential equations
        eqs = '''
        # Activation gates Na channel
            m = 1. / (1. + exp(-(v - Vh) / ik)) : 1
            Vh = 3.223725 * ik - 62.615488*mV : volt
            
        # Inactivation gates Na channel
            dh/dt = 5. * (alpha_h * (1 - h)- beta_h * h) : 1
            alpha_h = 0.07 * exp(-(v + 58.*mV) / (20.*mV))/ms : Hz
            beta_h = 1. / (exp(-0.1/mV * (v + 28.*mV)) + 1.)/ms : Hz
        # Activation gates K channel
            
            dn/dt = 5. * (alpha_n * (1. - n) - beta_n * n) : 1
            alpha_n = 0.01/mV * 10*mV / exprel(-(v + 34.*mV) / (10.*mV))/ms : Hz
            beta_n = 0.125 * exp(-(v + 44.*mV) / (80.*mV))/ms : Hz
            
        # Activation gates K3.1 channel
            dn3/dt = alphan3 * (1. - n3) - betan3 * n3 : 1
            alphan3 = (1. / exp(((param * ((-0.029 * v + (1.9*mV))/mV)))))/ms : Hz
            betan3 = (1. / exp(((param * ((0.021 * v + (1.1*mV))/mV)))))/ms : Hz
            
        # Currents
            I_leak = -igL * (v - iEL) : amp
            I_Na = -igNa * m**3 * h * (v - iENa) : amp
            I_Kl = -igK * n**4 * (v - iEK) : amp
            I_K3 = -igK3 * n3**4 * (v - iEK) : amp
            dv/dt = (I_leak + I_Na + I_Kl + I_K3 + I_inj) / iCm : volt
        '''

        # Neuron & parameter initialization

        neuron = b2.NeuronGroup(1, model=eqs + eqs_input, method='exponential_euler',
                                threshold='m > 0.5', refractory=2 * b2.ms, reset=None, dt=self.dt * b2.ms,
                                namespace=self.namespace)
        neuron.v = -65 * b2.mV

        self.neuron = neuron

    def create_namespace(self):
        parameters = np.loadtxt('IN_parameters.csv', delimiter=',')
        # Ni = np.random.randint(np.shape(parameters)[1])
        Ni = 12
        area = 20000 * b2.umetre ** 2

        namespace = {'area' :  20000 * b2.umetre ** 2,
                     'param': np.log(10),
                     'iCm' : 380.02 * b2.pfarad / area * b2.cm ** 2,
                     'igL' : 4.60* b2.nsiemens / area * b2.cm ** 2,
                     'igNa' : 107.52* b2.usiemens / area * b2.cm ** 2,
                     'igK' :  50.54 * b2.usiemens / area * b2.cm ** 2,
                     'igK3' : 3.85 * b2.msiemens / area * b2.cm ** 2,
                     'iEL' : -65 * b2.mV,
                     'iENa' : 50 * b2.mV,
                     'iEK' : -90 * b2.mV,
                     'iEr_e' : 0 * b2.mV,
                     'iEr_i' : -75 * b2.mV,
                     'ik' : parameters[4][Ni] * b2.volt,
                     'inj_input': self.inj_input}

        self.namespace = namespace
        #should I do this?? or should it be in a return function


    def getNeurongroup(self):
        return self.neuron

    def getNamespace(self):
        return self.namespace


#%%
