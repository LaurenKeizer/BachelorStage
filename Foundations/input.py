# This code will generate a hidden state and a input spike train. This code is based on the code of
# mik schutte

"""
    input.py

    This file contains the input class that generates the hidden state and the input theory.

    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017).
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series.
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
"""

import inspect
import os
import sys

from Documentation import parameters as p
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class Input:

    def __init__(self):
        # For all
        self.dt = p.dt
        self.T = p.duration
        self.fHandle = [None, None]
        self.xseed = None

        self.seed = self.set_seed()
        self.input = None

        # For Markov models
        self.ron = p.ron
        self.roff = p.roff

        self.qon, self.qoff = self.create_qonqoff(p.networksize, p.mean_firing_rate, p.stdq, self.seed)

        self.kernel = p.kernel
        self.kerneltau = p.tau_exponentioal_kernel

        self.xfix = None

        self.get_all()
        self.x = self.markov_hiddenstate()

    def get_hidden(self):
        return self.x

    # Get dependend variables
    def get_tvec(self):
        """Generate tvec and save the length
        """
        self.tvec = np.arange(self.dt, self.T + self.dt, self.dt)
        self.length = len(self.tvec)

    def generate(self):
        """Generate input and x from fHandle.
        """
        if not self.fHandle:
            print('fHandle isn''t provided object')
        else:
            [self.input, self.x] = self.fHandle

    def get_tau(self):
        '''Generates tau based on the hidden state switch rate
           i.e. ron/roff
        '''
        if self.ron == None or self.roff == None:
            print('Tau not defined, missing ron/roff')
        else:
            self.tau = 1 / (self.ron + self.roff)

    def get_p0(self):
        '''Generates the probability of finding the hidden state
           in the 'ON' state.
        '''
        if self.ron == None or self.roff == None:
            print('P0 not defined, missing ron/roff')
        else:
            self.p0 = self.ron / (self.ron + self.roff)

    def get_theta(self):
        '''Generates the firing rate differences.
        '''
        if self.qon == [] or self.qoff == []:
            print('Theta not defined, missing qon/qoff')
        else:
            sum(self.qon - self.qoff)

    def get_w(self):
        '''Generates the weight matrix based on qon/qoff.
        '''
        if self.qon == [] or self.qoff == []:
            print('Weight not defined, missing qon/qoff')
        else:
            self.w = np.log(self.qon / self.qoff)

    def set_seed(self, seed=None):
        if seed == None:
            np.random.seed()
            seed = np.random.randint(1000000000)
        # self.seed = seed
        self.xseed = seed

        return seed

    def get_all(self):
        '''Runs all the functions to create dependent variables.
        '''
        self.get_tvec()
        self.generate()
        self.get_tau()
        self.get_p0()
        self.get_theta()
        self.get_w()

    @staticmethod
    def create_qonqoff(N, meanq, stdq, qseed=None):
        ''' Generates normally distributed [qon, qoff] with qon and qoff
            being a matrix filled with the firing rate of each neuron based
            on the hidden state.

            INPUT
            N (int): number of neurons in the ANN
            meanq (float): mean of the normal distribution from which q is sampled
            stdq (float): standard deviation of the normal distribution
            qseed (int): seed to set the random number generator (rng)

            OUTPUT
            [qon, qoff]: array containing the firing rates of the neurons during both states
        '''
        # Sample qon and qoff from a rng.
        np.random.seed(qseed)
        qoff = np.random.randn(N, 1)
        qon = np.random.randn(N, 1)

        # Consider the normal distribution
        if N > 1:
            qoff = qoff / np.std(qoff)
            qon = qon / np.std(qon)
        qoff = stdq * (qoff - np.mean(qoff)) + meanq
        qon = stdq * (qon - np.mean(qon)) + meanq

        # Set all negative firing rates to absolute value
        qoff[qoff < 0] = abs(qoff[qoff < 0])
        qon[qon < 0] = abs(qon[qon < 0])

        return [qon, qoff]

    def markov_hiddenstate(self):
        ''' Takes ron and roff from class object and generates
            the hiddenstate if hidden is empty.
        '''
        np.random.seed(self.xseed)

        # Generate x
        if self.xfix == None:
            self.get_p0()
            xs = np.zeros(np.shape(self.tvec))

            # Initial value
            i = np.random.rand()
            if i < self.p0:
                xs[0] = 1
            else:
                xs[0] = 0

            # Make x
            for n in np.arange(1, self.length):
                i = np.random.rand()
                if xs[n - 1] == 1:
                    if i < self.roff * self.dt:
                        xs[n] = 0
                    else:
                        xs[n] = 1
                else:
                    if i < self.ron * self.dt:
                        xs[n] = 1
                    else:
                        xs[n] = 0
        else:
            xs = self.xfix

        return xs

    def markov_input(self, dynamic=False):
        ''' Takes qon, qoff and hiddenstate and generates input.
            Optionally when dynamic is a dictinary of g0_values it
            generates a conductance over time based on the hidden state.
        '''
        xs = self.x
        nt = self.length
        w = np.log(self.qon / self.qoff)

        ni = range(len(self.qon))

        # Make spike trains (implicit)
        stsum = np.zeros((nt, 1))
        if self.kernel != None:
            if self.kernel == 'exponential':
                tfilt = np.arange(0, 5 * self.kerneltau + self.dt, self.dt)
                kernelf = np.exp(-tfilt / self.kerneltau)
                kernelf = kernelf / (self.dt * sum(kernelf))
            elif self.kernel == 'delta':
                kernelf = 1. / self.dt

        xon = np.where(xs == 1)
        xoff = np.where(xs == 0)
        np.random.seed(self.seed)

        # Create the input generated by the artificial neural network
        for k in ni:
            randon = np.random.rand(np.shape(xon)[0], np.shape(xon)[1])
            randoff = np.random.rand(np.shape(xoff)[0], np.shape(xoff)[1])
            sttemp = np.zeros((nt, 1))
            sttempon = np.zeros(np.shape(xon))
            sttempoff = np.zeros(np.shape(xoff))

            sttempon[randon < self.qon[k] * self.dt] = 1.
            sttempoff[randoff < self.qoff[k] * self.dt] = 1.

            sttemp[xon] = np.transpose(sttempon)
            sttemp[xoff] = np.transpose(sttempoff)

            if dynamic:
                stsum = stsum + dynamic[k] * sttemp
            else:
                stsum = stsum + w[k] * sttemp

                # #SanityCheck for individual spikes
            # plt.plot(sttemp)
            # plt.show()

        if self.kernel != None:
            stsum = np.convolve(stsum.flatten(), kernelf, mode='full')

        stsum = stsum[0:nt]
        ip = stsum

        return ip
        self.input = ip

# %%
