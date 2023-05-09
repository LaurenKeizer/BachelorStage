''' MI_calculation.py
    File containing the functions to estimate the mutual information betweem the hidden state, input and
    output spike train.

    Described in:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017).
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series.
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
'''
import numpy as np
import pandas as pd
import Foundations.helpers as helper

def analyze_exp(hidden_state, input_theory, theta, spiketrain, samples, weight, scale):
    ''' Analyzes the the hidden state and the input that was created by the ANN to
        create the Output dictionary.
        Equations 13 & 14

        INPUT:
        ron, roff (kHz): switching speed of the hidden state
        x: array with hidden state values over time
        input_theory: array with unscaled input current values (output from ANN)
        dt: binsize recordings (ms)
        spiketrain: array (same size as hidden_state and input_theory) of 0 (no spike) and 1 (spike)
        samples: The amount of samples taken so duration/sampling rate
        weights: The weigths under which the analyses is done

        OUTPUT
        Output-dictionary with keys:
        MI_i        : mutual information between hidden state and input current
        xhat_i      : array with hidden state estimate based on input current
        MSE_i       : mean-squared error between hidden state and hidden state estimate based on input current
        MI          : mutual information between hidden state and spike train
        qon, qoff   : spike frequency during on, off state in spike train
        xhatspikes  : array with hidden state estimate based on spike train
        MI          : mean-squared error between hidden state and hidden state estimate based on spike train
    '''
    Output = {}
    Output['weights'] = weight
    Output['scales']= scale

    # Input
    Output['Hxx'], Output['Hxy'], Output['MI_i'], L_i = helper.calc_MI_input(input_theory, theta, hidden_state)
    Output['xhat_i'] = 1. / (1 + np.exp(-L_i))
    Output['MSE_i'] = np.sum((hidden_state - Output['xhat_i'])**2)/samples

    # Output
    Output['Hxx_2'], Output['Hxy_2'], Output['MI'], L, Output['qon'], Output['qoff'] = helper.calc_MI_ideal(spiketrain, hidden_state)
    Output['xhatspikes'] = 1./(1 + np.exp(-L))
    Output['MSE'] = np.sum((hidden_state - Output['xhatspikes'])**2)/samples
    Output['F'] = Output['MI_i'] / Output['Hxx']
    Output['F_I'] = Output['MI']/Output['MI_i']

    return pd.DataFrame.from_dict(Output, orient='index').T





