import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from brian2 import *
from matplotlib import pyplot as plt
import Documentation.parameters as p
from Foundations.helpers import scale_input_theory, make_spiketrain

from Foundations.MI import analyze_exp
from Old_testing.make_dynamic_experiment_parameters import make_dynamic_experiments
from Old_testing.currentmodel import Barrel_PC, Barrel_IN
import pandas as pd

defaultclock = p.dt*ms

def backward_run(scale, weights, delay, seed):
    [input_theory, hidden_state] = make_dynamic_experiments(p.qon_qoff_type, p.duration, p.seed)

    inj_current = scale_input_theory(input_theory, 0, scale, p.dt)


    start_scope()
    #Initialising the neuron classes
    PC = Barrel_PC(p.dt, inj_current)
    IN = Barrel_IN(p.dt, None, Input=False) #the barrel neuron has no input
    #Getting the neuron groups
    PC_neuron = PC.getNeurongroup()
    IN_neuron = IN.getNeurongroup()

    net2 = Network(PC_neuron,
               IN_neuron)

    PC_M = StateMonitor(PC_neuron, 'v', record=True)
    PC_S = SpikeMonitor(PC_neuron, 'v', record = True)
    IN_M = StateMonitor(IN_neuron, 'v', record=True)

    net2.add(PC_M, IN_M, PC_S)
    w = 0
    #getting all the parameters for in the synapse
    syn = Synapses(PC_neuron, IN_neuron, on_pre='''v_post += 4.27*mV''', delay=0.6 * ms, dt=p.dt * ms)
    syn.connect(i=0, j=0)

    syn2 = Synapses(IN_neuron, PC_neuron, on_pre='''v_post += -w1*mV''', delay=0.7 * ms, dt=p.dt * ms)
    syn2.connect(i=0, j=1)
    net2.add(syn, syn2)

    net2.store()
    output = pd.DataFrame()
    samples = p.duration/p.dt

    for w1 in weights:
        net2.restore()
        w1 = w1
        net2.run(p.duration*ms)
        plotting(PC_M, IN_M, w1, scale)
        spiketrain = make_spiketrain(PC_S)
        output = pd.concat([output, analyze_exp(hidden_state, input_theory,0, spiketrain,samples, scale, w)], axis = 0)

    return output, spiketrain

def forward_run(scale, weights,seed):

    [input_theory, hidden_state] = make_dynamic_experiments(p.qon_qoff_type, p.duration, p.seed)

    inj_current = scale_input_theory(input_theory, 0, scale, p.dt)

    start_scope()
    #Initialising the neuron classes
    PC = Barrel_PC(p.dt, inj_current)
    IN = Barrel_IN(p.dt, inj_current)
    #Getting the neuron groups
    PC_neuron = PC.getNeurongroup()
    IN_neuron = IN.getNeurongroup()

    net2 = Network(PC_neuron,
               IN_neuron)

    PC_M = StateMonitor(PC_neuron, 'v', record=True)
    PC_S = SpikeMonitor(PC_neuron, 'v', record = True)
    IN_M = StateMonitor(IN_neuron, 'v', record=True)
    IN_S = SpikeMonitor(IN_neuron, 'v', record = True)

    net2.add(PC_M, IN_M, PC_S, IN_S)
    #getting all the parameters for in the synapse
    delay = 0.6

    syn2 = Synapses(IN_neuron, PC_neuron, on_pre='''v_post += -w1*mV''', delay= delay*ms, dt=p.dt * ms)
    syn2.connect(i=0, j=1)
    net2.add(syn2)

    net2.store()

    output_PC = pd.DataFrame()
    output_IN = pd.DataFrame()
    samples = p.duration/p.dt

    for w1 in weights:
        net2.restore()
        w1 = w1
        net2.run(p.duration*ms)
        plotting(PC_M, IN_M, w1, scale)
        spiketrain = make_spiketrain(PC_S)
        output_PC = pd.concat([output_PC, analyze_exp(hidden_state, input_theory,0, spiketrain,samples, w1, scale, PC_S.count)], axis = 0)
        output_IN = pd.concat([output_IN, analyze_exp(hidden_state, input_theory,0, spiketrain,samples, w1, scale, IN_S.count)], axis = 0)

    return output_PC, output_IN, spiketrain


def plotting(PC_M, IN_M, weight, scale):



    fig, axs = plt.subplots(2, 2, figsize=(14,6))
    fig.subplots_adjust(hspace=0.5)
    axs[0, 0].plot(PC_M.t/ms, PC_M.v[0]*1000, label= 'PC SM (pre)')
    axs[0, 0].set_ylim([-100, 70])
    axs[0, 0].set(ylabel='Vm (mV)', xlabel='Time (ms)', title='Presynaptic PC neuron')

    axs[0, 1].plot(PC_M.t/ms, PC_M.v[1]*1000, label= 'PC SM (post)')
    axs[0, 1].set(ylabel='Vm (mV)', xlabel='Time (ms)', title='Postsynaptic PC neuron')
    axs[0, 1].set_ylim([-100, 70])
    axs[0, 1].set_title('Postsynaptic PC neuron')

    axs[1, 0].plot(IN_M.t/ms, IN_M.v[0]*1000, label= 'IN SM')
    axs[1, 0].set(ylabel='Vm (mV)', xlabel='Time (ms)', title='In Neuron')
    axs[1, 0].set_ylim([-100, 70])
    axs[1, 0].set_title('In Neuron')

    axs[1, 1].plot(PC_M.t/ms, PC_M.v[1]*1000 - PC_M.v[0]*1000, label= 'PC Spike')
    axs[1, 1].set_ylim([-100, 70])
    axs[1, 1].set(ylabel='VM (mV)', xlabel='Time (ms)', title='Difference Pre- and Postsynaptic PC neuron')

    fig.suptitle('State Monitors, Scale: '+ str(scale) + ' Weight: ' + str(weight))

    plt.show()

#%%
