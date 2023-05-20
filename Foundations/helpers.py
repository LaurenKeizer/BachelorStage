import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import brian2 as b2
from Documentation import parameters as p

def scale_input_theory(input_theory, baseline, scale, dt):
    ''' Scales the input !'''
    baseline = np.ones_like(input_theory, dtype=float)*baseline
    scaled_input = (baseline + input_theory * scale)*b2.uamp
    inject_input = b2.TimedArray(scaled_input, dt=dt*b2.ms)
    return inject_input

def make_spiketrain(spikemon):
    ''' Generates a binary array that spans the whole simulation and
        is 1 when a spike is fired.

        INPUT
        spikemon (array or brian2.SpikeMonitor): Brian2 object or array containing the spikes
        duration (float): simulation time in milliseconds
        dt (float): time step of the simulation is milliseconds

        OUTPUT
        spiketrain (array): binary array that's 1 when a spike occured
    '''
    # Create empty array of the right lenght
    spiketrain = np.zeros((1, int(p.duration/p.dt)), dtype=int)

    # Check the input and get index where a spike occured
    if isinstance(spikemon, b2.SpikeMonitor):
        spikeidx = np.array(spikemon.t/b2.ms/p.dt, dtype=int)
    elif isinstance(spikemon, (np.ndarray, list)):
        spikeidx = spikemon/p.dt
        spikeidx = spikeidx.astype('int')
    else:
        TypeError('Please provide SpikeMonitor or array of spiketimes')

    # Make the spiketrain
    spiketrain[:, spikeidx] = 1

    return spiketrain
#%%

def calc_MI_input(I, theta, x):
    ''' Calculate the mutual information between hidden state (x) and
        generated input train (same size vector) assuming a ideal observer
        that knows ron, roff and theta.

        Note that if dt in ms, then ron and roff in kHz.
        Note that information is calculated in bits. For nats use log instead of log2.
    '''
    # Integrate the posterior Log-likelihood
    L = np.empty(len(x))
    L[0] = np.log(p.ron/p.roff)
    for i in range(len(x) - 1):
        L[i + 1] = L[i] + dLdt_input(L[i], I[i], theta) * p.dt
        if abs(L[i+1]) > 1000:
            print('L diverges weights too large')
            break

    # Calculate the Mutual Information
    Hxx, Hxy, MI = MI_est(L, x)

    return [Hxx, Hxy, MI, L]

#def calc_MI_ideal(spiketrain, x):
##    ''' Calculate the (conditional) entropy, MI, and likelihood.
#    '''
#    qon, qoff, theta, w = calc_qonqoff(spiketrain, x)
#    ## Integrate L
#    I = spiketrain/p.dt
#    L = np.empty(np.shape(x))
#    L[0] = np.log(p.ron/p.roff)

#   for nn in range(len(x) - 1):
#       L[nn+1] = L[nn] + dLdt_spikes(L[nn], I[0][nn], w, theta) * p.dt
#        if abs(L[nn+1]) > 1000:
#            assert StopIteration('L diverges, weights too large')
#
#    # Calculate MI
#    Hxx, Hxy, MI = MI_est(L, x)
#
#    return Hxx, Hxy, MI, L, qon, qoff

def calc_MI_ideal(ron, roff, spiketrain, x, dt):

    ''' Calculate the (conditional) entropy, MI, and likelihood.
    '''
## Calculate qon, qoff, w and theta
    spikesup, spikesdown = reorder_x(x, spiketrain)
    spikesup = np.squeeze(spikesup)
    spikesdown = np.squeeze(spikesdown)

    nspikesup = abs(np.nansum(np.nansum(spikesup)))
    nspikesdown = abs(np.nansum(np.nansum(spikesdown)))
    if nspikesdown == 0:
        print('no down spikes, inventing one')
        nspikesdown = 1

    qon = nspikesup / (sum(x)*dt)
    qoff = nspikesdown / ((len(x) - sum(x))*dt)
    w = np.log(qon/qoff)
    theta = qon-qoff
    # print('w=', w, '; theta=', theta)

    ## Integrate L
    I = spiketrain/dt
    L = np.empty(np.shape(x))
    L[0] = np.log(p.ron/p.roff)

    for nn in range(len(x) - 1):
        L[nn+1] = L[nn] + dLdt_spikes(L[nn], I[0][nn], w, theta) * dt
        if abs(L[nn+1]) > 1000:
            assert StopIteration('L diverges, weights too large')

    # Calculate MI
    Hxx, Hxy, MI = MI_est(L, x)

    return Hxx, Hxy, MI, L, qon, qoff


def dLdt_input(L, I, theta):
    ''' Differential equation calculating the posterior Log-likelihood of the
        hidden state being 1 based on the input history.
        Equation 10.
    '''
    dLdt = p.ron * (1. + np.exp(-L)) - p.roff * (1. + np.exp(L)) + I - theta

    return dLdt

def dLdt_spikes(L, I, w, theta):
    'docstring'
    dLdt = p.ron * (1. + np.exp(-L)) - p.roff * (1. + np.exp(L)) + w*I - theta

    return dLdt

def p_conditional(L):
    ''' Estimates the probability that the hidden state equals 1 given the input history.
        Equation 11.
    '''
    return 1. / (1 + np.exp(-L))


def MI_est(L, x):
    ''' Calculates the mutual information (MI) based on the entorpy of the hidden state (Hxx)
        and the conditional entropy of the hidden state given the input (Hxy).
        Equations 4, 6 & 8
    '''
    Hxx = - np.mean(x) * np.log2(np.mean(x)) - (1 - np.mean(x)) * np.log2(1 - np.mean(x))
    Hxy = - np.mean(x * np.log2(p_conditional(L)) + (1 - x) * np.log2(1 - p_conditional(L)))
    MI = Hxx - Hxy

    return Hxx, Hxy, MI
#%%

def calc_qonqoff(spiketrain, x):
    spikesup, spikesdown = reorder_x(x, spiketrain)

    spikesup = np.squeeze(spikesup)
    spikesdown = np.squeeze(spikesdown)


    nspikesup = abs(np.nansum(np.nansum(spikesup)))
    nspikesdown = abs(np.nansum(np.nansum(spikesdown)))
    if nspikesdown == 0:
        print('no down spikes, inventing one')
        nspikesdown = 1

    qon = nspikesup / (sum(x)*p.dt)
    qoff = nspikesdown / ((len(x) - sum(x))*p.dt)
    w = np.log(qon/qoff)
    theta = qon-qoff

    return qon, qoff, w, theta

def reorder_x(x, ordervecs):
    ''' Reorder the vectors in ordervec (nvec * length) to x=1 (up)
        and x=0 (down). Ordervec is in this case the spike train.
    '''
    ## Check if transposing is necessary
    # Check ordervecs (check the spiketrain)
    number_of_vectors, timesteps = np.shape(ordervecs)
    if number_of_vectors > timesteps:
        s = input('Number of vectors larger than number of timesteps; transpose? (y/n')
        if s == 'y':
            ordervecs = np.transpose(ordervecs)
            number_of_vectors, timesteps = np.shape(ordervecs)

    # Check hiddenstate if needs o be transposed
    number_of_xvectors, _ = np.shape([x])
    if number_of_xvectors != 1:
        x = np.transpose(x)

    #This part of the code looks at where there are jumps. this is only when it is +1 or -1
    xt1 = np.insert(x, 0, x[0])
    xt2 = np.append(x, x[-1])
    xj = xt2 - xt1 # This is equal to xj[n] = x[n] - x[n-1]
    njumpup = len(xj[xj==1])
    njumpdown = len(xj[xj==-1])

    ## Reorder
    if njumpup > 0 and njumpdown > 0:
        firstjump = np.argwhere(abs(xj) == 1).flatten()
        firstjump = firstjump[0]
        revecsup = np.nan * np.empty((number_of_vectors, njumpup+1, round(10*timesteps/njumpup)))
        revecsdown = np.nan * np.empty((number_of_vectors, njumpdown+1, round(10*timesteps/njumpdown)))
        _, _, size3 = np.shape(revecsdown)

        if x[firstjump] == 1:
            up = 1
            down = 0
            revecsup[:, 0, 0] = ordervecs[:, firstjump]
        elif x[firstjump] == 0:
            up = 0
            down = 1
            revecsdown[:, 0, 0] = ordervecs[:, firstjump]
        else:
            raise AssertionError('First jump is not properly defined')

        tt = 1
        tmaxup = 1
        tmaxdown = 1

        for nn in range(firstjump+1, timesteps):
            try:
                jump = x[nn] - x[nn-1]
            except:
                raise AssertionError('size ordervecs not the same as size x')

            # Make jumps
            if jump == 0:
                tt = tt + 1

                if x[nn] == 1:
                    # Up state
                    if tt > tmaxup:
                        tmaxup = tt
                    revecsup[:, up, tt] = ordervecs[:, nn]
                elif x[nn] == 0:
                    # Down state
                    if tt > tmaxdown:
                        tmaxdown = tt
                    revecsdown[:, down, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: x not 0 or 1')

            elif jump == 1:
                #Jump up
                tt = 1
                up = up + 1
                if x[nn] == 1:
                    revecsup[:, up, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: jump up but x not 1')

            elif jump == -1:
                # Jump down
                tt = 1
                down = down + 1
                if x[nn] == 0:
                    revecsdown[:, down, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: jump down but x is not 0')

            else:
                raise AssertionError('Something went wrong: no jump up or down')

            if tt > size3 - 1:
                # tt will run out of Matrix size
                raise IndexError('Choose larger starting Matrix')

        revecsup = revecsup[:, 1:up, 1:tmaxup]
        revecsdown = revecsdown[:, 1:down, 1:tmaxdown]

    else:
        if njumpup < 1:
            print('No jumps up; reordering not possible')
            revecsup = None
            revecsdown = None
        if njumpdown < 1:
            print('No jumps down; reordering not possible')
            revecsup = None
            revecsdown = None

    return revecsup, revecsdown

