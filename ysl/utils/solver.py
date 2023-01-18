import numpy as np
from numba import njit, prange



def check_vel(vel):
    return not np.any(np.abs(vel)>1e-8)


## numba
@njit(fastmath=False)
def acc_numba(adjm, power, phase, dphase, gamma, K, mass):
    acc_vec = np.zeros((len(adjm), ))
    for n in range(len(acc_vec)):
        acc_vec[n] += power[n] - gamma[n] * dphase[n]
        for nn in range(n+1, len(acc_vec)):
            if adjm[n][nn]!=0:
                temp = adjm[n][nn] * K * np.sin(phase[nn] - phase[n])
                acc_vec[n] += temp
                acc_vec[nn] -= temp
    for n in range(len(acc_vec)):
        acc_vec[n] /= mass[n]
    return acc_vec

@njit(fastmath=False)
def acc_loss_numba(adjm, power, phase, dphase, gamma, K, mass, alpha):
    acc_vec = np.zeros((len(adjm), ))
    for n in range(len(acc_vec)):
        acc_vec[n] += power[n] - gamma[n] * dphase[n]
        for nn in range(len(acc_vec)):
            if adjm[n][nn]!=0:
                temp = adjm[n][nn] * K * (np.sin(alpha[n][nn] + phase[nn] - phase[n]) - np.sin(alpha[n][nn]))
                acc_vec[n] += temp
    for n in range(len(acc_vec)):
        acc_vec[n] /= mass[n]
    return acc_vec

@njit
def vel_numba(dphase):
    return dphase


## numba hybrid
@njit(parallel=False)
def acc_numba_hybrid(adjm, power, phase, dphase, gamma, K, mass, activity):
    acc_vec = np.zeros((len(adjm), ))
    for n in range(len(acc_vec)):
        if activity[n]: # if n is still active
            acc_vec[n] += power[n] - gamma[n] * dphase[n]
            for nn in range(n+1, len(acc_vec)):
                if adjm[n][nn]!=0:
                    temp = adjm[n][nn] * K * np.sin(phase[nn] - phase[n])
                    acc_vec[n] += temp
                    if activity[nn]:
                        acc_vec[nn] -= temp
        else:
            acc_vec[n] = 0
    for n in range(len(acc_vec)):
        if activity[n]:
            acc_vec[n] /= mass[n]
    return acc_vec

@njit(parallel=False)
def vel_numba_hybrid(adjm, power, phase, dphase, gamma, K, activity):
    vel_vec = np.zeros((len(adjm), ))
    for n in range(len(adjm)):
        if activity[n]: # if n is still active
            vel_vec[n] = dphase[n]
        else:
            vel_vec[n] += power[n]
            for nn in range(n+1, len(vel_vec)):
                if adjm[n][nn]!=0:
                    temp = adjm[n][nn] * K * np.sin(phase[nn] - phase[n])
                    vel_vec[n] += temp
                    if not activity[nn]:
                        vel_vec[nn] -= temp
    for n in range(len(vel_vec)):
        if not activity[n]:
            vel_vec[n] /= gamma[n]
    return vel_vec



# numba hybrid
# @njit('float64[:](float64[:,:], float64[:], float64[:], float64[:], float64[:], float64, float64[:])', parallel=False)
@njit(parallel=False)
def acc_loss_numba_hybrid(adjm, power, phase, dphase, gamma, K, mass, activity, alpha):
    acc_vec = np.zeros((len(adjm), ))
    for n in range(len(acc_vec)):
        if activity[n]: # if n is still active
            acc_vec[n] += power[n] - gamma[n] * dphase[n]
            for nn in range(len(acc_vec)):
                if adjm[n][nn]!=0:
                    temp = adjm[n][nn] * K * (np.sin(alpha[n][nn] + phase[nn] - phase[n]) - np.sin(alpha[n][nn]))
                    acc_vec[n] += temp
                    # if activity[nn]:
                    #     acc_vec[nn] -= temp
        else:
            acc_vec[n] = 0
    for n in range(len(acc_vec)):
        if activity[n]:
            acc_vec[n] /= mass[n]
    return acc_vec

# @njit('float64[:](float64[:,:], float64[:], float64[:], float64[:], float64[:], float64, float64[:])', parallel=False)
@njit(parallel=False)
def vel_loss_numba_hybrid(adjm, power, phase, dphase, gamma, K, activity, alpha):
    vel_vec = np.zeros((len(adjm), ))
    for n in range(len(adjm)):
        if activity[n]: # if n is still active
            vel_vec[n] = dphase[n]
        else:
            vel_vec[n] += power[n]
            for nn in range(len(vel_vec)):
                if adjm[n][nn]!=0:
                    temp = adjm[n][nn] * K * (np.sin(alpha[n][nn] + phase[nn] - phase[n]) - np.sin(alpha[n][nn]))
                    vel_vec[n] += temp
                    # if not activity[nn]:
                    #     vel_vec[nn] -= temp
    for n in range(len(vel_vec)):
        if not activity[n]:
            vel_vec[n] /= gamma[n]
    return vel_vec



@njit
def rk2_numba(adjm, power, phase, dphase, gamma, h, K, mass):

    k1 = acc_numba(adjm, power, phase, dphase, gamma, K, mass)
    j1 = vel_numba(dphase)
    temp1 = phase + h * j1
    temp2 = dphase + h * k1
    
    k2 = acc_numba(adjm, power, temp1, temp2, gamma, K, mass)
    j2 = vel_numba(temp2)

    return (k1 + k2) / 2., (j1 + j2) / 2.

@njit(fastmath=False)
def rk4_numba(adjm, power, phase, dphase, gamma, h, K, mass):

    k1 = acc_numba(adjm, power, phase, dphase, gamma, K, mass)
    j1 = vel_numba(dphase)
    temp1 = phase + 0.5 * h * j1
    temp2 = dphase + 0.5 * h * k1

    k2 = acc_numba(adjm, power, temp1, temp2, gamma, K, mass)
    j2 = vel_numba(temp2)
    temp1 = phase + 0.5 * h * j2
    temp2 = dphase + 0.5 * h * k2

    k3 = acc_numba(adjm, power, temp1, temp2, gamma, K, mass)
    j3 = vel_numba(temp2)
    temp1 = phase + h * j3
    temp2 = dphase + h * k3

    k4 = acc_numba(adjm, power, temp1, temp2, gamma, K, mass)
    j4 = vel_numba(temp2)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6., (j1 + 2 * j2 + 2 * j3 + j4) / 6.


@njit(fastmath=False)
def rk4_loss_numba(adjm, power, phase, dphase, gamma, h, K, mass, alpha):

    k1 = acc_loss_numba(adjm, power, phase, dphase, gamma, K, mass, alpha)
    j1 = vel_numba(dphase)
    temp1 = phase + 0.5 * h * j1
    temp2 = dphase + 0.5 * h * k1

    k2 = acc_loss_numba(adjm, power, temp1, temp2, gamma, K, mass, alpha)
    j2 = vel_numba(temp2)
    temp1 = phase + 0.5 * h * j2
    temp2 = dphase + 0.5 * h * k2

    k3 = acc_loss_numba(adjm, power, temp1, temp2, gamma, K, mass, alpha)
    j3 = vel_numba(temp2)
    temp1 = phase + h * j3
    temp2 = dphase + h * k3

    k4 = acc_loss_numba(adjm, power, temp1, temp2, gamma, K, mass, alpha)
    j4 = vel_numba(temp2)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6., (j1 + 2 * j2 + 2 * j3 + j4) / 6.



@njit(parallel=False)
def rk4_numba_hybrid(adjm, power, phase, dphase, gamma, h, K, mass, activity):

    k1 = acc_numba_hybrid(adjm, power, phase, dphase, gamma, K, mass, activity)
    j1 = vel_numba_hybrid(adjm, power, phase, dphase, gamma, K, activity)
    temp1 = phase + 0.5 * h * j1
    temp2 = dphase + 0.5 * h * k1

    k2 = acc_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity)
    j2 = vel_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity)
    temp1 = phase + 0.5 * h * j2
    temp2 = dphase + 0.5 * h * k2

    k3 = acc_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity)
    j3 = vel_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity)
    temp1 = phase + h * j3
    temp2 = dphase + h * k3

    k4 = acc_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity)
    j4 = vel_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6., (j1 + 2 * j2 + 2 * j3 + j4) / 6.



@njit(parallel=False)
def rk4_loss_numba_hybrid(adjm, power, phase, dphase, gamma, h, K, mass, activity, alpha):

    k1 = acc_loss_numba_hybrid(adjm, power, phase, dphase, gamma, K, mass, activity, alpha)
    j1 = vel_loss_numba_hybrid(adjm, power, phase, dphase, gamma, K, activity, alpha)
    temp1 = phase + 0.5 * h * j1
    temp2 = dphase + 0.5 * h * k1

    k2 = acc_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity, alpha)
    j2 = vel_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity, alpha)
    temp1 = phase + 0.5 * h * j2
    temp2 = dphase + 0.5 * h * k2

    k3 = acc_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity, alpha)
    j3 = vel_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity, alpha)
    temp1 = phase + h * j3
    temp2 = dphase + h * k3

    k4 = acc_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, mass, activity, alpha)
    j4 = vel_loss_numba_hybrid(adjm, power, temp1, temp2, gamma, K, activity, alpha)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6., (j1 + 2 * j2 + 2 * j3 + j4) / 6.


# for power dispatch
def rebal(power, amp, activity, failed):
    power_sum = sum(power[failed])
    amp_sum = sum(amp)
    assert amp_sum!=0
    power[activity] += power_sum*amp/amp_sum
    power[failed] = 0

def rebal_pm(power, mass, amp, activity, failed):
    power_sum = sum(power[failed])
    # mass_sum = sum(mass[failed])
    amp_sum = sum(amp)
    assert amp_sum!=0
    power[activity] += power_sum*amp/amp_sum
    # mass[activity] += mass_sum*amp/amp_sum
    power[failed] = 0
    mass[failed] = 0

def rebal_partial(power, mass, amp, act_gen, failed, p_fail):
    power_failed = [p_fail if abs(p) > p_fail else p for p in power[failed]]
    power_sum = sum(power_failed)
    # mass_sum = sum(mass[failed])
    amp_sum = sum(amp)
    assert amp_sum!=0
    power[act_gen] += power_sum*amp/amp_sum
    # mass[act_gen] += mass_sum*amp/amp_sum
    for n in range(len(power)):
        if failed[n]:
            if abs(power[n]) > p_fail:
                power[n] = np.sign(power[n]) * (abs(power[n]) - p_fail)
            else:
                power[n] = 0
                mass[n] = 0


# for power dispatch
def rebal2(power, amp, activity, failed):
    power_sum = sum(power[failed])
    amp_sum = sum(amp[activity])
    assert amp_sum!=0
    power[activity] += power_sum*amp[activity]/amp_sum
    power[failed] = 0


@njit(parallel=False)
def rebal_hp(power, amp, activity, failed, hp, mass):
    failed_1 = failed*(hp==1) # failed nodes having hp 1
    failed_2 = failed*(hp==2) # failed nodes having hp 2
    power_sum = sum(power[failed_1])
    power_sum += sum(power[failed_2])/2
    amp_sum = sum(amp)
    # assert amp_sum!=0
    power[activity] += power_sum*amp/amp_sum

    power[failed_1] = 0
    mass[failed_1] = 0
    power[failed_2] /= 2
    mass[failed_2] /= 2
    
    hp[failed] -= 1


@njit(parallel=False)
def cutting_edge(adjm, failed):
    adjm[failed, :] *= 0
    adjm[:, failed] *= 0
