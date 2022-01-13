"""This file does the validation by using
real power signals from IEEE and Valladolid
hospital. It shows the estimated signal and the
raw signal by GA-PSO and SANN-PQ."""

import pandas as pd
import numpy as np
from sannpq import estimate as sannpq
from gapsopq import estimate as gapsopq
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks.benchmark import Benchmark
import utils
import matplotlib.pyplot as plt


class freqphase_optfunc(Benchmark):
    def __init__(self, search_ranges, t, signal):
        self.Lower = 0
        self.Upper = 1

        self.t = t
        self.signal = signal
        self.search_ranges = search_ranges

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            [omega, phi] = utils.translate_ranges(sol, self.search_ranges)
            f = np.cos(omega*self.t+phi)
            diff = self.signal - f
            diff = [np.abs(x) for x in diff]
            val = np.sum(diff)
            return val

        return evaluate


search_ranges_freqphase = [
    [2.0*np.pi*45, 2.0*np.pi*65],  # Omega
    [0.0, 2*np.pi]  # Phi
]


def validate(t, signal, splot, dict, signalid):
    # Estimates the signal by both methodologies
    estimations = {'gapso': [], 'sannpq': []}
    # Extend signal for filtering purposes
    # Estimate frequency and phase by using DE
    task = StoppingTask(
        D=2, nGEN=40,
        optType=OptimizationType.MINIMIZATION,
        benchmark=freqphase_optfunc(
            search_ranges_freqphase,
            t,
            signal))
    algorithm = ParticleSwarmAlgorithm(
        # C1=2, C2=2,
        NP=50
    )
    best = algorithm.run(task=task)
    [omega, phi] = utils.translate_ranges(
        best[0],
        search_ranges_freqphase)
    print('Frequency found: ', omega/2.0/np.pi, '\nPhase found: ', phi)
    # plt.figure()
    # plt.plot(t, signal, color='black')
    # plt.plot(t, np.cos(omega*t+phi), color='blue')
    # plt.show()
    # Once frequency and phase are calculated then obtain sampling freq
    ran_max = max(t)
    ran_min = min(t)
    ran = max(t)-min(t)
    sps = int(np.round(len(t)/ran))
    print('Sampling frequency: ', sps)
    # Extend the signal, adding 1 second to the left and to the right
    pre_t = np.arange(-1.0, 0.0, step=1/sps)
    pos_t = np.arange(max(t)+1/sps, max(t)+1/sps+1.0, step=1/sps)
    t = np.concatenate([pre_t, t, pos_t])
    mask = np.where((t >= ran_min) & (t < ran_max))
    dif = 0.01
    mask_res = np.where((t >= ran_min+dif) & (t < ran_max-dif))
    signal = np.concatenate([
        np.cos(omega*pre_t+phi),
        signal,
        np.cos(omega*pos_t+phi)])
    # plt.figure()
    # plt.plot(t, signal, color='black')
    # plt.plot(t[maskidxs], signal[maskidxs], color='blue')
    # plt.show()
    estimation_gapsopq = gapsopq(
        t, signal,
        sps, omega,
        mask, mask_res, False)
    dict['method'].append('gapso')
    dict['signalid'].append(signalid)
    dict['rsse'].append(estimation_gapsopq['rsse'])
    dict['mad'].append(estimation_gapsopq['mad'])
    dict['duration'].append(estimation_gapsopq['duration'])
    estimation_sannpq = sannpq(
        t, signal,
        sps, omega,
        mask, mask_res, False, True)
    dict['method'].append('sannpq')
    dict['signalid'].append(signalid)
    dict['rsse'].append(estimation_sannpq['rsse'])
    dict['mad'].append(estimation_sannpq['mad'])
    dict['duration'].append(estimation_sannpq['duration'])
    l1 = splot.plot(t[mask_res], signal[mask_res],
                    label='Raw', color='black')
    l2 = splot.plot(t[mask_res], estimation_gapsopq['estimated'][mask_res],
                    label='GA-PSO', color='red')
    l3 = splot.plot(t[mask_res], estimation_sannpq['estimated'][mask_res],
                    label='SANN-PQ', color='blue')
    splot.grid()
    splot.set_title('U'+str(signalid))
    df = pd.DataFrame(data={
        'time':t[mask_res],
        'raw':signal[mask_res],
        'gapsopq':estimation_gapsopq['estimated'][mask_res],
        'sannpq':estimation_sannpq['estimated'][mask_res]})
    df.to_csv('plot_data_{}.csv'.format(str(signalid)))
    return [l1, l2, l3], dict


if __name__ == '__main__':
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Times New Roman',
        'font.weight': 'normal',
        'lines.linewidth': 1})
    f, axarr = plt.subplots(
        nrows=3,
        ncols=2)

    dict = {
        "method": [],
        "signalid": [],
        "rsse": [],
        "mad": [],
        "duration": []
    }

    # Signal 1
    df = pd.read_csv('./realsignals/example 01.csv')
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 1].values
    # Scale the signal to pu
    # Use 230*sqrt(3) as peak amplitude
    signal /= (230*np.sqrt(3))
    lines, dict = validate(t, signal, axarr[0][0], dict, 1)

    # Signal 2
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 2].values
    # Scale the signal to pu
    # Use 230*sqrt(3) as peak amplitude
    signal /= (230*np.sqrt(3))
    lines, dict = validate(t, signal, axarr[0][1], dict, 2)

    # Signal 3
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 3].values
    # Scale the signal to pu
    # Use 230*sqrt(3) as peak amplitude
    signal /= (230*np.sqrt(3))
    lines, dict = validate(t, signal, axarr[1][0], dict, 3)

    # Signal 4
    df = pd.read_csv('./realsignals/example 02.csv')
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 1].values
    # Scale the signal to pu
    # Use 10000 as peak amplitude
    signal /= 10000
    lines, dict = validate(t, signal, axarr[1][1], dict, 4)

    # Signal 5
    df = pd.read_csv('./realsignals/example 06.csv')
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 1].values
    # Scale the signal to pu
    # Use 10000 as peak amplitude
    signal /= 10000
    lines, dict = validate(t, signal, axarr[2][0], dict, 5)

    # Signal 6
    df = pd.read_csv('./realsignals/hospital signal.csv')
    t = df.iloc[:, 0].values
    signal = df.iloc[:, 1].values
    # Scale the signal to pu
    # Use 1 as peak amplitude
    lines, dict = validate(t, signal, axarr[2][1], dict, 6)

    df = pd.DataFrame(data=dict)
    for ax in axarr.flat:
        # ax.set_xlim(0.0,0.2)
        ax.set(xlabel='Time(s)', ylabel='V (pu)')
    for ax in axarr.flat:
        ax.label_outer()
    f.subplots_adjust(
        left=0.1,
        right=0.95,
        wspace=0.125,
        hspace=0.7,
        top=0.80)
    f.legend(lines,
             labels=['Raw', 'GA-PSO', 'SANN-PQ'],
             loc="upper center",
             borderaxespad=0.1,
             title='Signals', ncol=3)
    f.set_size_inches(7.0, 3.5)
    f.set_dpi(300)
    f.savefig('real_validation.png', dpi=300)
    f.show()
    print(df)
