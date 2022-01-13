"""This file creates a plot from the CSV
exported from the realvalidation.py script.
It creates three plots indeed."""

import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt

def print_plots(idx):
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Liberation Serif',
        'font.weight': 'normal',
        'lines.linewidth': 1})
    f, axarr = plt.subplots(
        nrows=3,
        ncols=2)
    df = pd.read_csv('plot_data_1.csv')
    splot = axarr[0][0]
    splot.grid()
    splot.set_title('U1')
    l1 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l12 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l12 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
    df = pd.read_csv('plot_data_2.csv')
    splot = axarr[0][1]
    splot.grid()
    splot.set_title('U2')
    l2 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l22 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l22 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
    df = pd.read_csv('plot_data_3.csv')
    splot = axarr[1][0]
    splot.grid()
    splot.set_title('U3')
    l3 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l32 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l32 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
    df = pd.read_csv('plot_data_4.csv')
    splot = axarr[1][1]
    splot.grid()
    splot.set_title('U4')
    l4 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l42 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l42 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
    df = pd.read_csv('plot_data_5.csv')
    splot = axarr[2][0]
    splot.grid()
    splot.set_title('U5')
    l5 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l52 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l52 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
    df = pd.read_csv('plot_data_6.csv')
    splot = axarr[2][1]
    splot.grid()
    splot.set_title('U6')
    l6 = splot.plot(df['time'], df['raw'],
                    label='Raw', color='black')
    if idx == 1:
        l62 = splot.plot(df['time'], df['gapsopq'],
                label='GA-PSO', color='red')
    elif idx == 2:
        l62 = splot.plot(df['time'], df['sannpq'],
                label='SANN-PQ', color='red')
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
    lines = [l1,l2,l3,l4,l5,l6]
    legends = ['Raw']
    if idx > 0:
        lines = [l1,l12,l2,l22,l3,l32,l4,l42,l5,l52,l6,l62]
    if idx == 1:
        legends += ['GA-PSO']
    elif idx == 2:
        legends += ['SANN-PQ']
    f.legend(lines,
             labels=legends,
             loc="upper center",
             borderaxespad=0.1,
             title='Signals', ncol=3)
    f.set_size_inches(7.0, 3.5)
    f.set_dpi(300)
    f.savefig('signals_{}.png'.format(idx), dpi=300)
    f.show()
    

if __name__ == '__main__':
    # Raw plot
    print_plots(0)
    print_plots(1)
    print_plots(2)

