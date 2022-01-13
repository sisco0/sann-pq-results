from pqsignal import generate
import numpy as np
import matplotlib.pyplot as plt
from sannpq import estimate
from gapsopq import estimate as estimate2

if __name__ == '__main__':
    sps = 3000
    t = np.arange(-1.0,1.0,1/sps)
    tmask = np.where((t>=0.0) & (t<0.2))
    tmaskres = np.where((t>=0.01) & (t<0.19))
    categories = [1,2,3,4,5,6,7,8]
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Times New Roman',
        'font.weight': 'normal',
        'lines.linewidth': 1})
    f, axarr = plt.subplots(4,2,sharex='col',sharey='row')
    cax = [0,0,1,1,2,2,3,3]
    cay = [0,1,0,1,0,1,0,1]
    #plt.title('Signals')
    for counter, c in enumerate(categories):
        print('Category: ', c)
        signal = generate(c, t)
        result = estimate(
            t, signal,
            sps, 60.0,
            tmask, tmaskres, False)
        # print(result)
        result2 = estimate2(
            t, signal,
            sps, 60.0,
            tmask, tmaskres, False)
        # print(result2)
        #plt.subplot(4,2,counter)
        ax = cax[counter]
        ay = cay[counter]
        l1 = axarr[ax][ay].plot(
                    t[tmaskres],signal[tmaskres],
                    label='Raw',color='black')
        l2 = axarr[ax][ay].plot(
                    t[tmaskres],result2['estimated'][tmaskres],
                    label='GA-PSO',color='red')
        l3 = axarr[ax][ay].plot(
                    t[tmaskres],result['estimated'][tmaskres],
                    label='SANN-PQ',color='blue')
        axarr[ax][ay].set_title('T' + str(c))
        #f.xlabel('Time (s)')
        #f.ylabel('Amplitude (pu)')
        #plt.setp(plt.gca().get_xticklabels(), visible=False)
        #plt.setp(plt.gca().get_yticklabels(), visible=False)
        axarr[ax][ay].grid()

    for ax in axarr.flat:
        ax.set_xlim(0.0,0.2)
        ax.set(xlabel='Time(s)', ylabel='V (pu)')
    for ax in axarr.flat:
        ax.label_outer()
    f.legend([l1, l2, l3],
             labels=['Raw', 'GA-PSO', 'SANN-PQ'],
             loc="upper center",
             borderaxespad=0.1,
             title='Signals', ncol=3)
    f.subplots_adjust(
        left=0.1,
        right=0.95,
        wspace=0.125,
        hspace=0.7,
        top=0.80)
    f.set_size_inches(7.0, 4.0)
    f.set_dpi(300)
    f.savefig('validation_signals.png', dpi=300)
    f.show()
    pass