import numpy as np

def generate_batch(tc, t, n):
    signals = [generate(tc,t) for k in range(n)]
    return signals

def generate(tc, t):
    """Generates a signal for test-case tc by
    using t as the timeline"""
    omega = np.random.uniform(
        2.0*np.pi*59.9,
        2.0*np.pi*60.1) # 60 Hz centered
    omega = 2.0*np.pi*60.0
    phi = np.random.uniform(0.0, 2.0*np.pi) # Random phi
    signal = np.cos(omega*t+phi) # Ideal power signal
    if tc==1:
        # Sane (micro sag-swell)
        tstart = np.random.uniform(0.0,0.1)
        tend = tstart+0.1
        factor = np.random.uniform(0.95,1.05)
        signal[(t>=tstart)&(t<tend)] *= factor
    elif tc==2:
        # Sag
        tstart = np.random.uniform(0.0,0.1)
        tend = tstart+0.1
        factor = np.random.uniform(0.1,0.9)
        signal[(t>=tstart)&(t<tend)] *= factor
    elif tc==3:
        # Swell
        tstart = np.random.uniform(0.0,0.1)
        tend = tstart+0.1
        factor = np.random.uniform(1.1, 1.5)
        signal[(t>=tstart)&(t<tend)] *= factor
    elif tc==4:
        # Interrupt
        tstart = np.random.uniform(0.0,0.1)
        tend = tstart+0.1
        factor = np.random.uniform(0.0, 0.1)
        signal[(t>=tstart)&(t<tend)] *= factor
    elif tc==5:
        # Harmonic
        hfactor = np.random.uniform(0.15, 0.30)
        hphi = np.random.uniform(0.0,2.0*np.pi)
        signal += hfactor*np.cos(omega*3*t+hphi)
    elif tc==6:
        # Flicker
        fomega = np.random.uniform(
            2.0 * np.pi * 4.0,
            2.0 * np.pi * 10.0)
        fphi = np.random.uniform(0.0,2.0*np.pi)
        ffactor = np.random.uniform(0.05, 0.1)
        signal = (1+ffactor*np.cos(fomega*t+fphi))* \
                 np.cos(omega * t + phi)
    elif tc == 7:
        # Harmonic and interharmonic
        hfactor = np.random.uniform(0.15, 0.30)
        hphi = np.random.uniform(0.0,2.0*np.pi)
        signal += hfactor*np.cos(omega*3*t+hphi)
        ihfactor = np.random.uniform(0.10, 0.15)
        ihphi = np.random.uniform(0.0,2.0*np.pi)
        signal += ihfactor*np.cos(omega*6.5*t+ihphi)
    elif tc == 8:
        # Transient
        trc = np.random.uniform(0.5, 0.7)
        trtau = np.random.uniform(0.003, 0.004)
        tromega = 2.0*np.pi*1000
        trphi = np.random.uniform(0.0, 2.0*np.pi)
        tstart = np.random.uniform(0.05, 0.15)
        tm = t-tstart
        signal += trc*(tm>0)*np.exp(-tm/trtau)* \
            np.cos(tromega*tm+trphi)
        # print({'c':trc, 'tau':trtau,
        #       'omega':tromega,'phi':trphi})
    return signal

