from scipy import signal

def butter_highpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff / (0.5 * fs), btype='high', output='sos')

def butter_lowpass(cutoff, fs, order=4):
    return signal.butter(order, cutoff / (0.5 * fs), btype='low', output='sos')

def notch_filter(freq, fs, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    return b, a
