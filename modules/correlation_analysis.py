import numpy as np
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt

import scipy.io
import scipy.signal
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf, ccf

import pandas as pd


def load_data_and_idxs(subject_number, n, dir = "../processed_data/"):
    """
    Load data and idxs for subject number *subject_number* and silence number *n*.

    Parameters
    ----------
    subject_number : int
        Subject number.
    n : int
        Silence number.
        Can be either 1 or 2.

    Returns
    -------
    data : numpy.ndarray
        EEG signals for the given subject and given recording (silence 1 or 2)
    """
    path = dir + f"silence_{n}_subj_{subject_number}"

    return np.load(path + ".npy"), np.load(path + "_idxs.npy")

def filter_data(data, band):
    """
    Filter data with a bandpass filter.

    Parameters
    ----------
    data : numpy.ndarray
        EEG signals.
    band : tuple
        Frequency band of the bandpass filter, in Hz.

    Returns
    -------
    numpy.ndarray
        Filtered data.
    """

    fs = 500
    nq = fs//2
    order = int(fs//band[0]*2)
    bb = scipy.signal.firwin(order, band, pass_zero = False, fs = fs)
    aa = 1
    return scipy.signal.filtfilt(bb, aa, data, axis = 1, padlen = 150)

def load_and_match_data(subject_number):
    """
    Load EEG data for subject *subject_number* and match the channels for silence 1
    and silence 2. Only the channels that are present in both silences are returned.

    Parameters
    ----------
    subject_number : int
        Subject number.

    Returns
    -------
    new_data1 : numpy.ndarray
        EEG signals for silence 1, with only the channels that are present in both
        silences.
    new_data2 : numpy.ndarray
        EEG signals for silence 2, with only the channels that are present in both
        silences.
    shared_idxs : numpy.ndarray
        Indices of the channels that are present in both silences.
    """
    data1, idxs1 = load_data_and_idxs(subject_number, 1)
    data2, idxs2 = load_data_and_idxs(subject_number, 2)
        
    shared_idxs = np.array(list(set(idxs1).intersection(set(idxs2))))
    new_data1 = np.zeros((shared_idxs.size, data1.shape[1]))
    new_data2 = np.zeros((shared_idxs.size, data2.shape[1])) 
    
    c = 0
    for i in range(10):
        if (i in idxs1) and (i in idxs2):
            new_data1[c] = data1[np.where(idxs1 == i)]
            new_data2[c] = data2[np.where(idxs2 == i)]
            c += 1
            
    return new_data1, new_data2, shared_idxs

def build_dataframe(tcorr, cutoff = 5):
    """
    Build a dataframe with the subject number, the ratio of the autocorrelation
    times between silence 1 and silence 2, and the channel number for each channel
    that has a ratio below *cutoff*.

    Parameters
    ----------
    tcorr : list
        Each element is a list containing the subject number, the channel number, and
        the autocorrelation times in silence 1 and silence 2.
    cutoff : float, optional
        Ratio cutoff. The default is 5.
        It is assumed that a large ratio means that the autocorrelation time was not
        correctly estimated.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the subject number, the ratio of the autocorrelation times,
        and the channel number.
    """
    df = pd.DataFrame(columns = ['subject', 'ratio', 'ch'])
    for i in range(len(tcorr)):
        data = tcorr[i]
        ratio = (data[2][1] - data[2][0])/data[2][0]
        for j, ch in enumerate(data[1]):
            if ratio[j] < cutoff:
                df = df.append({'subject': int(data[0]),
                                'ratio': ratio[j],
                                'ch': ch},
                                ignore_index=True) # type: ignore
    return df

def find_autocorr(data, nlag = 20000):
    """
    Computer the autocorrelation of the EEG signals for each channel.

    Parameters
    ----------
    data : numpy.ndarray
        EEG signals, of shape (n_channels, n_samples).
    nlag : int, optional
        Number of lags to compute the autocorrelation. The default is 20000.

    Returns
    -------
    autocorr : numpy.ndarray
        Autocorrelation of the EEG signals for each channel.
    """
    autocorr = np.zeros((data.shape[0], nlag+1))
    
    for idx, ddata in enumerate(data):
        autocorr[idx] = acf(ddata, nlags = nlag, adjusted=True)
        
    return autocorr

def monoExp(t, tau, a):
    """
    Exponetial function, starting at 1 in t = 0.

    Parameters
    ----------
    t : numpy.ndarray
        Time.
    tau : float
        Autocorrelation time.
    a : float
        Scale factor.

    Returns
    -------
    numpy.ndarray
        Exponential function.
    """
    return a*(np.exp(-t/tau) - 1) + 1

def fit_envelope_autocorr(autocorr, delta_t = 0.002):
    """
    Fit the exponential envelope of the autocorrelation function.

    Parameters
    ----------
    autocorr : numpy.ndarray
        Autocorrelation functions. Each row is the autocorrelation
        for a given channel.
    delta_t : float, optional
        Sampling time of the experimental signals.
        The default is 0.002s.

    Returns
    -------
    tcorr : numpy.ndarray
        Autocorrelation times for each channel.
    """
    tcorr = np.zeros(autocorr.shape[0])
    
    for idx, data in enumerate(autocorr):
        maxima = argrelextrema(data, np.greater, order = 100)[0]
        minima = argrelextrema(data, np.less, order = 100)[0]
        
        x_to_fit = np.concatenate([np.zeros(1), maxima])
        y_to_fit = np.concatenate([np.ones(1), data[maxima]])
        params_max, _ = curve_fit(monoExp, x_to_fit, y_to_fit, (250, 5))
        
        x_to_fit = np.concatenate([np.zeros(1), minima])
        y_to_fit = np.concatenate([-np.ones(1), data[minima]])
        params_min, _ = curve_fit(monoExp, x_to_fit, y_to_fit, (250, 5))
        
        tcorr[idx] = (params_max[0] + params_min[0])/2*delta_t
            
    return tcorr

def find_ac_times(subjs, band, verbose = False, nlag = 2000):
    """
    Computer the autocorrelation times in a given band, for a list of subjects
    and for each channel.

    Parameters
    ----------
    subjs : list
        List of subject numbers.
    band : tuple
        Frequency band to filter the data, in Hz.
    verbose : bool, optional
        If True, print the number of rejected subjects.
        A subject is rejected if it has less than 5 channels.
        The default is False.
    nlag : int, optional
        Number of lags to compute the autocorrelation. The default is 2000.

    Returns
    -------
    tcorr : list
        Each element is a list containing the subject number, the channel number, and
        the autocorrelation times in silence 1 and silence 2.
    """
    tcorr = []
    rejected = 0
    for i, subj in enumerate(subjs):
        data1, data2, ch = load_and_match_data(subj)
        n_ch = len(ch)
        tcorr_subj = np.zeros((2, n_ch))
        
        if n_ch >= 5:
            for j, data in enumerate((data1, data2)):
                data = filter_data(data, band)
                autocorr = find_autocorr(data, nlag = nlag)
                tcorr_subj[j] = fit_envelope_autocorr(autocorr)
            tcorr.append([subj, ch, tcorr_subj])
        else:
            if verbose: print(f"Subject {subj} rejected ({n_ch})")
            rejected += 1
            
    if verbose: print(rejected)
    return tcorr

import scipy.stats

def plot_band_results(tcorr, title, cutoff = 5):
    """
    Plot the ratio of the autocorrelation times in silence 1 and silence 2,
    for all subjects and channels.

    Parameters
    ----------
    tcorr : list
        Each element is a list containing the subject number, the channel number, and
        the autocorrelation times in silence 1 and silence 2.
    title : str
        Title of the plot.
    cutoff : float, optional
        Cutoff value for the ratio of the autocorrelation times.
        The default is 5.
    """
    ratios = [(data[2][1] - data[2][0])/data[2][0] for data in tcorr]
    accepted_subj = [data[0] for data in tcorr]
    
    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (25,5))

    for i, subj in enumerate(accepted_subj):
        ax[0].scatter([subj]*len(ratios[i]), ratios[i], c = "gray", s = 30)

        ax[1].errorbar(subj, ratios[i].mean(axis = -1),
                       yerr = ratios[i].std(axis = -1), ls = "none", fmt = "o", c = "gray", capsize = 4, elinewidth = 2)

    ax[0].axhline(0, ls = "--", c = "k")
    ax[0].set_xlabel('Subject ID')
    ax[0].set_ylabel(r"$(\tau_2 - \tau_1)/ \tau_1$")

    ax[1].axhline(0, ls = "--", c = "k")
    ax[1].set_xlabel('Subject ID')
    ax[1].set_ylabel(r"$(\tau_2 - \tau_1)/ \tau_1$ (channels average)")

    data = np.concatenate(ratios)
    data = data[data < cutoff]
    _, bins, _ = ax[2].hist(data, bins = 45, density = True, color = "darkred", alpha = 0.5)
    mu, sigma = scipy.stats.norm.fit(data)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    ax[2].plot(bins, best_fit_line, ls = "--", c = "k")
    ax[2].set_ylabel(r"$p[(\tau_2 - \tau_1)/ \tau_1]$")
    ax[2].set_xlabel(r"$(\tau_2 - \tau_1)/ \tau_1$")
    ax[1].set_title(title, size = 30, pad = 10)
    plt.show()