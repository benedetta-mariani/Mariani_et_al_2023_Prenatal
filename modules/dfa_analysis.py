### COMPUTE DFA ANALYSIS BETWEEN SILENCE 1 AND SILCENCE 2

import os
import numpy as np
import scipy.signal as ss
import scipy.io
from cleaned_functions import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy.signal import welch
import pandas as pd
import mne

### Parameters
hh = 14
ch_names = ['F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8']
ch_names = np.array(ch_names)
sfreq = 500

overlap =0.5
bands = [[1,3],[4,8],[8,13],[14,30],[31,60]]
labels = ['band-1', 'band-2', 'band-3', 'band-4', 'band-5']
subjs  = [5,6,9,11,12,13,16,17,18,19,21,23,24,25,30,33,34,35,37,39,40,45,47,49,52,53,56,57,58,60,62,63,65,66,67,68,69,71,14,20,22,27,28,29,31,41,46,64,70]


info = mne.create_info(ch_names = list(ch_names),
                       ch_types = 'eeg',
                       sfreq = sfreq)

### Functions for preprocessing
def find_bad_channels(data, MAX_PEAK=200, MAX_PEAK_SIGMA=15, MAX_DIST_PS=4.5e-3, OFFSET=30, fmin=1, fmax=100, 
                      verbose = True):
    '''
	Find bad channels based on the maximum peak and the distance from the mean of the PSD.
	Inputs:
		data: data to be analyzed
		MAX_PEAK: maximum peak allowed (in uV)
		MAX_PEAK_SIGMA: maximum peak allowed (in number of standard deviations)
		MAX_DIST_PS: maximum distance from the mean of the PSD allowed
		OFFSET: time (in seconds) to be ignored at the beginning and at the end of the signal
		fmin: minimum frequency for the PSD
		fmax: maximum frequency for the PSD
	'''
    idx_down, idx_up = 0, data.shape[1]
    ss = np.std(data)
    
	# Compute PSD
    psds, freqs = mne.time_frequency.psd_welch(mne.io.RawArray(data/1e6, info, verbose=False), fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024/2, verbose=False)
    psds = np.log10(psds)
    
	# Compute distance from mean
    dist = [((tmp-psds.mean(axis=0))**2).sum() for tmp in psds]
    dist = np.array(dist) / (psds**2).sum() * 1e2
    
	# Find bad channels
    rej = []
    for i, ch in enumerate(ch_names):
        delta = np.abs(data[i]).max()
    
        str_rej = str()

        if delta>MAX_PEAK or dist[i]>MAX_DIST_PS:
            rej.append(ch)
            str_rej = '-> rejected'
            str_rej += ' ('
            if delta>MAX_PEAK:
                str_rej += ' peak'
                idx = np.where(np.abs(data[i])>MAX_PEAK)[0]
            
            if dist[i]>MAX_DIST_PS:
                str_rej += ' ps'
            str_rej += ')'
            
        if verbose: print(f'* {ch}: max peak = {np.round(delta, 2)} mV ({np.round(delta/ss, 2)} std); dist ps = {np.round(dist[i],3)} '+str_rej)
        
        if delta>MAX_PEAK:
            if np.max(idx)/sfreq<OFFSET:
                idx_down = np.max([np.max(idx),idx_down])
                if verbose: print(f'[*] WARNING: Peak only in the initial part - time: {np.max(idx)/sfreq} s')
            if np.min(idx)/sfreq>180-OFFSET:
                idx_up = np.min([np.min(idx),idx_up])
                if verbose: print(f'[*] WARNING: Peak only in the last part - time: {np.min(idx)/sfreq} s')
    if verbose: print('\nBad channels:', rej, '\n')
    return rej, idx_down, idx_up

### Functions for DFA
for i, subj in enumerate(subjs):
	dfa_sil_params = [[] for r in range(len(bands)-1)]
	dfa_sil2_params = [[] for r in range(len(bands)-1)]
	print(f'\n\n########## SUBJ {subj} ({i+1}/{len(subjs)})##########')

	# Load data
	s1 = loadmat(f'BB{subj}_Filtered (1-100)/Silence (500)/BB{subj} silence1 (continuous).mat')
	s2 = loadmat(f'BB{subj}_Filtered (1-100)/Silence (500)/BB{subj} silence2 (continuous).mat')

	# Preprocessing silence 1
	data = s1['eeg_rest'].astype('float')
	rej, idx_down, idx_up = find_bad_channels(data, verbose = False)
	if idx_down > 0 or idx_up < data.shape[1]:
		if idx_down > 0:
			idx_down +=1
		if idx_up < data.shape[1]:
			idx_up -= 1
		data = data[:,idx_down:idx_up]
		rej, idx_down, idx_up = find_bad_channels(data, verbose = False)
	idx_accepted = np.sort([np.where(ch_names == i)[0][0] for i in list(set(ch_names) - set(rej))])
	data = data[list(idx_accepted), idx_down:idx_up]
	nchan = data.shape[0]
	
	# Preprocessing silence 2
	data2 = s2['eeg_rest'].astype('float')
	rej, idx_down, idx_up = find_bad_channels(data2, verbose = False)
	if idx_down > 0 or idx_up < data2.shape[1]:
		if idx_down > 0:
			idx_down +=1
		if idx_up < data2.shape[1]:
			idx_up -= 1
		data2 = data2[:,idx_down:idx_up]
		rej, idx_down, idx_up = find_bad_channels(data2, verbose = False)
	idx_accepted = np.sort([np.where(ch_names == i)[0][0] for i in list(set(ch_names) - set(rej))])
	data2 = data2[list(idx_accepted), idx_down:idx_up]
	nchan2 = data2.shape[0]
	print(nchan, nchan2)

	# DFA analysis
	if nchan >= 5 and nchan2 >=5:
		print('doing')
		for band in range(1,len(bands)):
			nn= int((2/bands[band][0])*500)
			low = bands[band][0]
			high = bands[band][1]
			filt = True
			bb =ss.firwin(nn,[low,high],pass_zero = False, fs = 500)
			aa = 1
			if filt: 
				filtered = ss.filtfilt(bb,aa,data,axis =1, padlen = 500)
			else:
				filtered =data
			for g in range(nchan):
				asil,bsil,esil = dfa(np.abs(ss.hilbert(filtered[g])), scale_lim =(3,hh),overlap =  overlap, det = 1)
				dfa_sil_params[band-1].append(np.array([asil,bsil,esil]))

			if filt: 
				filtered = ss.filtfilt(bb,aa,data2,axis =1, padlen = 500)
			else:
				filtered = data2
			for g in range(nchan2):
				asil,bsil,esil = dfa(np.abs(ss.hilbert(filtered[g])), scale_lim =(3,hh),overlap =  overlap, det = 1)
				dfa_sil2_params[band-1].append(np.array([asil,bsil,esil]))
	try:
		os.chdir('Data')
	except:
		os.mkdir('Data')
		os.chdir('Data')
	if nchan >= 5 and nchan2 >=5:
		print('saving')
		print(np.array(dfa_sil_params).shape, np.array(dfa_sil2_params).shape)
		np.save("sil1" + str(subj) + ".npy", dfa_sil_params)
		np.save("sil2" + str(subj) + ".npy", dfa_sil2_params)
	os.chdir('..')
	