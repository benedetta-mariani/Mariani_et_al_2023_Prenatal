import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def weight_average(arr,weights,axis):
    return np.average(arr,axis = axis, weights = weights), 1/np.sqrt(np.sum(weights, axis = axis))

def calc_rms(x,scale,overlap,det):
    """
    Root Mean Square in windows with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
      *overlap*: percentage of allowed overlap between windows
      *minscale*: minumum length of the windows considered
      *maxscale*: maximum length of the windows considered
      
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    
    """
    
    scale_ax = np.arange(scale)
    if not (overlap > 0):
        shape = (x.shape[0]//scale, scale)
        ln = (x.shape[0]//scale)*scale
        X = np.reshape(x[:ln],shape)
        rms = np.zeros(X.shape[0])
        coeff = np.polyfit(scale_ax, X.T, det)
        for e in range(len(rms)):
            xfit = np.polyval(coeff[:,e], scale_ax)
            # detrending and computing RMS of each window
            rms[e] = np.sqrt(np.mean((X[e]-xfit)**2))
            
    else:
        rms = []
        i = 0
        while i + scale < len(x):
            xcut = x[i:i + scale]
            coeff = np.polyfit(scale_ax, xcut, det)
            xfit = np.polyval(coeff, scale_ax)
            # detrending and computing RMS of each window
            rms.append(np.sqrt(np.mean((xcut-xfit)**2)))
            i += overlap
        rms = np.array(rms)

    return rms


    
def dfa(x,scale_lim=[5,9],scale_dens=0.25,overlap = 0,det  = 1):
    
    """
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of lenght 2 
        boundaries of the scale where scale means windows in which RMS
        is calculated. Numbers from list are indexes of 2 to the power
        of range.
      *scale_dens* = 0.25 : float
        density of scale divisions
      *show* = False
        if True it shows matplotlib picture
      *overlap*: percentage of allowed overlap between windows
      
    Returns:
    --------
      *scales* : numpy.array
        vector of scales
      *fluct* : numpy.array
        fluctuation function
      *alpha* : float
        DFA exponent
    """
    
    y = np.cumsum(x - np.mean(x))# Signal profile
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(int)
    

    fluct = np.zeros(len(scales))
    overlap = 1 -overlap
    err = np.zeros(len(scales))
    for e, sc in enumerate(scales):
        c = calc_rms(y, sc, int(overlap*sc), det)
        fluct[e] = np.mean(c)
        err[e] = np.std(c)/np.sqrt(len(c))

    return scales, fluct, err

def plot_fluct(scales,fluct,err,show = 0,ax = None, xmin = 'default', xmax = 'default', col = 'orange'):
    
    if xmin == 'default':
        xmin = min(scales)
    elif type(xmin) == int or type(xmin) == float:
        xmin = 2**xmin
    else:
        raise ValueError('xmin must be int or float or "default"')
        
    if xmax == 'default':
        xmax = max(scales)
    elif type(xmax) == int or type(xmax) == float:
        xmax = 2**xmax
    else:
        raise ValueError('xmax must be int or float or "default"')
        
    fluctcopy = fluct[np.array(scales >= xmin) & np.array(scales <= xmax)]
    errcopy = err[np.array(scales >= xmin) & np.array(scales <= xmax)]
    scalescopy = scales[np.array(scales >= xmin) & np.array(scales <= xmax)]
    
    x = sm.add_constant(np.log2(scalescopy), prepend=False)
    mod = sm.OLS(np.log2(fluctcopy),x)
    v =mod.fit()
    rsq = v.rsquared
    
    coef = [v.params[0], v.bse[0]]
    inter = v.params[1]
    fluctfit = 2**np.polyval([coef[0], inter],np.log2(scalescopy))
   
    if show:
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)    
        ax.errorbar(scales, fluct, yerr = np.array(err)*2, fmt = '^', color = col, ecolor =col, ms = 7, alpha =0.8, elinewidth = 2, capsize = 2)
        ax.plot(scalescopy, fluctfit, color = 'tab:red',lw =6, label='DFA exp = %0.2f'%coef[0])
        ax.set_xlabel('Time window [seconds]')
        ax.set_ylabel('F(t)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

    return coef[0], rsq
    