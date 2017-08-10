from scipy import optimize
from numpy import sum, mean, max, empty_like, zeros_like, array
import numpy as np
import pandas as pd

def rolling_minimum(arr, window):
    retval = empty_like(arr)
    for i in range(len(retval)):
        retval[i] = min(arr[i:i+window])
    return retval

def max_mass(cdf_file, min=np.inf):
    mass_values = cdf_file.variables['mass_values'].data
    intensities = cdf_file.variables['intensity_values'].data
    time_values = cdf_file.variables['scan_acquisition_time'].data
    scan_indices = array(cdf_file.variables['scan_index'].data[1:]) - 1
    retval = zeros_like(scan_indices)
    if min == np.inf:
        return mass_values[scan_indices]
    time_iter = iter(time_values)
    time = next(time_iter)
    out_ix = 0
    for ix, (mass, int) in enumerate(zip(mass_values, intensities)):
        if int < min:
            continue
        if ix > time:
            try:
                time = next(time_iter)
            except StopIteration:
                time = np.inf
            out_ix  += 1
        try:
            retval[out_ix] = mass
        except IndexError:
            pass
    return retval



norm_methods = {
    'sum': sum,
    'mean': mean,
    'max': max,
}

def normalize_tic(tic, times, t_offset=0.0, zeroed=0, norm_method='mean', normed=[1]):
    corr = zeroed
    if corr == 0:
        pass
    else:
        corr = rolling_minimum(tic,
                               corr)
        tic -= corr

    if norm_method not in norm_methods and not callable(norm_method):
        raise ValueError('Unknown normalization method: "{}"; must be one of {}'
                         .format(norm_method,
                                 ', '.join(norm_methods)))
    else:
        norm_method = norm_methods.get(norm_method, norm_method)

    if len(normed) >= 2:
        normer = norm_method(tic[(normed[0] < times) & (times < normed[1])])
        tic /= (normer+.01)
        if len(normed) >=3:
            tic *= normed[2]
    else:
        tic /= normed[0]


    if t_offset == 'auto' and 'normed' in locals():
        range = normed[:2]
        sel = (range[0] < times) & (times < range[1])
        errfunc = lambda p, x, y: p[0] * np.exp(-(x-p[1])**2/p[2]) - y
        p0 = [1, mean(times[sel]), 1]
        p2, success = optimize.leastsq(errfunc, p0,
                                       args=(times[sel], tic[sel])
                                      )
        if (min(times[sel]) - 5) < p2[1] < (max(times[sel]) + 5):
            t_offset = p2[1]
        else:
            t_offset = 0.0
            print("Poor fit for normalization peak: t={}".format(p2[1]))
    times -= t_offset

    return tic, times


bins = {
    # Odd carbon chains
    # E indicates early, L indicates late
    '23C': (-75, -71),
    '23C-1x-L': (-78.6, -76.23),
    '23C-1x-E': (-80.5, -78.6),
    '23C-2x': (-84.5, -81),
    '25C': (-25, -22),
    '25C-1x-L': (-28, -26.2),
    '25C-1x-E': (-30, -28),
    '25C-2X': (-33, -30),
    '27C': (21, 23.5),
    '27C-1x-L': (18.6, 20.2),
    '27C-1x-E': (16.9, 18.6),
    '27C-2x-L': (14.7, 16.4),
    '27C-2x-E': (13.0, 14.7),
    '29C-4x?': (56, 58.3),
    '29C-2x': (58.3, 60.5),
    '29C-2me?': (64, 66.5),
    # Even carbon chains
    '28C': (35, 40),
    '24C': (-49, -47.5),
    '22C': (-100.6, -98.5),
}

# Jallon & David "Peak No" to GCMS compound I have
jd_to_compound = pd.Series({
    1:  '23C-2x',
    3:  '23C-1x-L',
    2:  '23C-1x-E',
    5:  '23C',
    6:  '25C-2X',
    8:  '25C-1x-E',
    9:  '25C-1x-L',
    11: '25C',
    12: '27C-2x-E',
    13: '27C-2x-L',
    15: '27C-1x-E',
    16: '27C-1x-L',
    17: '27C',
    18: '29C-2x',
    19: '29C-4x?',
    21: '29C-2me?',
})

compound_to_jd = pd.Series({value:key for key, value in jd_to_compound.items()})


def measure_bins(tic, times, bins, acc_func=np.sum):
    ret = pd.Series(index=bins, data=np.nan)
    for bin in bins:
        bin_lo, bin_hi = min(bins[bin]), max(bins[bin])
        ret[bin] = acc_func(tic[(bin_lo < times) & (times < bin_hi)])
    return ret


