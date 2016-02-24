from numpy import mean, array,  min, empty_like, zeros_like
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from scipy import optimize

def rolling_minimum(arr, window):
    retval = empty_like(arr)
    for i in range(len(retval)):
        retval[i] = min(arr[i:i+window])
    return retval

def convert_to_matrix(cdf_file):
    mass_values = cdf_file.variables['mass_values'].data
    time_values = cdf_file.variables['scan_acquisition_time'].data
    scan_indices = iter(cdf_file.variables['scan_index'].data)
    intensities = cdf_file.variables['intensity_values'].data
    output = pd.DataFrame(
        index=sorted(set(mass_values)),
        columns=time_values,
        data=0
    )

    time_iter = iter(time_values)
    time = next(time_iter)

    max_ix = next(scan_indices)
    max_ix = next(scan_indices)
    for ix, (m, i) in enumerate(zip(mass_values, intensities)):
        while ix > max_ix:
            try:
                max_ix = next(scan_indices)
            except StopIteration:
                max_ix = 1e100
            time = next(time_iter)
        output.ix[m, time] = i
    print(time, max(time_values), ix)
    return output

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


def plot_max_mass(cdf_file, *args, **kwargs):
    min = kwargs.pop('min_mass', np.inf)
    return plot(cdf_file.variables['scan_acquisition_time'].data[1:],
                max_mass(cdf_file, min))




norm_methods = {
    'sum': sum,
    'mean': mean,
    'max': max,
}

def plot_tic(cdf_file, *args, **kwargs):
    tic = array(cdf_file.variables['total_intensity'].data)
    times = array(cdf_file.variables['scan_acquisition_time'].data)
    corr = kwargs.pop('zeroed', 0)
    if corr == 0:
        pass
    else:
        corr = rolling_minimum(tic,
                               corr)
        tic -= corr

    normed = kwargs.pop('normed', False)
    norm_method = kwargs.pop('norm_method', 'mean')
    if norm_method not in norm_methods and not callable(norm_method):
        raise ValueError('Unknown normalization method: "{}"; must be one of {}'
                         .format(norm_method,
                                 ', '.join(norm_methods)))
    else:
        norm_method = norm_methods.get(norm_method, norm_method)

    if normed and len(normed) >= 2:
        normer = norm_method(tic[(normed[0] < times) & (times < normed[1])])
        tic /= normer
        if len(normed) >=3:
            tic *= normed[2]
    elif normed:
        tic /= normed[0]


    t_offset = kwargs.pop('t_offset', 0.0)
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
            print("Poor fit for normalization peak: t={} ({})".format(p2[1]),
                  cdf_file.experiment_title)
    times -= t_offset


    label = kwargs.pop('label', cdf_file.experiment_title.decode())
    retval = plot(times,
                  tic,
                  *args,
                  label=label,
                  **kwargs)
    return retval

