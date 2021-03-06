from numpy import  array,  empty_like
from matplotlib.pyplot import plot, vlines
import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
import itertools
from glob import glob
from os import path
from sys import argv
from scipy.io import netcdf_file
from GCMSUtils import max_mass, normalize_tic

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


def plot_max_mass(cdf_file, *args, **kwargs):
    min = kwargs.pop('min_mass', np.inf)
    return plot(cdf_file.variables['scan_acquisition_time'].data[1:],
                max_mass(cdf_file, min))



plot_cycler = (mpl.cycler(lw=[1,2,3,4,5,])
                   * mpl.cycler(linestyle=['-', ':', '-.', '--'])
                   * mpl.cycler(c='bgrcmk'))

def plot_tic(cdf_file, t_offset=0.0, zeroed=0, normed=[1], norm_method='mean',
             jitter=0.0, *args, **kwargs):
    kwargs = kwargs.copy()
    tic = array(cdf_file.variables['total_intensity'].data)
    times = array(cdf_file.variables['scan_acquisition_time'].data)
    ax = kwargs.pop('ax', mpl.gca())
    if not hasattr(ax, 'cycler'):
        ax.cycler = iter(plot_cycler)
    for key, value in next(ax.cycler).items():
        if key not in kwargs:
            kwargs[key] = value


    tic, times = normalize_tic(tic, times,
                               t_offset, zeroed, norm_method, normed)


    label = kwargs.pop('label', cdf_file.experiment_title.decode())
    retval = ax.plot(times+(2*np.random.rand()-1)*jitter,
                     tic,
                     *args,
                     label=label,
                     **kwargs)
    return tic, times, retval

def plot_spectrum(cdf_file, time, jitter=0.0, normed=False, *args, **kwargs):
    if not np.iterable(time):
        time = [time]
    colors = kwargs.pop('colors', itertools.cycle('brgcmk'))
    if hasattr(colors, 'len') and len(colors)==1:
        colors = itertools.repeat(colors)
    for t,c in zip(time, colors):
        times = cdf_file.variables['scan_acquisition_time'].data
        best_time_ix = np.argmin(abs(times - t))
        label = ('{} @t= {:.1f}'
                 .format(kwargs.pop('label', cdf_file.experiment_title.decode()),
                         times[best_time_ix]))


        ms_coords = empty_like(cdf_file.variables['scan_acquisition_time'].data)
        ms_coords[0] = 0
        ms_coords[1:] = np.cumsum(cdf_file.variables['point_count'].data)[:-1]
        ms_idx_lo, ms_idx_hi = ms_coords[best_time_ix: best_time_ix+2]
        masses = array(cdf_file.variables['mass_values']
                       .data[ms_idx_lo: ms_idx_hi])
        masses += jitter * np.random.rand()
        heights = array(cdf_file .variables['intensity_values'].
                        data[ms_idx_lo: ms_idx_hi])
        if normed:
            heights /= max(heights)

        vlines(masses, 0, heights, *args, colors=c, label=label, **kwargs)

def plot_all_tics(list_of_lists):
    color_list = [color for  color in
                      mpl.cm.Set1(np.linspace(0, 1, len(list_of_lists),
                                              endpoint=True))
                 ]

    for color, samples in zip(color_list, list_of_lists):
        for sample in samples:
            plot_tic(sample,
                     color=color,
                     zeroed=20,
                     t_offset='auto',
                     normed=(1016,1022),
                     norm_method='max')


if __name__ == "__main__":
    if argv[1:] and np.all([path.isdir(dname) for dname in argv[1:]]):
        files = {dir: glob(path.join(dir, '*.CDF')) for dir in argv[1:]}
    elif argv[1:]:
        files = {'Input': argv[1:]}
    else:
        files = glob('*/*.CDF')
        files = {dir: glob(path.join(dir, '*.CDF'))
                 for dir in {path.dirname(file) for file in files}}

    for day in files:
        day_files = [netcdf_file(file) for file in files[day]]
        sample_types = {file.experiment_title.decode().split('_r')[0]
                        for file in day_files}
        if not day_files: continue
        mpl.figure()
        mpl.title(day)
        color_list = {type: color for type, color in
                      zip(sample_types,
                          mpl.cm.Set3(np.linspace(0, 1, len(sample_types)))
                         )
                     }
        for file in day_files:
            type = file.experiment_title.decode().split('_r')[0]
            plot_tic(file,
                     color=color_list[type],
                     zeroed=20,
                     t_offset='auto',
                     normed=(1016,1022),
                     norm_method='max')
        mpl.legend()


