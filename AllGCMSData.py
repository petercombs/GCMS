from GCMS_Plots import normalize_tic
from GCMSTests import bins as compound_bins
from scipy.io import netcdf_file
import numpy as np
import pandas as pd
from sys import stdout
from os import path

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('in_files', nargs='+')
    parser.add_argument('--outfile', default=stdout)
    parser.add_argument('--t-offset', default='auto')
    parser.add_argument('--zeroed', default=20, type=int)
    parser.add_argument('--norm-method', default='max')
    return parser.parse_args()


def parse_file(sample, norm_args, acc_func=np.sum):

    tic = np.array(sample.variables['total_intensity'].data)
    times = np.array(sample.variables['scan_acquisition_time'].data)
    tic, times = normalize_tic(tic, times, **norm_args)
    expt_title = (sample
                  .experiment_title
                  .decode()
                  .replace('pac', '')
                  .replace('PAC', '')
                  .lstrip('-_')
                  #.split('_r')[0]
                 )
    compounds = pd.Series(index=compound_bins, data=np.nan, name=expt_title)
    for bin in compound_bins:
        bin_lo, bin_hi = min(compound_bins[bin]), max(compound_bins[bin])
        compounds[bin] = acc_func(tic[(bin_lo < times) & (times < bin_hi)])

    return compounds



if __name__ == "__main__":
    args = parse_args()
    samples = {fname: netcdf_file(fname) for fname in args.in_files}
    norm_kwargs = dict(zeroed=20, t_offset='auto', normed=(1010, 1032),
                       norm_method='max')
    out = pd.DataFrame(
        parse_file(sample, norm_kwargs)
        for sample in samples.values()
    )

