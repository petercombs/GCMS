from GCMS_Plots import normalize_tic
from GCMSUtils import bins
from scipy.io import netcdf_file
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from sys import argv
from os import path

def compare_samples(samples, test_types, control_types, bins, acc_func=np.sum,
                    skip_types=None):
    out = pd.DataFrame(index=sorted(bins),
                       columns=['t_start', 't_end',
                                'test_mean', 'control_mean', 'tstat',
                                'pval']
                      )

    test_data = []
    control_data = []

    test_filenames = []
    control_filenames = []

    norm_kwargs = dict(zeroed=20, t_offset='auto', normed=(1016, 1022),
                       norm_method='max')

    for sample in samples:
        tic = np.array(sample.variables['total_intensity'].data)
        times = np.array(sample.variables['scan_acquisition_time'].data)
        expt_title = sample.experiment_title.decode().split('_r')[0]
        ided = False
        for skip_type in skip_types:
            if skip_type in expt_title:
                ided = True
        if ided:
            print("Explicitly skipping ", expt_title, sample.filename)
            continue
        for test_type in test_types:
            if test_type in expt_title:
                test_data.append(normalize_tic(tic, times, **norm_kwargs))
                test_filenames.append(sample.filename)
                ided = True
                break
        for control_type in control_types:
            if control_type in expt_title and not ided:
                control_data.append(normalize_tic(tic, times, **norm_kwargs))
                control_filenames.append(sample.filename)
                ided = True
                break
        if not ided:
            print("skipping ", expt_title, sample.filename)

    for bin in bins:
        bin_lo, bin_hi = min(bins[bin]), max(bins[bin])
        bin_test = []
        bin_control = []
        for tic, times in test_data:
            bin_test.append(acc_func(tic[(bin_lo < times) & (times < bin_hi)]))
        for tic, times in control_data:
            bin_control.append(acc_func(tic[(bin_lo < times) & (times < bin_hi)]))

        out.ix[bin, 't_start'] = bin_lo
        out.ix[bin, 't_end'] = bin_hi
        out.ix[bin, 'test_mean'] = np.mean(bin_test)
        out.ix[bin, 'control_mean'] = np.mean(bin_control)
        tstat, pval = ttest_ind(bin_test, bin_control)
        out.ix[bin, 'tstat'] = tstat
        out.ix[bin, 'pval'] = pval

    out.filenames = {'test': test_filenames, 'control':control_filenames}
    print(out.filenames)
    return out




def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--control-types', '-C', nargs='+')
    parser.add_argument('--test-types', '-T', nargs='+')
    parser.add_argument('--test-name', '-n', default='GCMS_test')
    parser.add_argument('--skip-types', '-s', nargs='+')
    parser.add_argument('--outdir', '-o', default='.')
    parser.add_argument('in_files', nargs='+')
    return parser.parse_args()


out_header = """###
# {} vs {}
# Test files: {test}
# Control files: {control}
#
#
"""
if __name__ == "__main__":
    args = parse_args()
    samples = [netcdf_file(fname) for fname in args.in_files]
    result = compare_samples(samples, args.test_types, args.control_types,
                             bins, skip_types=args.skip_types)
    outfile = open(path.join(args.outdir, args.test_name + '.tsv'), mode='w')
    outfile.write(out_header.format(','.join(args.test_types),
                                    ','.join(args.control_types),
                                    **result.filenames
                                   )
                 )
    result.sort_values(by='t_start').to_csv(outfile, sep='\t')
    outfile.close()

