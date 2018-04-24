from GCMS_Plots import normalize_tic
from GCMSUtils import bins, measure_bins
from scipy.io import netcdf_file
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as mpl
from matplotlib import rc

rc('text', usetex=True)
mpl.ion()

verbose = False


def compare_samples(samples, test_types, control_types, bins, acc_func=np.sum,
                    skip_types=None, norm_range=(1010, 1022)):
    out = pd.DataFrame(index=sorted(bins),
                       columns=['t_start', 't_end',
                                'test_mean', 'control_mean', 'tstat',
                                'l2fc',
                                'pval'],
                       data=np.nan,
                      )

    test_data = []
    control_data = []

    test_samples = []
    control_samples = []

    test_filenames = []
    control_filenames = []

    seen_files = set()

    norm_kwargs = dict(zeroed=20, t_offset='auto', normed=norm_range,
                       norm_method='max')
    if skip_types is None:
        skip_types = []

    for sample in samples:
        base = path.basename(sample.filename)
        if base in seen_files:
            print("Already seen", base)
            continue
        seen_files.add(base)

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
                print(expt_title, sample.filename)
                test_data.append(normalize_tic(tic, times, **norm_kwargs))
                test_filenames.append(sample.filename)
                test_samples.append(sample)
                ided = True
                break
        for control_type in control_types:
            if control_type in expt_title and not ided:
                print(expt_title, sample.filename)
                control_data.append(normalize_tic(tic, times, **norm_kwargs))
                control_filenames.append(sample.filename)
                control_samples.append(sample)
                ided = True
                break
        if not ided:
            if verbose:
                print("skipping ", expt_title, sample.filename)


    test_dataset = pd.DataFrame()
    control_dataset = pd.DataFrame()

    for sample, (tic, times) in zip(test_samples, test_data):
        test_dataset[sample.filename] = measure_bins(tic, times,
                                                     bins=sample,
                                                     acc_func=acc_func)
    for sample, (tic, times) in zip(control_samples, control_data):
        control_dataset[sample.filename] = measure_bins(tic, times,
                                                        bins=sample,
                                                        acc_func=acc_func)
    for bin in control_dataset.index:
        out.ix[bin, 'test_mean'] = test_dataset.ix[bin].mean()
        out.ix[bin, 'control_mean'] = control_dataset.ix[bin].mean()
        tstat, pval = ttest_ind(test_dataset.ix[bin], control_dataset.ix[bin])
        out.ix[bin, 'tstat'] = tstat
        out.ix[bin, 'pval'] = pval
        out.ix[bin, 'l2fc'] = np.log2(test_dataset.ix[bin].mean() /
                                      control_dataset.ix[bin].mean())

    '''
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
        '''

    out.filenames = {'test': test_filenames, 'control':control_filenames}
    print(out.filenames)
    return out, test_data, control_data, test_samples, control_samples




def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--control-types', '-C', nargs='+')
    parser.add_argument('--test-types', '-T', nargs='+')
    parser.add_argument('--test-name', '-n', default='GCMS_test')
    parser.add_argument('--test-label', default='Test')
    parser.add_argument('--control-label', default='Control')
    parser.add_argument('--test-color', default='lightblue')
    parser.add_argument('--control-color', default='blue')
    parser.add_argument('--skip-types', '-s', nargs='+')
    parser.add_argument('--outdir', '-o', default='.')
    parser.add_argument('--out-fname', '-O', default=False)
    parser.add_argument('--norm-peak-start', '-l', default=1010, type=float)
    parser.add_argument('--norm-peak-stop', '-r', default=1022, type=float)
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
    retval = compare_samples(samples, args.test_types, args.control_types, bins,
                                                      skip_types=args.skip_types,
                                                      norm_range=(args.norm_peak_start,
                                                                  args.norm_peak_stop))
    result, test_data, control_data, test_samples, control_samples  = retval
    outfile = (open(path.join(args.outdir, args.test_name + '.tsv'), mode='w') if
               not args.out_fname else open(args.out_fname, mode='w'))
    outfile.write(out_header.format(','.join(args.test_types),
                                    ','.join(args.control_types),
                                    **result.filenames
                                   )
                 )
    result.sort_values(by='t_start').to_csv(outfile, sep='\t',
                                            float_format='%.4f')
    outfile.close()

    fbase, ext = path.splitext(outfile.name)


    mpl.style.use('lowink')
    mpl.figure(figsize=(3.25,2))
    c = args.control_color
    label=args.control_label
    top_abs = 0
    for i, (tic, times) in enumerate(control_data):
        times_in = (-100 < times) & (times < 100)
        if sum(times_in) == 0: continue
        mpl.plot(times, tic, color=c, label=label)
        label = '_' + label
        top_abs = max(tic[times_in].max(), top_abs)

    c = args.test_color
    label=args.test_label
    for tic, times in test_data:
        times_in = (-100 < times) & (times < 100)
        if sum(times_in) == 0: continue
        mpl.plot(times, -tic, color=c, label=label)
        label = '_' + label
        top_abs = max(tic[times_in].max(), top_abs)
    mpl.legend(loc='upper left', frameon=False,
               bbox_to_anchor=(0.0, 1.2))
    ax = mpl.gca()
    #mpl.ylim(-top_abs*1.1, top_abs*1.1)
    mpl.ylim(-7,7)
    mpl.yticks([-5, 0, 5], [5, 0, 5])
    mpl.xticks([-50, 0, 50], [-50, 0, 50])
    mpl.show()
    top_abs = np.round(top_abs / 5) * 5
    #ax.spines['left'].set_bounds(-top_abs, top_abs)
    ax.spines['left'].set_bounds(-5, 5)
    mpl.xlim(-90, 90)
    mpl.xlabel('Relative Retention Time (s)')
    mpl.ylabel('Relative TIC')
    mpl.tight_layout()
    mpl.savefig(fbase + '.eps', dpi=300)
    mpl.savefig(fbase + '.png', dpi=300)


