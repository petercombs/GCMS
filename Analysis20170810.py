from collections import defaultdict
from glob import glob
from matplotlib import pyplot as mpl
from matplotlib.pyplot import figure, legend, xticks, text, scatter
from scipy.io import netcdf_file
from sklearn.decomposition import PCA
import GCMSUtils as gu
import GCMS_Plots as gp
import Utils as ut
import numpy as np
import pandas as pd


norm_kwargs = dict(zeroed=20, t_offset='auto', normed=(1010, 1032),
                       norm_method='max')
norm_kwargs2 = norm_kwargs.copy()
norm_kwargs2['normed'] = (1050, 1075)


if __name__ == "__main__":
    samples = defaultdict(list)
    for fname in glob('*/*.CDF'):
        sample = netcdf_file(fname)
        title = sample.experiment_title.decode().replace('pac', '').split('r')[0].strip('_').lstrip('_-').replace('zaza-', 'zaza')
        samples[title].append(sample)

    figure()

    plot_cycler2 = (mpl.cycler(lw=[1,2,3,4,5,])
                    * mpl.cycler(linestyle=['-', ':', '-.', '--']))

    all_data = pd.DataFrame()
    for key, c in zip(['eloF947+', 'eloF947-', 'gfp186+',  'secCRISPRA-',
                        'secCRISPRB-', 'tsimbazaza'],
                       ['r', 'g', 'm', 'c', 'b', 'k']):
        cycler = iter(plot_cycler2)
        for sample in samples[key]:
            kw = norm_kwargs2 if sample.experiment_date_time_stamp.decode()[:4] == '2017' else norm_kwargs
            kw = kw.copy()
            kw['c'] = c
            kw.update(next(cycler))
            kw['jitter'] = 0

            tic, times, artists = gp.plot_tic(sample, **kw)
            sample_title = (sample.experiment_title
                            .decode()
                            .strip('pac').strip('PAC')
                            .lstrip('-_')
                           )
            all_data[sample_title] = gu.measure_bins(tic, times, gu.bins)

    legend()
    xticks([-77, -29, 16, 63], ["23C", "25C", "27C", "29C"])

    all_data_normed = all_data.T - all_data.T.mean()
    all_data_normed /= all_data_normed.std()

    figure()
    pca = PCA()
    pca.fit(all_data_normed.select(ut.contains(['+', 'tsimbazaza'])))
    pc1 = np.dot(pca.components_[0], all_data_normed.T)
    pc2 = np.dot(pca.components_[1], all_data_normed.T)
    scatter(pc1, pc2)
    for sample, x, y in zip(all_data_normed.index, pc1, pc2):
        text(x, y, sample, )


    '''
    all_data
    pca = PCA()
    pca.fit(all_data)
    pca.components_
    pca.components_.shape
    all_data.shape
    pca.fit(all_data.T)
    pca.components_.shape
    get_ipython().magic('pinfo pca.inverse_transform')
    all_data[0]
    all_data.ix[:, 0]
    np.dot(pca[0], all_data.ix[:, 0])
    np.dot(pca.components_[0], all_data.ix[:, 0])
    np.dot(pca.components_[0], all_data.ix[:, 0] - all_data.T.mean())
    np.dot(pca.components_[1], all_data.ix[:, 0] - all_data.T.mean())
    np.dot(pca.components_[1], all_data.T - all_data.T.mean())
    np.dot(pca.components_[1], all_data - all_data.T.mean())
    np.dot(pca.components_[1], (all_data.T - all_data.T.mean()).T)
    pca.components_.shape
    all_data.T - all_data.T.mean()
    all_data.T.mean()
    '''
