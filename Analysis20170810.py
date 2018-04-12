from collections import defaultdict, OrderedDict
from glob import glob
from matplotlib import pyplot as mpl
from matplotlib.pyplot import (figure, legend, xticks, text, scatter,
                               bar, savefig, style)
from os import path
from scipy.io import netcdf_file
from sklearn.decomposition import PCA
import GCMSUtils as gu
import GCMS_Plots as gp
import Utils as ut
import numpy as np
import pandas as pd

style.use('lowink')


norm_kwargs = dict(zeroed=20, t_offset='auto', normed=(1010, 1032),
                       norm_method='max')
norm_kwargs2 = norm_kwargs.copy()
norm_kwargs2['normed'] = (1060, 1070)


if __name__ == "__main__":
    normers = {}
    allsamples = defaultdict(list)
    for fname in glob('*/*.CDF'):
        sample = netcdf_file(fname)
        title = sample.experiment_title.decode().replace('pac', '').split('r')[0].strip('_').lstrip('_-').replace('zaza-', 'zaza')
        allsamples[title].append(sample)

    figure()

    plot_cycler2 = (mpl.cycler(lw=[1,2,3,4,5,])
                    * mpl.cycler(linestyle=['-', ':', '-.', '--']))

    clist = OrderedDict([
        (1, 'darkblue'),  # MelWT
        (2, 'lightblue'), # eloF-
        (5, 'red'),        # Sec WT
        (3, 'darkorange'),       # CRISPR A
        (4, 'lightsalmon'),     # CRISPR B
        (6, 'black'),
    ])
    sample_colors = {
        'eloF947+': clist[1],
        'eloF947-': clist[2],
        'gfp186+': clist[1],
        #'gfp202': clist[1],
        #'gfp201': clist[1],
        'gfp201-': clist[1],
        'bond+': clist[1],
        'secCRISPRA-': clist[3],
        'sechellia-CRISPRA-': clist[3],
        'secCRISPRB-': clist[4],
        'sechellia-CRISPRB-': clist[4],
        'sechellia_CRISPRB_4d': clist[4],
        'sechellia_WT_4d': clist[5],
        'tsimbazaza': clist[6],
        'tsimbazaza_4d': clist[6],
    }
    color_names = {
        clist[1]: r'$\it{D.\ mel}$ WT',
        clist[2]: r'$\it{D.\ mel\ eloF}$-',
        clist[3]: r'$\it{D.\ sec\ eloF}$- A',
        clist[4]: r'$\it{D.\ sec\ eloF}$- B',
        clist[5]: r'$\it{D.\ sec}$ WT',
        clist[6]: r'$\it{D.\ sim}$ WT'
    }
    r_color_names = {val: key for key,val in color_names.items()}
    all_data = pd.DataFrame()
    for key, c in sorted(sample_colors.items()):
        cycler = iter(plot_cycler2)
        for sample in allsamples[key]:
            kw = norm_kwargs2 if sample.experiment_date_time_stamp.decode()[:4] == '2017' else norm_kwargs
            normers[sample] = kw.copy()
            kw = kw.copy()
            kw['c'] = c
            kw.update(next(cycler))
            #kw['jitter'] = 0

            tic, times, artists = gp.plot_tic(sample, **kw)
            sample_title = (sample.experiment_title
                            .decode()
                            .strip('pac').strip('PAC')
                            .lstrip('-_')
                           )
            print(sample_title)
            binfile = path.join(path.dirname(sample.filename),
                                'bins.pkl')
            all_data[sample_title] = gu.measure_bins(tic, times,
                                                     binfile)

    legend()
    xticks([-77, -29, 16, 63], ["23C", "25C", "27C", "29C"])

    all_data = all_data.sort_index(axis=0)
    all_data_normed = all_data.T - all_data.T.mean()
    all_data_normed /= all_data_normed.std()

    pca = PCA()
    wt_data = all_data_normed.select(ut.contains(['+', 'tsimbazaza', 'WT']))
    pca.fit(wt_data)
    pc1 = pd.Series(np.dot(pca.components_[0], all_data_normed.T),
                    index=all_data.columns)
    pc2 = pd.Series(np.dot(pca.components_[1], all_data_normed.T),
                    index=all_data.columns)

    mpl.figure(figsize=(6.5, 2.25))

    bar(height=pca.components_[0],
        left=np.arange(pca.components_.shape[1])-.125,
        width=.2,
        label='PC1')
    bar(height=pca.components_[1],
        left=np.arange(pca.components_.shape[1])+.125,
        width=.2,
        label='PC2')
    legend(loc='upper left')

    xticks(np.arange(len(wt_data.columns)), wt_data.columns, rotation=90)
    mpl.hlines(0, -0.3, pca.components_.shape[1]+.3)
    ax = mpl.gca()
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_bounds(-.25, .25)
    mpl.tight_layout()
    mpl.savefig('pcs.clean.eps', dpi=300)


    figure(figsize=(3.5,2))
    pc_all = pd.DataFrame(
        dict(pc1=pc1,
             pc2=pc2,
             c=[next(sample_colors[k] for k in sample_colors if k in i)
                for i in all_data.columns]
            )
    )
    for c in clist.values():
        subset = (pc_all.c == c)
        scatter(
            pc_all.loc[subset, 'pc1'],
            pc_all.loc[subset, 'pc2'],
            c=c,
            label=color_names[c]
        )
    mpl.xlabel('PC 1 ({:0.1%})'.format(pca.explained_variance_ratio_[0]))
    mpl.ylabel('PC 2 ({:0.1%})'.format(pca.explained_variance_ratio_[1]))

    ax = mpl.gca()
    fig = mpl.gcf()
    ax.set_aspect(1.1)
    fig.subplots_adjust(left=0.1, right=0.7, bottom=0.2)
    savefig('pca_unlabelled.eps')
    mpl.legend(loc='center left', bbox_to_anchor=(.95,0.5), frameon=True)
    savefig('pca_legend.eps')
    for sample, x, y in zip(all_data_normed.index, pc1, pc2):
        if '+' in sample:
            sample = 'D. mel WT'
        elif 'tsimb' in sample:
            sample = 'D. sim WT'
        elif 'sec' in sample and 'WT' in sample:
            sample = 'D. sec WT'
        text(x, y, sample, )
    savefig('pca_labelled.eps')

    figure()

    scatter(pc1.select(ut.contains(wt_data.index)),
            pc2.select(ut.contains(wt_data.index)))

    for sample, x, y in zip(all_data_normed.index, pc1, pc2):
        if '+' in sample:
            sample = 'D. mel WT'
        elif 'tsimb' in sample:
            sample = 'D. sim WT'
        elif 'sec' in sample and 'WT' in sample:
            sample = 'D. sec WT'
        else:
            continue
        text(x, y, sample, )
    mpl.xlabel('PC 1 ({:0.1%})'.format(pca.explained_variance_ratio_[0]))
    mpl.ylabel('PC 2 ({:0.1%})'.format(pca.explained_variance_ratio_[1]))


    pcs = (pd.DataFrame(columns=all_data.index, data=pca.components_)
           .sort_index(axis=1))


    savefig('labelled_pca_WTonly.eps')

    figure(figsize=(3.25, 2))

    genotype_data = [
        (('gfp186-', 'gfp186+'), 'D. mel WT', 'b'),
        (('eloF947-',), 'D. mel eloF-', 'lightblue')
    ]
    i = -1
    for genotypes, label, color in genotype_data:
        i=-i
        for genotype in genotypes:
            for sample in allsamples[genotype]:
                gp.plot_tic(sample,
                            normed=[1016,1022,i],
                            t_offset='auto',
                            zeroed=5,
                            label=label, color=color, linewidth=.5)
                if not label.startswith('_'):
                    label = '_' + label
    mpl.xlim(-90, 90)
    mpl.ylim(-25, 25)
    legend(ncol=2,loc='upper center', frameon=False)
    mpl.ylabel('Relative TIC')
    mpl.xlabel('Relative Retention Time (s)')
    ax = mpl.gca()
    ax.spines['left'].set_bounds(-20, 20)
    mpl.tight_layout()
    savefig('eloF_vs_WT.eps', transparent=True)


    figure(figsize=(3.25, 2))

    genotype_data = [
        (('sechellia_WT_4d',), 'D. sec WT', 'red'),
        (('sechellia_CRISPRB_4d', 'secCRISPRA-', 'secCRISPRB-'), 'D. sec eloF-',
         'darksalmon')
    ]
    i = -1
    for genotypes, label, color in genotype_data:
        i=-i
        for genotype in genotypes:
            for sample in allsamples[genotype]:
                kwargs = normers[sample]
                kwargs['normed'] = list(kwargs['normed']) + [i]
                gp.plot_tic(sample,
                            color=color, linewidth=0.5,
                            label=label,
                            **kwargs
                           )
                if not label.startswith('_'):
                    label = '_' + label
    mpl.xlim(-90, 90)
    mpl.ylim(-8, 8)
    legend(ncol=2,loc='upper center', frameon=False)
    mpl.ylabel('Relative TIC')
    mpl.xlabel('Relative Retention Time (s)')
    ax = mpl.gca()
    ax.spines['left'].set_bounds(-5, 5)
    mpl.tight_layout()
    savefig('eloF_sec_vs_WT.eps', transparent=True)



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
