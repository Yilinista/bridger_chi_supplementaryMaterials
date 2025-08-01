# -*- coding: utf-8 -*-

# from notebooks/Untitled78_analysisworking.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string, math

from .analysis import filter_highpass, filter_lowhigh

CONFIDENCE = 90


def plots_preference(df_results: pd.DataFrame):
    colnames_sets = [
        (
            "Checked Papers/Facets",
            [
                c
                for c in df_results.columns
                if "novelty_score_overall" in c and "only" not in c
            ],
        ),
        (
            "Only Facets",
            [c for c in df_results.columns if "novelty_score_overall_onlyfacets" in c],
        ),
        (
            "Only Papers",
            [c for c in df_results.columns if "novelty_score_overall_onlypaper" in c],
        ),
    ]

    idx = 0
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(colnames_sets),
        #                          figsize=(9,4),
        figsize=(10, 3.2),
        squeeze=True,
    )

    def _f(x):
        # discard ties
        if np.isnan(x):
            return x
        else:
            return x >= 0.5

    for i in range(axes.shape[0]):
        title, colnames = colnames_sets[idx]
        ax = axes[i]
        _df = df_results.copy()
        _df = filter_highpass(_df)
        _df = _df[["overall_id"] + colnames].drop_duplicates()
        _df = _df.replace(0.5, np.nan)  # discard ties
        # _df = _df.drop(columns=['overall_id']).apply(lambda x: x>=.5, axis=1)
        _df = _df.drop(columns=["overall_id"])
        # drop the "overall" column
        _df = _df.drop(columns=[c for c in colnames if "sT" not in c])
        for col in _df.columns:
            _df[col] = _df[col].apply(_f)

        _df = _df.replace(False, "specter wins")
        _df = _df.replace(True, "bridger wins")
        _df = _df.melt(value_vars=_df.columns).groupby(["variable", "value"]).size()
        _df = (
            _df.groupby(level=0)
            .apply(lambda x: x.dropna() / sum(x.dropna()) * 100)
            .reset_index(name="percent")
        )
        sns.barplot(data=_df, x="variable", y="percent", hue="value", ax=ax)
        ax.set_ylim(0, 100)
        legend = ax.get_legend()
        if i == 0:
            legend.set_title(None)
            ax.legend(loc="upper left")
            ax.set_ylabel("% participants")
        else:
            legend.remove()
            ax.set_ylabel(None)
        #     ax.set_xlabel(title, labelpad=-10)
        ax.set_xlabel(None)
        ax.set_title(title)
        # ax.set_xticklabels(['overall', 'sT', 'sTdM'])
        ax.set_xticklabels(["sT", "sTdM"])
        ax.text(
            -0.07,
            1.05,
            f"({string.ascii_lowercase[idx]})",
            transform=ax.transAxes,
            size=12,
            weight="bold",
        )
        idx += 1
    # plt.savefig('plots/figureNoveltyScoreByConditionCombined.png', dpi=500)

    # plt.show()

    return fig, axes


def draw_chart_split_axis(subfig, colors, df, colname, title, ylim=(0.9, 1.01)):
    # adapted from example at https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    ax1, ax2 = subfig.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [500, 1]}
    )
    subfig.subplots_adjust(hspace=0.08)
    sns.barplot(
        data=df, x="condition_short", y=colname, ax=ax1, ci=CONFIDENCE, palette=colors
    )
    sns.barplot(
        data=df, x="condition_short", y=colname, ax=ax2, ci=CONFIDENCE, palette=colors
    )
    ax1.set_ylim(ylim)
    ax2.set_ylim(0, 0.1)
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    for ax in ax1, ax2:
        ax.set_xlabel(None)
    ax1.set_ylabel("Jaccard distance", labelpad=20)
    ax2.set_ylabel(None)
    ax1.set_title(title)

    # remove the top y tick label from the lower plot
    plt.draw()  # need to call this or else the labels come back blank
    ticks = ax2.get_yticks()
    labels = ax2.get_yticklabels()
    labels[-1] = ""
    labels[0] = ""
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels)

    # make the slanted lines
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


def plots_other_distance_measures(df_results: pd.DataFrame):
    _df = df_results.copy()
    _df = filter_highpass(_df)
    _df.sort_values("condition", inplace=True)
    # _df = _df[_df.condition!="simspecter_hideTerms"]
    vals_rename = {
        "simTask": "sT",
        "simTask_distMethod": "sTdM",
        "simspecter": "ss",
        "simspecter_hideTerms": "ss",
    }
    _df["condition_short"] = _df.condition.replace(vals_rename)
    colnames = [
        "incitation_overlap",
        "outcitation_overlap",
        "venue_overlap",
        "coauthor_shortest_path",
    ]
    display_titles = [
        "Incoming Citations",
        "Outgoing Citations",
        "Publication Venues",
        "Coauthorship Network",
    ]
    # convert similarities to distances
    for colname in colnames:
        if "overlap" in colname:
            _df[colname] = 1 - _df[colname]
    ncols = 2
    # ncols = 4
    fig = plt.figure(figsize=(6, 5))  # for 2-column version
    # fig = plt.figure(figsize=(9,3))  # for 4-column (horizontal) version
    # for i in range(len(colnames)):
    idx = 0
    subfigs = fig.subfigures(
        ncols=ncols, nrows=math.ceil(len(colnames) / ncols), squeeze=False
    )
    colors = sns.color_palette()
    colors = [colors[0], colors[9], colors[1]]
    ylims = [(0.9, 1.01), (0.9, 1.01), (0.75, 1.01)]
    for i in range(subfigs.shape[0]):
        for j in range(subfigs.shape[1]):
            try:
                colname = colnames[idx]
            except IndexError:
                continue
            subfig = subfigs[i][j]
            if "overlap" in colnames[idx]:
                # draw these charts with a broken y-axis
                draw_chart_split_axis(
                    subfig,
                    colors,
                    _df,
                    colnames[idx],
                    display_titles[idx],
                    ylim=ylims[idx],
                )
            else:
                # This one doesn't have a broken axis
                ax = subfig.subplots()
                sns.barplot(
                    data=_df,
                    x="condition_short",
                    y=colnames[idx],
                    ax=ax,
                    ci=CONFIDENCE,
                    palette=colors,
                )
                ax.set_xlabel(None)
                # labelpad looks terrible in notebook but works for savefig for some reason
                ax.set_ylabel("Avg Shortest Path Length", labelpad=65)
                ax.set_title(display_titles[idx])

            subfig.text(
                -0.1,
                1.05,
                f"({string.ascii_lowercase[idx]})",
                transform=subfig.get_axes()[0].transAxes,
                size=12,
                weight="bold",
            )

            idx += 1
    plt.subplots_adjust(left=0.22)
    plt.draw()
    # plt.savefig('plots/figureOtherDistanceMeasures.png', dpi=500)
    # plt.show()
    return fig