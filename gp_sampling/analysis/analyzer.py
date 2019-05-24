#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from .change_points import detect_cusum, pettitt_test
from .trends import mkt

import sys

from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import CHTest, ADFTest
from pmdarima.arima import CHTest, ADFTest
import scipy.stats as stats
import seaborn as sns

class PerformanceAnalysis:
    """
    Class containing methods for detecting performance changes worth reporting, including
    performance improvements, regressions and other spooky anomalies.
    """

    def __init__(self, performance: pd.DataFrame):
        """
        Initialize the performance analysis

        :param performance: 1-dimensional performance data
        """
        self.performance = performance

    def detect_anomalies_threshold(self, window: int = 10, threshold: float = 0.1):
        """
        Identifies performance anomalies as such revisions, for which performance increases/decreases
        by more than a relative threshold (default is 10%) from the preceeding revisions (default is 10).

        :param window: number of preceeding commits to compare current performance against
        :param threshold: critical performance change threshold
        :return:
        """
        means = self.performance.shift(1).rolling(window=window).mean()
        change = (self.performance - means) / self.performance
        change = change.abs()
        return change > threshold

    def detect_anomalies_cic(self, window: int = 10, z: float = 5.0):
        """
        Identifies performance anomalies as such revisions, for which performance is not contained in
        a confidence interval (CI) specified by preceeding commits.
        The default z-score is 5, the window size is 10.

        :param window: number of preceeding commits to compare current performance against
        :param z: z-score for the confidence interval
        :return:
        """
        roll = self.performance.rolling(window=window)
        ci = lambda s: s[-1] <= np.mean(s[:-1]) + z * np.std(s[:-1]) and s[-1] >= np.mean(s[:-1]) - z * np.std(s[:-1])
        sigs = roll.apply(ci, raw=True)
        return sigs != 1.0

    def detect_anomalies_sig(self, window = 10, p = 0.05):
        """
        Identifies performance anomalies as such revisions, for which performance is [statistically] significantly
        different compared to preceeding revisions using the Mann-Whitney-U test (Wilcoxon rank sum test).
        The default window size is 10, the significance niveau is p = 0.05.

        :param window: number of preceeding commits to compare current performance against
        :param p: significance level
        :return:
        """
        roll = self.performance.rolling(window=window, center=True)
        sig = lambda s: stats.mannwhitneyu(s[:int(window/2)], s[int(window/2):]).pvalue
        sigs = roll.apply(sig)
        return sigs < p

    def detect_anomalies_cusum(self):
        performance = self.performance.ffill()
        performance = performance.bfill()

        drift = 0.1 * np.abs(np.max(performance) - np.min(performance))
        threshold = np.max(performance)
        cp, cp_start, cp_end, amp = detect_cusum(performance, ending=True, drift=drift, threshold=threshold)
        while len(cp) == 0:
            threshold = threshold / 2.
            cp, cp_start, cp_end, amp = detect_cusum(performance, ending=True, drift=drift, threshold=threshold)
        return cp

class PerformancePatternAnalysis:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def segment(self, performance_col: str, method: str = "cusum", cusum_window=30):
        if method == "version":
            self.data["segment"] = self.data["version"]

        elif method == "merge":
            ts = self.data["message"].str.contains("performance")
            cps = ts[ts == True].index.values#
            segments = []
            for i in range(1, len(cps)):
                segments.append( (cps[i-1], cps[i]) )
            if len(segments) > 0:
                segments.append( (cps[-1], self.data.shape[0]) )
                segments.insert(0, (0, cps[0]))
            else:
                segments.append( (0, self.data.shape[0]) )

            for i, segment in enumerate(segments):
                for j in range(segment[0], segment[1]):
                    self.data.set_value(j, "segment", i)



        elif method == "cusum":
            # TODO parameter tuning
            #self.data[performance_col].ffill(inplace=True)
            #self.data[performance_col].bfill(inplace=True)

            data = self.data[performance_col]
            data = data.dropna()

            drift = 0.05 * np.abs(np.max(data) - np.min(data))
            ptp = lambda x: np.ptp(x[~np.isnan(x)])
            threshold = 5 * np.nanmean(data.rolling(window=cusum_window+1, center=True).std())

            cp, cp_start, cp_end, amp = detect_cusum(data.values, ending=True, drift=drift,
                                                     threshold=threshold)
            #cp = cp.tolist()
            #cp.insert(0, 0)
            #cp.append(data.shape[0]-1)
            #segments = []

            #plt.hist(amp)
            #plt.show()
            """
            cp2 = []#cp
            for i, amplitude in enumerate(amp):
                if amplitude > threshold:
                    cp2.append(cp[i])

            for i in range(1, len(cp2)):
                segments.append((cp2[i - 1], cp2[i]))
            if len(segments) > 0:
                segments.insert(0,  (0, cp2[0]) )
                segments.append((cp2[-1], len(data)))
            else:
                segments.append( (0,len(data)) )

            for i, segment in enumerate(segments):
                for j in range(segment[0], segment[1]):
                    self.data.set_value(j, "segment", i)
            """

            #rel_amps = []
            #for c in cp:
            #    amp = (data.values[c] - data.values[c-1])/data.values[c-1]
            #    rel_amps.append(np.abs(amp))
            #if any(i >= 1.1 for i in rel_amps):
            #    print(performance_col, cp, rel_amps)
             #   print(data.values[cp])
            return np.abs(amp / np.mean(data))



    def test_for_seasonality(self, performance_col: str) -> pd.DataFrame:
        """
        Checks whether the signal provided contains a seasonal component
        using the Canova-Hansen-Test (1995) for seasonal stability

        References
        - https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html#pmdarima.arima.CHTest

        :return: bool (True if seasonal, Else otherwise)
        """

        def ch_test(xs):
            xs = xs.values

            ms = np.arange(2, xs.shape[0])

            ms2 = list(map(lambda m: CHTest(m).estimate_seasonal_differencing_term(xs), ms))
            ms = list(zip(ms, ms2))
            msd = dict(ms)
            keyz = list(msd.keys())
            ms = list(filter(lambda k: msd[k] == 1, keyz))
            return {"sample_size": xs.shape[0], "seasonality_m": ms}

        df = self.data

        df = df.sort_values(by=["timestamp"])
        df['index'] = np.arange(len(df))

        df.dropna(subset=[performance_col], inplace=True)
        seasonality = df.groupby("segment").agg({performance_col: ch_test})
        return seasonality[performance_col].apply(pd.Series)

    def test_for_trends(self, performance_col: str):
        """
        Applies the Mann-Kendall-Test and reports significance + effect size

        :param performance: name of the column in the DataFrame to look for trends for
        :return: DataFrame with trend information
        """
        #print("TEST: trends")
        df = self.data

        df = df.sort_values(by=["timestamp"])
        df['index'] = np.arange(len(df))

        df.dropna(subset=[performance_col], inplace=True)
        trends = df.groupby("segment").agg({performance_col: mkt})

        return trends[performance_col].apply(pd.Series)

    def test_for_stationarity(self, performance_col: str, alpha = 0.05):
        """
        Checks whether the signal provided is stationary by using the augmented Dickey-Fuller-Test

        References:
        - https://machinelearningmastery.com/time-series-data-stationary-python/

        :param alpha:
        :return:
        """
        #print("TEST: stationarity")

        df = self.data.sort_values(by=["timestamp"])
        df['index'] = np.arange(len(df))
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df = df.dropna()


        def stationatity_check(xs):
            xs = xs.values
            try:
                r = adfuller(xs)[1]
            except:
                r = np.nan
            return {"sample_size": xs.shape[0], "stationary": r <= 0.05, "stationary_p": r}

        stationarity = df.groupby("segment").agg({performance_col: stationatity_check})
        stationarity = stationarity[performance_col].apply(pd.Series)
        return stationarity

    def test_for_change_points(self, performance_col: str, alpha = 0.05) -> int:
        """
        Detection of statistically significant change-points in a time-series using the Pettitt's test.
        This method bisects the signal as long as the test finds significant change-points, and returns
        the total number of change-points found.

        :param alpha: significance level (default is 0.05)
        :return: number of statistically significant change-points
        """

        df = self.data.sort_values(by=["timestamp"])
        df['index'] = np.arange(len(df))

        cp = df.groupby("segment").agg({performance_col: pettitt_test})
        cp = cp[performance_col].apply(pd.Series)
        return cp

    def apply_tests(self, performance_col: str, visualize: bool = False):
        df = self.data
        #df.ffill(inplace=True)
        #df.bfill(inplace=True)
        stationarity = self.test_for_stationarity(performance_col)
        trends = self.test_for_trends(performance_col)
        #changepoints = p.test_for_change_points(performance_col)

        merged = pd.merge(stationarity, trends, on='segment')
        #merged = pd.merge(merged, changepoints, on='segment')

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        classes = ["stationary", "trendy", "trendstat"]
        stationary = merged[
            np.all([merged["stationary"] == True, merged["trend"] == False], axis=0)]
        trendy = merged[
            np.all([merged["stationary"] == False, merged["trend"] == True], axis=0)]
        disruptive = merged[
            np.all([merged["stationary"] == True, merged["trend"] == True], axis=0)]

        df["category"] = "unclassified"
        df.loc[df["segment"].isin(set(stationary.index.values.tolist())), "category"] = "stationary"
        df.loc[df["segment"].isin(set(trendy.index.values.tolist())), "category"] = "trendy"
        df.loc[df["segment"].isin(set(disruptive.index.values.tolist())), "category"] = "trendstat"

        if visualize:
            parledde = {"unclassified": (0.647, 0.584, 0.552), "trendstat": (0.921, 0.350, 0.0), "trendy": (0.231, 0.686, 0.031), "stationary": (0.921, 0.098, 0.7)}
            sns.set_style("darkgrid")
            df.sort_values(by="timestamp")
            df["index"] = np.arange(0, df.shape[0])

            versions = list(set(df["segment"].values.tolist()))
            xses = []
            maxy = np.max(df[performance_col])
            for version in versions:
                f = df.loc[df["segment"] == version, "index"].values
                if f.shape[0] > 0:
                    minf = min(f)
                    if minf > 215:
                        plt.axvline(minf, color="black", linewidth=0.5)
                    #plt.text(minf, maxy, version, rotation=45, fontsize=8, color="#aeb4bf",horizontalalignment='left',verticalalignment='bottom')
                if len(versions) > 1:
                    if version == versions[1]:
                        break
            #, hue="category", palette=parledde,
            #sns.scatterplot(data=df.iloc[215:], x="index", y=performance_col, marker="x", s=25)
            #plt.legend()
            #plt.xlabel("revision")
            #plt.ylabel("execution time [s]")
            #plt.axes().set_aspect(16./9)
            #plt.draw()
            #plt.savefig("{}.pdf".format(col), bbox_inches="tight")
            #plt.clf()
            #print("dhfhf")