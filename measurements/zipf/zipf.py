# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw
import streamlit as st
from scipy.stats import ks_2samp
from scipy.stats import zipf as zipf_lib

# treating inf values as NaN as well

pd.set_option("use_inf_as_na", True)

logs = logging.getLogger(__name__)
logs.setLevel(logging.INFO)
logs.propagate = False

if not logs.handlers:

    Path("./log_files").mkdir(exist_ok=True)

    # Logging info to log file
    file = logging.FileHandler("./log_files/zipf.log")
    fileformat = logging.Formatter("%(asctime)s:%(message)s")
    file.setLevel(logging.INFO)
    file.setFormatter(fileformat)

    # Logging debug messages to stream
    stream = logging.StreamHandler()
    streamformat = logging.Formatter("[data_measurements_tool] %(message)s")
    stream.setLevel(logging.WARNING)
    stream.setFormatter(streamformat)

    logs.addHandler(file)
    logs.addHandler(stream)


class Zipf:
    def __init__(self, vocab_counts_df=pd.DataFrame(), CNT="count", PROP="prop"):
        self.vocab_counts_df = vocab_counts_df
        self.alpha = None
        self.xmin = None
        self.xmax = None
        self.fit = None
        self.ranked_words = {}
        self.uniq_counts = []
        self.uniq_ranks = []
        self.uniq_fit_counts = None
        self.term_df = None
        self.pvalue = None
        self.ks_test = None
        self.distance = None
        self.fit = None
        self.predicted_zipf_counts = None
        if not self.vocab_counts_df.empty:
            logs.info("Fitting based on input vocab counts.")
            self.calc_fit(vocab_counts_df, CNT, PROP)
            logs.info("Getting predicted counts.")
            self.predicted_zipf_counts = self.calc_zipf_counts(vocab_counts_df)

    def load(self, zipf_dict):
        self.set_xmin(zipf_dict["xmin"])
        self.set_xmax(zipf_dict["xmax"])
        self.set_alpha(zipf_dict["alpha"])
        self.set_ks_distance(zipf_dict["ks_distance"])
        self.set_p(zipf_dict["p-value"])
        self.set_unique_ranks(zipf_dict["uniq_ranks"])
        self.set_unique_counts(zipf_dict["uniq_counts"])

    def calc_fit(self, vocab_counts_df, CNT, PROP):
        """
        Uses the powerlaw package to fit the observed frequencies to a zipfian distribution.
        We use the KS-distance to fit, as that seems more appropriate that MLE.
        :param vocab_counts_df:
        :return:
        """
        self.vocab_counts_df = vocab_counts_df
        # TODO: These proportions may have already been calculated.
        vocab_counts_df[PROP] = vocab_counts_df[CNT] / float(sum(vocab_counts_df[CNT]))
        rank_column = vocab_counts_df[CNT].rank(
            method="dense", numeric_only=True, ascending=False
        )
        vocab_counts_df["rank"] = rank_column.astype("int64")
        observed_counts = vocab_counts_df[CNT].values
        # Note another method for determining alpha might be defined by
        # (Newman, 2005): alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
        self.fit = powerlaw.Fit(observed_counts, fit_method="KS", discrete=True)
        # This should probably be a pmf (not pdf); using discrete=True above.
        # original_data=False uses only the fitted data (within xmin and xmax).
        # pdf_bin_edges: The portion of the data within the bin.
        # observed_pdf: The probability density function (normalized histogram)
        # of the data.
        pdf_bin_edges, observed_pdf = self.fit.pdf(original_data=False)
        # See the 'Distribution' class described here for info:
        # https://pythonhosted.org/powerlaw/#powerlaw.Fit.pdf
        theoretical_distro = self.fit.power_law
        # The probability density function (normalized histogram) of the
        # theoretical distribution.
        predicted_pdf = theoretical_distro.pdf()
        # !!!! CRITICAL VALUE FOR ZIPF !!!!
        self.alpha = theoretical_distro.alpha
        # Exclusive xmin: The optimal xmin *beyond which* the scaling regime of
        # the power law fits best.
        self.xmin = theoretical_distro.xmin
        self.xmax = theoretical_distro.xmax
        self.distance = theoretical_distro.KS()
        self.ks_test = ks_2samp(observed_pdf, predicted_pdf)
        self.pvalue = self.ks_test[1]
        logs.info("KS test:")
        logs.info(self.ks_test)

    def set_xmax(self, xmax):
        """
        xmax is usually None, so we add some handling to set it as the
        maximum rank in the dataset.
        :param xmax:
        :return:
        """
        if xmax:
            self.xmax = int(xmax)
        elif self.uniq_counts:
            self.xmax = int(len(self.uniq_counts))
        elif self.uniq_ranks:
            self.xmax = int(len(self.uniq_ranks))

    def get_xmax(self):
        """
        :return:
        """
        if not self.xmax:
            self.set_xmax(self.xmax)
        return self.xmax

    def set_p(self, p):
        self.p = int(p)

    def get_p(self):
        return int(self.p)

    def set_xmin(self, xmin):
        self.xmin = xmin

    def get_xmin(self):
        if self.xmin:
            return int(self.xmin)
        return self.xmin

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def get_alpha(self):
        return float(self.alpha)

    def set_ks_distance(self, distance):
        self.distance = float(distance)

    def get_ks_distance(self):
        return self.distance

    def calc_zipf_counts(self, vocab_counts_df):
        """
        The fit is based on an optimal xmin (minimum rank)
        Let's use this to make count estimates for the zipf fit,
        by multiplying the fitted pmf value by the sum of counts above xmin.
        :return: array of count values following the fitted pmf.
        """
        # TODO: Limit from above xmin to below xmax, not just above xmin.
        counts = vocab_counts_df[CNT]
        self.uniq_counts = list(pd.unique(counts))
        self.uniq_ranks = list(np.arange(1, len(self.uniq_counts) + 1))
        logs.info(self.uniq_counts)
        logs.info(self.xmin)
        logs.info(self.xmax)
        # Makes sure they are ints if not None
        xmin = self.get_xmin()
        xmax = self.get_xmax()
        self.uniq_fit_counts = self.uniq_counts[xmin + 1 : xmax]
        pmf_mass = float(sum(self.uniq_fit_counts))
        zipf_counts = np.array(
            [self.estimate_count(rank, pmf_mass) for rank in self.uniq_ranks]
        )
        return zipf_counts

    def estimate_count(self, rank, pmf_mass):
        return int(round(zipf_lib.pmf(rank, self.alpha) * pmf_mass))

    def set_unique_ranks(self, ranks):
        self.uniq_ranks = ranks

    def get_unique_ranks(self):
        return self.uniq_ranks

    def get_unique_fit_counts(self):
        return self.uniq_fit_counts

    def set_unique_counts(self, counts):
        self.uniq_counts = counts

    def get_unique_counts(self):
        return self.uniq_counts

    def set_axes(self, unique_counts, unique_ranks):
        self.uniq_counts = unique_counts
        self.uniq_ranks = unique_ranks

    # TODO: Incorporate this function (not currently using)
    def fit_others(self, fit):
        st.markdown(
            "_Checking log likelihood ratio to see if the data is better explained by other well-behaved distributions..._"
        )
        # The first value returned from distribution_compare is the log likelihood ratio
        better_distro = False
        trunc = fit.distribution_compare("power_law", "truncated_power_law")
        if trunc[0] < 0:
            st.markdown("Seems a truncated power law is a better fit.")
            better_distro = True

        lognormal = fit.distribution_compare("power_law", "lognormal")
        if lognormal[0] < 0:
            st.markdown("Seems a lognormal distribution is a better fit.")
            st.markdown("But don't panic -- that happens sometimes with language.")
            better_distro = True

        exponential = fit.distribution_compare("power_law", "exponential")
        if exponential[0] < 0:
            st.markdown("Seems an exponential distribution is a better fit. Panic.")
            better_distro = True

        if not better_distro:
            st.markdown("\nSeems your data is best fit by a power law. Celebrate!!")
