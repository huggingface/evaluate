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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Might be nice to print to log instead? Happens when we drop closed class.
warnings.filterwarnings(action="ignore", category=UserWarning)
# When we divide by 0 in log
np.seterr(divide="ignore")

# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)

logs = logging.getLogger(__name__)
logs.setLevel(logging.INFO)
logs.propagate = False

if not logs.handlers:

    Path("./log_files").mkdir(exist_ok=True)

    # Logging info to log file
    file = logging.FileHandler("./log_files/npmi.log")
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

_NUM_BATCHES = 500


class nPMI:
    # TODO: Expand beyond pairwise
    def __init__(
        self,
        vocab_counts_df,
        tokenized_df,
        tokenized_col_name="tokenized_text",
    ):
        logs.info("Initiating npmi class.")
        logs.info("vocab is")
        logs.info(vocab_counts_df)
        self.vocab_counts_df = vocab_counts_df
        logs.info("tokenized is")
        self.tokenized_df = tokenized_df
        logs.info(self.tokenized_df)
        self.tokenized_col_name = tokenized_col_name
        # self.mlb_list holds num batches x num_sentences
        self.mlb_list = []

    def binarize_words_in_sentence(self):
        logs.info("Creating co-occurrence matrix for PMI calculations.")
        batches = np.linspace(0, self.tokenized_df.shape[0], _NUM_BATCHES).astype(int)
        i = 0
        # Creates list of size (# batches x # sentences)
        while i < len(batches) - 1:
            # Makes a sparse matrix (shape: # sentences x # words),
            # with the occurrence of each word per sentence.
            mlb = MultiLabelBinarizer(classes=self.vocab_counts_df.index)
            logs.info(
                "%s of %s sentence binarize batches." % (str(i), str(len(batches)))
            )
            # Returns series: batch size x num_words
            mlb_series = mlb.fit_transform(
                self.tokenized_df[self.tokenized_col_name][batches[i]:batches[i + 1]]
            )
            i += 1
            self.mlb_list.append(mlb_series)

    def calc_cooccurrences(self, subgroup, subgroup_idx):
        initialize = True
        coo_df = None
        # Big computation here!  Should only happen once.
        logs.info(
            "Approaching big computation! Here, we binarize all words in the sentences, making a sparse matrix of sentences."
        )
        if not self.mlb_list:
            self.binarize_words_in_sentence()
        for batch_id in range(len(self.mlb_list)):
            logs.info(
                "%s of %s co-occurrence count batches"
                % (str(batch_id), str(len(self.mlb_list)))
            )
            # List of all the sentences (list of vocab) in that batch
            batch_sentence_row = self.mlb_list[batch_id]
            # Dataframe of # sentences in batch x vocabulary size
            sent_batch_df = pd.DataFrame(batch_sentence_row)
            # logs.info('sent batch df is')
            # logs.info(sent_batch_df)
            # Subgroup counts per-sentence for the given batch
            subgroup_df = sent_batch_df[subgroup_idx]
            subgroup_df.columns = [subgroup]
            # Remove the sentences where the count of the subgroup is 0.
            # This way we have less computation & resources needs.
            subgroup_df = subgroup_df[subgroup_df > 0]
            logs.info("Removing 0 counts, subgroup_df is")
            logs.info(subgroup_df)
            mlb_subgroup_only = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
            logs.info("mlb subgroup only is")
            logs.info(mlb_subgroup_only)
            # Create cooccurrence matrix for the given subgroup and all words.
            logs.info("Now we do the T.dot approach for co-occurrences")
            batch_coo_df = pd.DataFrame(mlb_subgroup_only.T.dot(subgroup_df))

            # Creates a batch-sized dataframe of co-occurrence counts.
            # Note these could just be summed rather than be batch size.
            if initialize:
                coo_df = batch_coo_df
            else:
                coo_df = coo_df.add(batch_coo_df, fill_value=0)
            logs.info("coo_df is")
            logs.info(coo_df)
            initialize = False
        logs.info("Returning co-occurrence matrix")
        logs.info(coo_df)
        return pd.DataFrame(coo_df)

    @staticmethod
    def calc_paired_metrics(subgroup_pair, subgroup_npmi_dict):
        """
        Calculates nPMI metrics between paired subgroups.
        Special handling for a subgroup paired with itself.
        :param subgroup_npmi_dict: vocab, pmi, and npmi for each subgroup.
        :return:

        Args:
            subgroup_pair:
        """
        paired_results_dict = {"npmi": {}, "pmi": {}, "count": {}}
        # Canonical ordering. This is done previously, but just in case...
        subgroup1, subgroup2 = sorted(subgroup_pair)
        vocab_cooc_df1, pmi_df1, npmi_df1 = subgroup_npmi_dict[subgroup1]
        logs.info("vocab cooc")
        logs.info(vocab_cooc_df1)
        if subgroup1 == subgroup2:
            shared_npmi_df = npmi_df1
            shared_pmi_df = pmi_df1
            shared_vocab_cooc_df = vocab_cooc_df1
        else:
            vocab_cooc_df2, pmi_df2, npmi_df2 = subgroup_npmi_dict[subgroup2]
            logs.info("vocab cooc2")
            logs.info(vocab_cooc_df2)
            # Note that lsuffix and rsuffix should not come into play.
            shared_npmi_df = npmi_df1.join(
                npmi_df2, how="inner", lsuffix="1", rsuffix="2"
            )
            shared_pmi_df = pmi_df1.join(pmi_df2, how="inner", lsuffix="1", rsuffix="2")
            shared_vocab_cooc_df = vocab_cooc_df1.join(
                vocab_cooc_df2, how="inner", lsuffix="1", rsuffix="2"
            )
            shared_vocab_cooc_df = shared_vocab_cooc_df.dropna()
            shared_vocab_cooc_df = shared_vocab_cooc_df[
                shared_vocab_cooc_df.index.notnull()
            ]
            logs.info("shared npmi df")
            logs.info(shared_npmi_df)
            logs.info("shared vocab df")
            logs.info(shared_vocab_cooc_df)
        npmi_bias = (
            shared_npmi_df[subgroup1 + "-npmi"] - shared_npmi_df[subgroup2 + "-npmi"]
        )
        paired_results_dict["npmi-bias"] = npmi_bias.dropna()
        paired_results_dict["npmi"] = shared_npmi_df.dropna()
        paired_results_dict["pmi"] = shared_pmi_df.dropna()
        paired_results_dict["count"] = shared_vocab_cooc_df.dropna()
        return paired_results_dict

    def calc_metrics(self, subgroup):
        # Index of the subgroup word in the sparse vector
        subgroup_idx = self.vocab_counts_df.index.get_loc(subgroup)
        logs.info("Calculating co-occurrences...")
        df_coo = self.calc_cooccurrences(subgroup, subgroup_idx)
        vocab_cooc_df = self.set_idx_cols(df_coo, subgroup)
        logs.info(vocab_cooc_df)
        logs.info("Calculating PMI...")
        pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
        logs.info(pmi_df)
        logs.info("Calculating nPMI...")
        npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
        logs.info(npmi_df)
        return vocab_cooc_df, pmi_df, npmi_df

    def set_idx_cols(self, df_coo, subgroup):
        """
        :param df_coo: Co-occurrence counts for subgroup, length is num_words
        :return:
        """
        count_df = df_coo.set_index(self.vocab_counts_df.index)
        count_df.columns = [subgroup + "-count"]
        count_df[subgroup + "-count"] = count_df[subgroup + "-count"].astype(int)
        return count_df

    def calc_PMI(self, vocab_cooc_df, subgroup):
        """
        # PMI(x;y) = h(y) - h(y|x)
        #          = h(subgroup) - h(subgroup|word)
        #          = log (p(subgroup|word) / p(subgroup))
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        """
        # Calculation of p(subgroup)
        subgroup_prob = self.vocab_counts_df.loc[subgroup]["proportion"]
        # Calculation of p(subgroup|word) = count(subgroup,word) / count(word)
        # Because the inidices match (the vocab words),
        # this division doesn't need to specify the index (I think?!)
        p_subgroup_g_word = (
            vocab_cooc_df[subgroup + "-count"] / self.vocab_counts_df["count"]
        )
        logs.info("p_subgroup_g_word is")
        logs.info(p_subgroup_g_word)
        pmi_df = pd.DataFrame()
        pmi_df[subgroup + "-pmi"] = np.log(p_subgroup_g_word / subgroup_prob)
        # Note: A potentially faster solution for adding count, npmi,
        # can be based on this zip idea:
        # df_test['size_kb'],  df_test['size_mb'], df_test['size_gb'] =
        # zip(*df_test['size'].apply(sizes))
        return pmi_df.dropna()

    def calc_nPMI(self, pmi_df, vocab_cooc_df, subgroup):
        """
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        #                                           = -log(p(word|subgroup)p(word))
        """
        p_word_g_subgroup = vocab_cooc_df[subgroup + "-count"] / sum(
            vocab_cooc_df[subgroup + "-count"]
        )
        p_word = pmi_df.apply(
            lambda x: self.vocab_counts_df.loc[x.name]["proportion"], axis=1
        )
        normalize_pmi = -np.log(p_word_g_subgroup * p_word)
        npmi_df = pd.DataFrame()
        npmi_df[subgroup + "-npmi"] = pmi_df[subgroup + "-pmi"] / normalize_pmi
        return npmi_df.dropna()
