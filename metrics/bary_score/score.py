from __future__ import absolute_import, division, print_function

from collections import Counter, defaultdict
from math import log

import numpy as np
import ot
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


class BaryScoreMetric:
    def __init__(self, model_name="bert-base-uncased", last_layers=5, use_idfs=True, sinkhorn_ref=0.01):
        """
        BaryScore metric
        :param model_name: model name or path from HuggingFace Librairy
        :param last_layers: last layer to use in the pretrained model
        :param use_idfs: if true use idf costs else use uniform weights
        :param sinkhorn_ref:  weight of the KL in the SD
        """

        self.model_name = model_name
        self.load_tokenizer_and_model()
        n = self.model.config.num_hidden_layers + 1
        assert n - last_layers > 0
        self.layers_to_consider = range(n - last_layers, n)
        self.use_idfs = use_idfs
        self.sinkhorn_ref = sinkhorn_ref
        self.idfs = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_idfs(self, hyps, refs):
        """
        :param hyps: hypothesis list of string sentences has to be computed at corpus level
        :param refs:reference list of string sentences has to be computed at corpus level
        """
        t_hyps = self.tokenizer(hyps)["input_ids"]
        t_refs = self.tokenizer(refs)["input_ids"]
        idf_dict_ref = self.ref_list_to_idf(t_refs)
        idf_dict_hyp = self.ref_list_to_idf(t_hyps)
        idfs_tokenizer = (idf_dict_ref, idf_dict_hyp)
        self.model_ids = idfs_tokenizer
        return idf_dict_hyp, idf_dict_ref

    def ref_list_to_idf(self, input_refs):
        """
        :param input_refs: list of input reference
        :return: idf dictionnary
        """
        idf_count = Counter()
        num_docs = len(input_refs)

        idf_count.update(sum([list(set(i)) for i in input_refs], []))

        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def load_tokenizer_and_model(self):
        """
        Loading and initializing the chosen model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained("{}".format(self.model_name))
        model = AutoModelForMaskedLM.from_pretrained("{}".format(self.model_name))
        model.config.output_hidden_states = True
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def evaluate_batch(self, batch_hyps, batch_refs, idf_hyps=None, idf_ref=None):
        """
        :param batch_hyps: hypothesis list of string sentences
        :param batch_refs: reference list of string sentences
        :param idf_hyps: idfs of hypothesis computed at corpus level
        :param idf_ref: idfs of references computed at corpus level
        :return: dictionnary of scores
        """
        # Extract Embeddings From Pretrained Models
        if isinstance(batch_hyps, str):
            batch_hyps = [batch_hyps]
        if isinstance(batch_refs, str):
            batch_refs = [batch_refs]
        nb_sentences = len(batch_refs)
        baryscores = []
        assert len(batch_hyps) == len(batch_refs)

        if (idf_hyps is None) and (idf_ref is None):
            idf_hyps, idf_ref = self.model_ids

        model = self.model.to(self.device)

        with torch.no_grad():
            # Extract Embeddings From Pretrained Models
            batch_refs = self.tokenizer(batch_refs, return_tensors="pt", padding=True, truncation=True).to(self.device)
            batch_refs_embeddings_ = model(**batch_refs)[-1]

            batch_hyps = self.tokenizer(batch_hyps, return_tensors="pt", padding=True, truncation=True).to(self.device)
            batch_hyps_embeddings_ = model(**batch_hyps)[-1]

            batch_refs_embeddings = [batch_refs_embeddings_[i] for i in list(self.layers_to_consider)]
            batch_hyps_embeddings = [batch_hyps_embeddings_[i] for i in list(self.layers_to_consider)]

            batch_refs_embeddings = torch.cat([i.unsqueeze(0) for i in batch_refs_embeddings])
            batch_refs_embeddings.div_(torch.norm(batch_refs_embeddings, dim=-1).unsqueeze(-1))
            batch_hyps_embeddings = torch.cat([i.unsqueeze(0) for i in batch_hyps_embeddings])
            batch_hyps_embeddings.div_(torch.norm(batch_hyps_embeddings, dim=-1).unsqueeze(-1))

            ref_tokens_id = batch_refs["input_ids"].cpu().tolist()
            hyp_tokens_id = batch_hyps["input_ids"].cpu().tolist()

            # Unbatched BaryScore Prediction
            for index_sentence in tqdm(range(nb_sentences), "BaryScore Progress"):
                dict_score = {}
                ref_ids_idf = batch_refs["input_ids"][index_sentence]
                hyp_idf_ids = batch_hyps["input_ids"][index_sentence]

                ref_tokens = [
                    i
                    for i in self.tokenizer.convert_ids_to_tokens(
                        ref_tokens_id[index_sentence], skip_special_tokens=False
                    )
                    if i != self.tokenizer.pad_token
                ]
                hyp_tokens = [
                    i
                    for i in self.tokenizer.convert_ids_to_tokens(
                        hyp_tokens_id[index_sentence], skip_special_tokens=False
                    )
                    if i != self.tokenizer.pad_token
                ]

                ref_ids = [k for k, w in enumerate(ref_tokens)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens)]

                # With stop words
                ref_idf_i = [idf_ref[i] for i in ref_ids_idf[ref_ids]]
                hyp_idf_i = [idf_hyps[i] for i in hyp_idf_ids[hyp_ids]]

                ref_embedding_i = batch_refs_embeddings[:, index_sentence, ref_ids, :]
                hyp_embedding_i = batch_hyps_embeddings[:, index_sentence, hyp_ids, :]
                measures_locations_ref = ref_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_ref = [np.array(i) for i in measures_locations_ref]
                measures_locations_hyps = hyp_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_hyps = [np.array(i) for i in measures_locations_hyps]

                # ADDED
                measures_locations_ref = [
                    np.array(i) for i in np.array(measures_locations_ref).transpose(1, 0, 2).tolist()
                ]
                measures_locations_hyps = [
                    np.array(i) for i in np.array(measures_locations_hyps).transpose(1, 0, 2).tolist()
                ]

                if self.use_idfs:
                    # Use TF-IDF weights
                    baryscore = self.baryscore(measures_locations_ref, measures_locations_hyps, ref_idf_i, hyp_idf_i)
                else:
                    # Uniform Weights
                    baryscore = self.baryscore(measures_locations_ref, measures_locations_hyps, None, None)

                for key, value in baryscore.items():
                    dict_score["baryscore_{}".format(key)] = value
                baryscores.append(dict_score)
            baryscores_dic = {}
            for k in dict_score.keys():
                baryscores_dic[k] = []
                for score in baryscores:
                    baryscores_dic[k].append(score[k])

        return baryscores_dic

    def baryscore(self, measures_locations_ref, measures_locations_hyps, weights_refs, weights_hyps):
        """
        :param measures_locations_ref: input measure reference locations
        :param measures_locations_hyps: input measure hypothesis locations
        :param weights_refs: references weights in the Wasserstein Barycenters
        :param weights_hyps: hypothesis weights in the Wasserstein Barycenters
        :return:
        """
        if weights_hyps is not None or weights_refs is not None:
            assert weights_refs is not None
            assert weights_hyps is not None
            weights_hyps = np.array([i / sum(weights_hyps) for i in weights_hyps]).astype(np.float64)
            weights_refs = np.array([i / sum(weights_refs) for i in weights_refs]).astype(np.float64)

        self.n_layers = len(measures_locations_ref)
        self.d_bert = measures_locations_ref[0].shape[1]
        # Compute Wasserstein Barycenter
        bary_ref = self.w_barycenter(measures_locations_ref, weights_refs)
        bary_hyp = self.w_barycenter(measures_locations_hyps, weights_hyps)

        # Compute Wasserstein and Sinkhorn Divergence

        C = ot.dist(bary_ref, bary_hyp)
        weights_first_barycenter = np.zeros((C.shape[0])) + 1 / C.shape[0]
        weights_second_barycenter = np.zeros((C.shape[1])) + 1 / C.shape[1]
        wasserstein_distance = ot.emd2(weights_first_barycenter, weights_second_barycenter, C, log=True)[0]
        dic_results = {
            "W": wasserstein_distance,
        }
        for reg in [10, 1, 5, 1, 0.1, 0.5, 0.01, 0.001]:
            wasserstein_sinkhorn = ot.bregman.sinkhorn2(
                weights_first_barycenter, weights_second_barycenter, C, reg=reg, numItermax=10000
            ).tolist()
            if isinstance(wasserstein_sinkhorn, list):
                wasserstein_sinkhorn = wasserstein_sinkhorn[0]  # for POT==0.7.0
            dic_results["SD_{}".format(reg)] = wasserstein_sinkhorn
        return dic_results

    def w_barycenter(self, measures_locations, weights):
        """
        :param measures_locations: location of the discrete input measures
        :param weights: weights of the input measures
        :return: barycentrique distribution
        """
        X_init = np.zeros((measures_locations[0].shape[0], self.d_bert)).astype(np.float64)
        if weights is None:
            measures_weights = [
                np.array([1 / measures_locations[0].shape[0]] * measures_locations[0].shape[0])
            ] * self.n_layers
        else:
            measures_weights = [weights / sum(weights)] * self.n_layers
        b = np.array([1 / measures_locations[0].shape[0]] * measures_locations[0].shape[0]).astype(np.float64)
        mesure_bary = ot.lp.free_support_barycenter(
            measures_locations, measures_weights, X_init, b=b, numItermax=1000, verbose=False
        )
        return mesure_bary

    @property
    def supports_multi_ref(self):
        """
        :return: BaryScore does not support multi ref
        """
        return False
