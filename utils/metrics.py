import torch
import os
import numpy as np


def compute_distance_matrix(emb1, emb2):
    """
    Compute distance matrix between two sets of embeddings
    :param emb1: node embeddings of graph 1
    :param emb2: node embeddings of graph 2
    :return: distance matrix
    """

    emb1 = emb1 / torch.linalg.norm(emb1, ord=2, axis=1, keepdims=True)
    emb2 = emb2 / torch.linalg.norm(emb2, ord=2, axis=1, keepdims=True)
    dists = 1 - emb1 @ emb2.T

    return dists


def compute_metrics_ltr(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    distances = dissimilarity[test_pairs[:, 0]]
    device = dissimilarity.device

    hits = {}
    ranks = torch.argsort(distances, dim=1)
    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal_hit = ranks == test_pairs[:, 1].view(-1, 1)
    for k in hit_top_ks:
        hits[k] = torch.sum(signal_hit[:, :k]) / test_pairs.shape[0]

    mrr = torch.mean(1 / (torch.where(ranks == test_pairs[:, 1].view(-1, 1))[1] + 1))

    return hits, mrr


def compute_metrics_rtl(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    distances = dissimilarity.T[test_pairs[:, 1]]
    device = dissimilarity.device

    hits = {}
    ranks = torch.argsort(distances, dim=1)
    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal_hit = ranks == test_pairs[:, 0].view(-1, 1)
    for k in hit_top_ks:
        hits[k] = torch.sum(signal_hit[:, :k]) / test_pairs.shape[0]

    mrr = torch.mean(1 / (torch.where(ranks == test_pairs[:, 0].view(-1, 1))[1] + 1))

    return hits, mrr


def compute_metrics(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute metrics for the model (HITS@k, MRR)
    :param dissimilarity: dissimilarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :param hit_top_ks: list of k for HITS@k
    :return:
        hits: HITS@k
        mrr: MRR
    """

    distances1 = dissimilarity[test_pairs[:, 0]]
    distances2 = dissimilarity.T[test_pairs[:, 1]]
    device = dissimilarity.device

    hits = {}

    ranks1 = torch.argsort(distances1, dim=1)
    ranks2 = torch.argsort(distances2, dim=1)

    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    for k in hit_top_ks:
        hits_ltr = torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]
        hits_rtl = torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]
        hits[k] = torch.max(hits_ltr, hits_rtl)

    mrr_ltr = torch.mean(1 / (torch.where(ranks1 == test_pairs[:, 1].view(-1, 1))[1] + 1))
    mrr_rtl = torch.mean(1 / (torch.where(ranks2 == test_pairs[:, 0].view(-1, 1))[1] + 1))
    mrr = torch.max(mrr_ltr, mrr_rtl)

    return hits, mrr


def save_path(dataset, out_dir, use_attr=False):
    if dataset == 'ACM-DBLP':
        dataset = 'ACM-DBLP_attr' if use_attr else 'ACM-DBLP'

    if not os.path.exists(f'{out_dir}'):
        os.makedirs(f'{out_dir}')
    if not os.path.exists(f'{out_dir}/{dataset}_results'):
        os.makedirs(f'{out_dir}/{dataset}_results')
    runs = len([f for f in os.listdir(f'{out_dir}/{dataset}_results') if os.path.isdir(f'{out_dir}/{dataset}_results/{f}')])
    runs_str = str(runs).zfill(3)
    return f'{out_dir}/{dataset}_results/run_{runs_str}'
