import os.path

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time
import json

from args import *
from utils import *
from model import *

if __name__ == '__main__':
    args = make_args()

    # check compatibility between dataset and use_attr
    if args.dataset == 'noisy-cora1-cora2':
        assert args.use_attr is True, 'noisy-cora1-cora2 requires using node attributes'
    elif args.dataset == 'foursquare-twitter' or args.dataset == 'phone-email':
        assert args.use_attr is False, f'{args.dataset} does not have node attributes'

    if os.path.exists(f"settings/{args.dataset}.json"):
        print(f"Loading settings from settings/{args.dataset}.json")
        with open(f"settings/{args.dataset}.json", 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                print(f"Setting {key} to {value}")
                setattr(args, key, value)
    else:
        print(f"Using default arguments from command line")

    # load data and build networkx graphs
    print("Loading data...", end=" ")
    edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data(f"datasets/{args.dataset}", args.ratio,
                                                                           args.use_attr, dtype=np.float64)
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1, G2 = build_nx_graph(edge_index1, anchor1, x1), build_nx_graph(edge_index2, anchor2, x2)
    print("Done")

    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, args.dataset, args.ratio, dtype=np.float64)
    if x1 is None:
        x1 = rwr1
    else:
        x1 = np.concatenate([x1, rwr1], axis=1)
    if x2 is None:
        x2 = rwr2
    else:
        x2 = np.concatenate([x2, rwr2], axis=1)

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)

    # build PyG Data objects
    G1_tg = build_tg_graph(edge_index1, x1, rwr1, dtype=torch.float64).to(device)
    G2_tg = build_tg_graph(edge_index2, x2, rwr2, dtype=torch.float64).to(device)
    n1, n2 = G1_tg.x.shape[0], G2_tg.x.shape[0]
    args.gw_weight = args.alpha / (1 - args.alpha) * min(n1, n2) ** 0.5

    out_dir = 'logs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    writer = SummaryWriter(save_path(args.dataset, out_dir, args.use_attr))

    max_hits_list = defaultdict(list)
    max_mrr_list = []
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        model = MLP(input_dim=G1_tg.x.shape[1],
                    hidden_dim=args.hidden_dim,
                    output_dim=args.out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FusedGWLoss(G1_tg, G2_tg, anchor1, anchor2,
                                gw_weight=args.gw_weight,
                                gamma_p=args.gamma_p,
                                init_threshold_lambda=args.init_threshold_lambda,
                                in_iter=args.in_iter,
                                out_iter=args.out_iter,
                                total_epochs=args.epochs).to(device)

        print("Training...")
        max_hits = defaultdict(int)
        max_mrr = 0
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            optimizer.zero_grad()
            out1, out2 = model(G1_tg, G2_tg)
            loss, similarity, threshold_lambda = criterion(out1=out1, out2=out2)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}', end=', ')

            # testing
            with torch.no_grad():
                model.eval()
                hits, mrr = compute_metrics(-similarity, test_pairs)
                s_entropy = torch.sum(-similarity * torch.log(similarity))
                end = time.time()
                print(f's_entropy: {s_entropy:.4f}, threshold_lambda: {threshold_lambda * n1 * n2:.4f}, '
                      f'{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}')

                max_mrr = max(max_mrr, mrr.cpu().item())
                for key, value in hits.items():
                    max_hits[key] = max(max_hits[key], value.cpu().item())

                writer.add_scalar('Loss', loss.item(), epoch)
                writer.add_scalar('MRR', mrr, epoch)
                for key, value in hits.items():
                    writer.add_scalar(f'Hits/Hits@{key}', value, epoch)

        for key, value in max_hits.items():
            max_hits_list[key].append(value)
        max_mrr_list.append(max_mrr)

        print("")

    max_hits = {}
    max_hits_std = {}
    for key, value in max_hits_list.items():
        hits_list = np.array([val for val in value])
        max_hits[key] = hits_list.mean()
        max_hits_std[key] = hits_list.std()
    max_mrr = np.array(max_mrr_list).mean()
    max_mrr_std = np.array(max_mrr_list).std()

    hparam_dict = {
        'dataset': args.dataset,
        'use_attr': args.use_attr,
        'epochs': args.epochs,
        'lr': args.lr,
        'alpha': args.alpha,
        'gamma_p': args.gamma_p,
        'threshold_lambda': threshold_lambda.cpu().item(),
    }
    writer.add_hparams(hparam_dict, {'hparam/MRR': max_mrr,
                                     'hparam/std_MRR': max_mrr_std,
                                     **{f'hparam/Hits@{key}': value for key, value in max_hits.items()},
                                     **{f'hparam/std_Hits@{key}': value for key, value in max_hits_std.items()}})
