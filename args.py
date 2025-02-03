from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP',
                        choices=['ACM-DBLP', 'cora', 'foursquare-twitter', 'phone-email', 'Douban'],
                        help='datasets: ACM-DBLP; cora; foursquare-twitter; phone-email; Douban')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2,
                        choices=[0.2], help='training ratio: 0.2')
    parser.add_argument('--use_attr', dest='use_attr', default=False, action='store_true',
                        help='use input node attributes')

    # Device settings
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu', help='use GPU')

    # Model settings
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=128, help='output dimension')

    # Loss settings
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.9, help='weight of gw distance')
    parser.add_argument('--gamma_p', dest='gamma_p', type=float, default=1e-2, help='entropy regularization parameter')
    parser.add_argument('--in_iter', dest='in_iter', type=int, default=5, help='number of inner iterations')
    parser.add_argument('--out_iter', dest='out_iter', type=int, default=10, help='number of outer iterations')

    # Training settings
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')

    # Experiment settings
    parser.add_argument('--init_threshold_lambda', dest='init_threshold_lambda', type=float, default=1.0, help='initial sampling threshold (lambda)')

    args = parser.parse_args()
    return args
