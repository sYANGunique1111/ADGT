import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ADGT: train / evaluate on Human3.6M')

    # data
    parser.add_argument('--data_path', default='data', type=str, help='folder with data_3d_h36m.npz and the 2D keypoint file')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str,
                        help='2D keypoints: cpn_ft_h36m_dbb (detected) or gt (ground-truth projections)')
    parser.add_argument('--num_workers', default=0, type=int)

    # run
    parser.add_argument('--checkpoint', default='checkpoints/adgt_h36m', type=str, help='output directory')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILE', help='checkpoint to evaluate (skips training)')
    parser.add_argument('--snapshot', default=20, type=int, help='save a checkpoint every N epochs')
    parser.add_argument('--seed', default=54862, type=int)

    # optimisation
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=8e-4, type=float, help='paper: 1e-3 for GT 2D input, 8e-4 for CPN input')
    parser.add_argument('--lr_decay_epochs', default=2, type=int, help='epochs between learning-rate decays')
    parser.add_argument('--lr_gamma', default=0.95, type=float, help='paper: 0.9 for GT 2D input, 0.95 for CPN input')
    parser.add_argument('--max_norm', default=False, action='store_true', help='clip gradient norm to 1')

    # model
    parser.add_argument('--dim_model', default=96, type=int, help='channels C')
    parser.add_argument('--n_layer', default=5, type=int, help='number of ADGT layers L')
    parser.add_argument('--n_head', default=4, type=int, help='heads of the RET transformer')
    parser.add_argument('--n_register', default=8, type=int, help='register tokens M')
    parser.add_argument('--h_ca', default=4, type=int, help='heads of ALFE')
    parser.add_argument('--mlp_ratio', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--attn_drop', default=0.0, type=float)
    parser.add_argument('--drop_path', default=0.1, type=float)
    parser.add_argument('--hops', default=[1, 2, 3], nargs='+', type=int, help='hop distances of HS-GCN (K = len)')

    return parser.parse_args()
