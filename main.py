"""
Train / evaluate ADGT on Human3.6M.

ADGT: Enhancing 3D human pose estimation with attention-driven graph-transformers
(J. Vis. Commun. Image Represent. 118, 2026).

See README.md for data preparation.
"""
import os
import random
import time
from os import path

import numpy as np
import torch
import torch.nn as nn
from progress.bar import Bar
from torch.utils.data import DataLoader

from common.arguments import parse_args
from common.data import PoseGenerator, fetch, load_h36m
from common.h36m_dataset import TEST_SUBJECTS, TRAIN_SUBJECTS
from common.loss import auc, mpjpe, p_mpjpe, pck
from common.utils import AverageMeter, save_ckpt
from models import ADGT


def build_model(args, device):
    model = ADGT(dim_model=args.dim_model, n_layer=args.n_layer, n_pts=16, n_head=args.n_head,
                 n_register=args.n_register, h_ca=args.h_ca, mlp_ratio=args.mlp_ratio,
                 dropout=args.dropout, attn_drop=args.attn_drop, drop_path=args.drop_path,
                 hops=tuple(args.hops)).to(device)
    print('==> Total parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    return model


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k.replace('module.', '', 1): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    print('==> Loaded checkpoint {} (epoch {} | MPJPE {:.2f})'.format(ckpt_path, ckpt['epoch'], ckpt['error']))


def train(loader, model, optimizer, device, args):
    model.train()
    torch.set_grad_enabled(True)
    epoch_loss = AverageMeter()
    bar = Bar('Train', max=len(loader))

    for targets_3d, inputs_2d, _ in loader:
        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        outputs_3d = model(inputs_2d)

        optimizer.zero_grad()
        loss = mpjpe(outputs_3d, targets_3d)  # Eq. 18
        loss.backward()
        if args.max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss.update(loss.item(), targets_3d.size(0))
        bar.suffix = '({}/{}) Total: {} | ETA: {} | Loss: {:.4f}'.format(
            bar.index + 1, len(loader), bar.elapsed_td, bar.eta_td, epoch_loss.avg)
        bar.next()
    bar.finish()
    return epoch_loss.avg


@torch.no_grad()
def evaluate(loader, model, device, desc='Eval '):
    """Returns MPJPE (P1, mm), P-MPJPE (P2, mm), AUC and PCK (%)."""
    model.eval()
    meters = {k: AverageMeter() for k in ('p1', 'p2', 'auc', 'pck')}
    bar = Bar(desc, max=len(loader))

    for targets_3d, inputs_2d, _ in loader:
        outputs_3d = model(inputs_2d.to(device)).cpu()
        outputs_3d[:, :1, :] = 0  # root joint at the origin
        n = targets_3d.size(0)

        meters['p1'].update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, n)
        meters['p2'].update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, n)
        n_joints = n * targets_3d.size(1)
        meters['auc'].update(auc(outputs_3d * 1000.0, targets_3d * 1000.0) * 100, n_joints)
        meters['pck'].update(pck(outputs_3d * 1000.0, targets_3d * 1000.0) * 100, n_joints)

        bar.suffix = '({}/{}) MPJPE: {:.2f} | P-MPJPE: {:.2f} | AUC: {:.2f} | PCK: {:.2f}'.format(
            bar.index + 1, len(loader), *(meters[k].avg for k in ('p1', 'p2', 'auc', 'pck')))
        bar.next()
    bar.finish()
    return [float(meters[k].avg) for k in ('p1', 'p2', 'auc', 'pck')]


def make_loader(subjects, dataset, keypoints, args, shuffle, action_filter=None):
    data = PoseGenerator(*fetch(subjects, dataset, keypoints, action_filter))
    return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)


def run_evaluation(args, model, dataset, keypoints, device):
    """Per-action results, averaged over actions (Tables 1-2 of the paper)."""
    actions = dataset.define_actions()
    results = np.zeros((len(actions), 4))
    for i, action in enumerate(actions):
        loader = make_loader(TEST_SUBJECTS, dataset, keypoints, args, shuffle=False, action_filter=[action])
        results[i] = evaluate(loader, model, device, desc='{:<12}'.format(action))
        print('{:<12} P1 {:.2f} | P2 {:.2f}'.format(action, results[i, 0], results[i, 1]))
    mean = results.mean(axis=0)
    print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(mean[0]))
    print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(mean[1]))
    print('AUC: {:.2f} | PCK: {:.2f}'.format(mean[2], mean[3]))


def main():
    args = parse_args()
    print('==> Using settings {}'.format(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('==> Loading dataset...')
    dataset, keypoints = load_h36m(args.data_path, args.keypoints)
    model = build_model(args, device)

    if args.evaluate:
        load_checkpoint(model, args.evaluate)
        run_evaluation(args, model, dataset, keypoints, device)
        return

    os.makedirs(args.checkpoint, exist_ok=True)
    train_loader = make_loader(TRAIN_SUBJECTS, dataset, keypoints, args, shuffle=True)
    valid_loader = make_loader(TEST_SUBJECTS, dataset, keypoints, args, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = path.join(args.checkpoint, 'log.txt')
    with open(log_path, 'w') as f:
        f.write('epoch\tlr\tloss_train\tmpjpe\tp_mpjpe\tauc\tpck\n')

    best = None
    for epoch in range(args.epochs):
        lr_now = args.lr * args.lr_gamma ** (epoch // args.lr_decay_epochs)
        for group in optimizer.param_groups:
            group['lr'] = lr_now
        print('\nEpoch: {} | LR: {:.8f}'.format(epoch + 1, lr_now))
        loss = train(train_loader, model, optimizer, device, args)
        p1, p2, auc_, pck_ = evaluate(valid_loader, model, device)

        with open(log_path, 'a') as f:
            f.write('{}\t{:.8f}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch + 1, lr_now, loss, p1, p2, auc_, pck_))

        state = {'epoch': epoch + 1, 'lr': lr_now, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'error': p1}
        if best is None or p1 < best:
            best = p1
            save_ckpt(state, args.checkpoint, suffix='best')
        if (epoch + 1) % args.snapshot == 0:
            save_ckpt(state, args.checkpoint)


if __name__ == '__main__':
    main()
