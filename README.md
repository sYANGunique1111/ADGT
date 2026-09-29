# ADGT

Official implementation of:

**ADGT: Enhancing 3D human pose estimation with attention-driven graph-transformers**
Shuo Yang, Anh Tuan Luu, Xuan Son Nguyen, Aymeric Histace, Bart Jansen, Hichem Sahli — *Journal of Visual Communication and Image Representation* 118 (2026) 104829

ADGT is a frame-based 2D-to-3D pose lifting network that runs a GCN and a Transformer **in parallel** on shared hidden states, then fuses them with a query-key mechanism.

## Method overview

Each of the `L` ADGT layers (`models/adgt.py: ADGTLayer`) applies three components (Sec. 3 of the paper):

- **HS-GCN** (`HSGCN`): Hop-wise Scalable GCN. One graph convolution per hop distance (1, 2, 3 hops), re-weighted per joint by scaling factors computed by attention between the hop features and the layer input, then summed.
- **RET** (`RET`): Register-based Enhancement for Transformers. Learnable register tokens are appended to the joint tokens for self-attention and discarded afterwards, so they absorb excess global information and let the joint tokens focus on local detail (mostly helps high-variance joints such as hands and feet).
- **ALFE** (`ALFE`): Attention-based Local Feature Extractor. The RET (global) output is the query, the HS-GCN (local) output supplies keys and values, so the global features select the local features they need. `Fusion` merges the result with the global embedding, followed by the layer skip connection.

The network is a linear embedding, `L` ADGT layers, and a linear regressor (`ADGT`).

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

Human3.6M is not redistributed here. Prepare it following the standard [VideoPose3D DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) instructions and place under `data/`:

1. `data_3d_h36m.npz` — 3D ground-truth poses.
2. `data_2d_h36m_cpn_ft_h36m_dbb.npz` — CPN-detected 2D keypoints (or `data_2d_h36m_gt.npz` for ground-truth 2D input; pass `--keypoints gt`).

Training uses subjects S1, S5, S6, S7, S8; evaluation uses S9, S11.

## Training

```bash
./train.sh
```

Defaults match the paper: 5 layers, 96 channels, 3 hops, 8 registers, 100 epochs, batch size 256, Adam, single GPU. Learning rate: 8e-4 decayed by 0.95 every 2 epochs for CPN input; for ground-truth 2D input use `--keypoints gt --lr 0.001 --lr_gamma 0.9`. Checkpoints and `log.txt` are written to `checkpoints/adgt_h36m_cpn/`.


## Citation

```bibtex
@article{yang2026adgt,
  title   = {ADGT: Enhancing 3D human pose estimation with attention-driven graph-transformers},
  author  = {Yang, Shuo and Luu, Anh Tuan and Nguyen, Xuan Son and Histace, Aymeric and Jansen, Bart and Sahli, Hichem},
  journal = {Journal of Visual Communication and Image Representation},
  volume  = {118},
  pages   = {104829},
  year    = {2026}
}
```

## Acknowledgements

This implementation builds on [GraFormer](https://github.com/Graformer/GraFormer), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) and [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).
