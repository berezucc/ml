# ml

Machine learning experiments, course work, and notes.

## Highlights

[**Kaggle Playground S6E5 - Predicting F1 Pit Stops**](experiments/kaggle-f1-pit-stops/)

5-model ensemble (LightGBM, bagged CatBoost, XGBoost, sklearn MLP, PyTorch
residual MLP) with OOF target encoding and a logistic stacker. Public
leaderboard 0.9545 (blend with public anchor); own-model ceiling 0.9492.
See [`NOTES.md`](experiments/kaggle-f1-pit-stops/NOTES.md) for the writeup.

## Layout

| Directory | Contents |
|---|---|
| `experiments/` | Self-contained Kaggle and side projects |
| `coursework/` | Homework and labs from ML courses (TMU) |
| `notes/` | Topic notes and cheatsheets |

`implementations/` and `paper-notes/` are reserved for future work.

## Related repos

Polished standalone projects live in their own repositories:

- [speculative-decoding](https://github.com/berezucc/speculative-decoding) - speculative decoding on Apple M2 Max; 1.16x over greedy in PyTorch+MPS, 2.69x with MLX
- [hft-jump-diffusion](https://github.com/berezucc/hft-jump-diffusion) - optimal execution under Semi-Markov and Hawkes jump-diffusion models with PDE solvers, Monte Carlo, Streamlit dashboard

## Setup

Per-experiment dependencies live in each experiment's own `requirements.txt`.
For example:

```bash
make setup-f1
```

## License

MIT, see [`LICENSE`](LICENSE).
