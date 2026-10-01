# Fitting and choosing radiative-transfer emulators for the retrieval

Manuscript, code and run records for a study of emulators of radiative-transfer codes that are fitted on the code's
outputs but used through an inverse, the retrieval of surface reflectance from top-of-atmosphere radiance.

[Paper (PDF)](paper/retrieval_training.pdf) · [LaTeX source](paper/retrieval_training.tex)

## Findings

The retrieval error of an emulator has an exact closed form in its errors in path radiance, transmission and spherical
albedo. To first order it is a Mahalanobis norm of those errors whose metric depends on the state only through the
transmission and the albedo, and the forward and retrieval rankings of two emulators differ by a change of measure with
density proportional to t²/q⁴. Fitting on the retrieval error is therefore regression under a known, unbounded
covariate shift; the rate that a weighted certificate buys is sharp, and the price of the weighting in effective sample
size can be computed before any training.

On the EMIT atmospheric-correction table (23,313 states, 285 bands, ten splits), a network trained on the square root
of the retrieval weights floored at a hundredth of the median transmission lowers the radiance error from 0.350% to
0.322% and the 95th percentiles of the retrieval error by 32 to 44% on average, each on every split. Two failed
evaluations of the code in the table, with zero transmission in most bands, make every weighted network worse than the
plain one. Refitting a convex stack of emulators on the retrieval error lowers its tail by a quarter on 39 of 40
held-out halves.

On the libRadtran benchmark of Mazid and Rishe (Remote Sensing 18:1826, 2026; 50,000 states, 13 Sentinel-2 bands, the
release's own split), the same weights lower the radiance error from 0.91% to 0.11% and the all-band 95th percentile of
the retrieval error from 4.63 to 0.35 points over ten seeds, on every seed. In the release's metrics they bring the
mean absolute error to 0.00051, against 0.00186 for the release's best model, at a root-mean-square error 26% higher,
all of it from the spherical albedo. On an out-of-distribution split, whose test states lie above the 0.85 quantile of
the aerosol optical depth or of the water-vapour column, the weighted networks also lower the root-mean-square error,
from 0.0086 to 0.0062, on every one of five seeds.

| emulator on the libRadtran benchmark | RMSE | MAE | SMAPE [%] | radiance error [%] | retrieval error, 95th percentile [points] |
|---|---:|---:|---:|---:|---:|
| pKANrtm, published | 0.00619 | 0.00186 | 3.91 | | |
| sRTMnet, published | 0.00691 | 0.00211 | 4.28 | | |
| network, plain | 0.00711 | 0.00254 | 4.30 | 0.91 | 4.63 |
| network, floored weights (u = 0.1) | 0.00772 | 0.00061 | 1.85 | 0.13 | 0.36 |
| network, flattened weights (κ = 1/2, u = 0.01) | 0.00780 | 0.00051 | 1.52 | 0.11 | 0.35 |
| network, 0.3 plain + 0.7 flattened | 0.00630 | 0.00162 | 3.11 | 0.55 | 2.28 |

Test scores on the release's split. The first two rows are from Table 3 of the release; the network rows are means over
ten seeds. The retrieval columns need an emulator's predictions on every test state and are given for the networks
only: the mean relative error of the top-of-atmosphere reflectance, and the 95th percentile over all bands of the error
of the surface reflectance retrieved at ρ = 0.7. Section 5.4 of the paper has the full table, with every baseline of the
release, the error of each coefficient and the out-of-distribution split.

## Layout

- `paper/`: the manuscript (`retrieval_training.tex`), its bibliography, the generated benchmark table and the compiled
  PDF.
- `code/`: the drivers of every experiment, the scoring, and the scripts that produce each table and figure from the
  records. `bench_data.py` loads the libRadtran benchmark; `p3_pkan.py` and `p3_pkan2.py` are its drivers.
- `results/`: one JSON record per run, each with the SHA-256 of its input arrays and of its driver. `results/dgx/`
  holds the runs of the network, kernel and stacking experiments, `results/pilot/` the network runs on the EMIT table as
  provided, and `results/pkan2/` the benchmark runs of the paper: ten seeds on the release's split, five on the
  out-of-distribution split, three for each ablation, three of a wider network and one of the wider network on the
  out-of-distribution split, written by `code/p3_pkan2.py`.
  `results/pkan/` holds the values the release publishes in its Table 3 and five seeds of a smaller network (four hidden
  layers of 384 units, 80 epochs) written by `code/p3_pkan.py`. The `driver_sha256` of a record is that of the driver
  as it ran; the published drivers differ from those in their docstrings and local paths, so their digests differ.
- `figures/`: the figures of the paper.
- `data/README.md`: where the two tables come from.

## Reproduction

Regenerating the tables and figures from the records needs Python 3.10 or later with numpy and matplotlib:

```sh
python -m pip install -r requirements.txt
python code/make_pkan_table.py           # paper/table_pkan*.tex, the published benchmark, and the numbers of its section
sh code/make_network_table.sh            # the network table, written into paper/retrieval_training.tex
python code/plot_paper_figures.py        # figures/fig_weight and figures/fig_frontier
latexmk -pdf -cd paper/retrieval_training.tex
```

The scripts named `*_numbers.py` and `summarize_*.py` print the numbers quoted in the text from the same records. The
training drivers also need scipy and PyTorch, and the data described in `data/README.md`.

## Citation

```bibtex
@misc{shmalo2026retrieval,
  author       = {Yitzchak Shmalo},
  title        = {Fitting and choosing radiative-transfer emulators for the retrieval},
  year         = {2026},
  howpublished = {\url{https://github.com/yspennstate/rt-emulators-for-retrieval}}
}
```

## License

MIT, see `LICENSE`. The data tables are not covered: see `data/README.md`.
