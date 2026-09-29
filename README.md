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
release's own split), the same weights lower the radiance error from 0.81% to 0.17% and the all-band 95th percentile of
the retrieval error from 3.21 to 0.48 points over five seeds, on every seed. In the release's metrics they bring the
mean absolute error to 0.00069, against 0.00186 for the release's best model, at a root-mean-square error 28% higher,
all of it from the spherical albedo.

## Layout

- `paper/`: the manuscript (`retrieval_training.tex`), its bibliography, the generated benchmark table and the compiled
  PDF.
- `code/`: the drivers of every experiment, the scoring, and the scripts that produce each table and figure from the
  records. `bench_data.py` loads the libRadtran benchmark; `p3_pkan.py` and `p3_pkan2.py` are its drivers.
- `results/`: one JSON record per run, each with the SHA-256 of its input arrays and of its driver. `results/dgx/`
  holds the runs of the network, kernel and stacking experiments, `results/pilot/` the network runs on the EMIT table as
  provided, and `results/pkan/` the benchmark runs with the release's published table.
- `figures/`: the figures of the paper.
- `data/README.md`: where the two tables come from.

## Reproduction

Regenerating the tables and figures from the records needs Python 3.10 or later with numpy and matplotlib:

```sh
python -m pip install -r requirements.txt
python code/make_pkan_table.py           # paper/table_pkan.tex, the published benchmark
sh code/make_network_table.sh            # the network table, written into paper/retrieval_training.tex
python code/plot_paper_figures.py        # figures/fig_weight and figures/fig_frontier
latexmk -pdf -cd paper/retrieval_training.tex
```

The scripts named `*_numbers.py` and `summarize_*.py` print the numbers quoted in the text from the same records. The
training drivers also need scipy and PyTorch, and the data described in `data/README.md`.

## License

MIT, see `LICENSE`.
