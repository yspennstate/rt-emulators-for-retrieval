# Data

Neither table is redistributed here.

**EMIT.** The atmospheric-correction lookup table of the EMIT study cited in the paper (23,313 evaluations of a
radiative-transfer code on the 285-band EMIT grid) was supplied by the Jet Propulsion Laboratory. It is available from
the author on request, subject to the permission of the Jet Propulsion Laboratory. The drivers read NumPy exports of its
state matrix and four components from `data/emit/jpl_reg_data`; every record carries the SHA-256 of the arrays it read.

**libRadtran.** The paired 6S and libRadtran correction coefficients of Mazid and Rishe, *Multi-fidelity emulation of
atmospheric correction coefficients with physics-guided Kolmogorov-Arnold networks*, Remote Sensing 18(11):1826, 2026
(doi:10.3390/rs18111826), are public with that paper. `code/bench_data.py` reads the release from the directory named by
the environment variable `DATA_NEW` (subdirectory `pkanrtm`) and uses the release's own state-level split; the records in
`results/pkan/` carry the SHA-256 of the arrays they read.
