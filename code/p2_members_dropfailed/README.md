# The member emulators refitted without the two failed evaluations

The stacks of the paper combine the six member families of the second paper (cubic ridge, isotropic and
input-scaled Matern kernel ridge, the network, the corrected network, the kernel on the network's features), fitted by
that paper's `emit_campaign.py` at repository commit 1f5d4df. The EMIT table holds two failed evaluations of the code,
states 4011 and 7439: from about 1030 nm on, both fluxes are exactly zero in 197 of the 285 bands and the spherical
albedo is exactly 2 in 107.

The second paper's training-target experiment selects training rows by a named policy (`emit_target_quality.py`) and
leaves the validation and test blocks unchanged. The two files here add one policy to a copy of that repository
(`~/p23/tq_repo_dropfailed` on the machine that ran it) and nothing else:

- `emit_target_quality.py`: policy `drop-failed` removes the training rows whose flux `Y2 + Y3` is exactly zero in more
  than half of the bands. On the ten splits this removes exactly the failed states that fall in the training block
  (both on eight splits, state 7439 alone on splits 104 and 105).
- `tq_lane_dropfailed.py`: the second paper's lane wrapper pointed at that copy, with the policy added to its choices.

Everything else in the copy is the repository at 1f5d4df (line endings of the patched file are LF). The lanes are
`tq_s<seed>_dropfailed_w512` and `tq_s<seed>_dropfailed_w2000`, seeds 101-110, with the second paper's scorer
(`conditioned_reflectance.py`) run on each.
