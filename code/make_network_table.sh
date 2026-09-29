#!/bin/sh
# Table 1 of the paper: the clean-table runs (DGX, p3_weight_family.py --drop-rows 4011 7439) and the pilot runs on the
# table as provided (Kaggle, retrieval_weighted_pilot.py, lanes p3rwA/p3rwB). The libRadtran records (results/dgx/p3rl)
# must not be passed: they carry the same arm names.
# The clean block reads three record sets: p3cl (floored, mixed and cut weights), p3jc (the joint arms) and p3fc (the
# flattened weights of Shimodaira, bar w_tau^kappa renormalized to mean one). Every difference is taken against the
# plain network of p3cl, which the plain networks of p3fc equal on every split to five decimals (same driver, same
# machine, deterministic training).
PILOT=${PILOT:-results/pilot}
python -X utf8 code/make_network_table.py \
  --block "without the two failed evaluations" results/dgx/p3cl results/dgx/p3jc results/dgx/p3fc $CLEAN_EXTRA \
  --block "the table as provided" "$PILOT" \
  --arms w1_u0.1 w1_u0.01 w1_u0.001 w1_u0.01_sp mix0.2_u0.01 flat0.25_u0.01 flat0.5_u0.01 flat0.25_u0.01_sp \
         flat0.25_u0.001_sp cut-w1_u0.01 weighted_u0.1 weighted_u0.01 weighted_u0.001 joint_u0.01 joint_u0.001 \
  --out-json results/network_table.json --out-tex results/network_table_rows.tex --into paper/retrieval_training.tex
