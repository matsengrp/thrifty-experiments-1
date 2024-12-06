## DeepSHM

This directory contains data and script to reproduce the DeepSHM performance metrics in the manuscript.

The DeepSHM outputs on the `val_curatedShmoofNotbigNoN` data is `deepshm_predictions.csv.gz`.
The CSV file has the following columns:
|column name | description|
|---|---|
| `pcp_index` | index of the parent-child pair in `val_curatedShmoofNotbigNoN` |
| `branch_length` | number of mutations divided by sequence length |
| `site` | nucleotide site |
| `mutability` | DeepSHM mutation frequency |
| `A` | DeepSHM conditional substitution probability to A |
| `C` | DeepSHM conditional substitution probability to C |
| `G` | DeepSHM conditional substitution probability to G |
| `T` | DeepSHM conditional substitution probability to T |
| `mutation` | if a mutation occur at the site in the parent-child pair, indicates the resulting nucleotide; otherwise, `-` |

Each row in the CSV file corresponds to a site in a parent-child pair.
Note that for sites at the edge of the sequence, which cannot be the center of a 15-mer, the result is taken from the average of all possible combinations of nucleotides for the out-of-bounds positions in the 15-mer.

Run the script

    python deepshm_eval.py

to compute substitution accuracy, AUROC, AUPRC, and R-precision.
