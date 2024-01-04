import numpy as np
import torch
import pandas as pd
from netam import framework, models
from shmple import SHMoof
from shmple.evaluate import r_precision
from shmple.util import randomly_concretize_dna_string_pair
from sklearn.metrics import roc_auc_score, average_precision_score

#
# script settings
#
pcp_fname = "/fh/fast/matsen_e/shared/bcr-mut-sel/pcps/tang-deepshm-oof_pcp_2023-11-30_MASKED.csv"
out_fname = "performance_metrics.csv"
max_seq_length = 500
my_device = "cpu"
model_dir = "../pretrained"
model_names = [
    "hs_shmoof", #"hs_fivemer",
    "hs_cnn_sml",
    "hs_cnn_med_orig", "hs_cnn_med",
    "hs_cnn_lrg",
    "hs_cnn_4k", "hs_cnn_4k_k13",
    "hs_cnn_8k"
]
print('')
print("Evaluating performances on:", pcp_fname)
print("Device selected:", my_device)
print("Models directory:", model_dir)
print("==========")


df = pd.read_csv(pcp_fname,index_col=0)
df = df.iloc[[len(x) <=max_seq_length for x in df.parent]]

parents = df['parent'].to_numpy()
childs = df['child'].to_numpy()

# derive mutation flags for each PCP in the dataset
seqs_true_muts = [ np.array([nt1!=nt2 for nt1,nt2 in zip(p, c)]) for p, c in zip(parents, childs) ]

# combine all mutation flags in the dataset in to one array (for AUROC and AUPRC)
all_true_muts = np.concatenate(seqs_true_muts)

# compute "branch lengths" for all PCPs
mut_freqs=[]
for parent, child in zip(df['parent'].to_numpy(), df['child'].to_numpy()):
    numer = np.sum(np.array(list(parent))!=np.array(list(child)))
    denom = np.sum([1 if nt!='N' else 0 for nt in parent])
    mut_freqs.append(numer/denom)


results_df = pd.DataFrame(columns=['model','AUROC','AUPRC','r-prec'])
model_col=[]
auroc_col=[]
auprc_col=[]
rprec_col=[]

print("Processing SHMoof model")
model = SHMoof()
outs = model.predict_mutabilities(parents, df['branch_length'].to_numpy())
seqs_mutabilities = [o.squeeze() for o in outs]
all_mutabilities = np.concatenate(seqs_mutabilities)
model_col.append('shmoof')
auroc_col.append(roc_auc_score(all_true_muts, all_mutabilities))
auprc_col.append(average_precision_score(all_true_muts, all_mutabilities))
rprec_col.append(r_precision(seqs_true_muts, seqs_mutabilities))


print("Processing netam models...")
for modname in model_names:
    print(" >", modname)
    crepe_prefix = f"{model_dir}/{modname}"
    assert framework.crepe_exists(crepe_prefix)
    crepe = framework.load_crepe(crepe_prefix, device = my_device)
    outs = crepe(parents)
    seqs_mutabilities = [o[:len(p)].detach().numpy()*mf for p,o,mf in zip(parents, outs, mut_freqs)]
    all_mutabilities = np.concatenate(seqs_mutabilities)
    model_col.append(modname)
    auroc_col.append(roc_auc_score(all_true_muts, all_mutabilities))
    auprc_col.append(average_precision_score(all_true_muts, all_mutabilities))
    rprec_col.append(r_precision(seqs_true_muts, seqs_mutabilities))

results_df['model'] = model_col
results_df['AUROC'] = auroc_col
results_df['AUPRC'] = auprc_col
results_df['r-prec'] = rprec_col
print(results_df)
results_df.to_csv(out_fname,index=False)
print('')
print("Results written to:", out_fname)
print('')
