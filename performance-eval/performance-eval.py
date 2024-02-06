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
out_fname = "new_models.csv"
max_seq_length = 500
my_device = "cpu"
new_model_names = [
    "rsshmoof-shmoof_small",
    "hs_cnn_ind_med",
    "cnn_ind_lrg-shmoof_small",
    "cnn_ind_med-shmoof_small",
]
print('')
print("Evaluating performances on:", pcp_fname)
print("Device selected:", my_device)
print("==========")


def seq_to_vector(seq: str) -> np.ndarray:
    """Converts a string of characters to a vector representation."""
    lookup = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": -1
    }
    f = lambda x: lookup[x]
    return np.array(list(map(f, seq)))


df = pd.read_csv(pcp_fname,index_col=0)
df = df.iloc[[len(x) <=max_seq_length for x in df.parent]]

parents = df['parent'].to_numpy()
childs = df['child'].to_numpy()

# derive mutation flags for each PCP in the dataset
seqs_true_muts = [ np.array([nt1!=nt2 for nt1,nt2 in zip(p, c)]) for p, c in zip(parents, childs) ]

# combine all mutation flags in the dataset into one array (for AUROC and AUPRC)
all_true_muts = np.concatenate(seqs_true_muts)

# combine all substitutions in the dataset into one array for substitution accuracy calculation
all_mut_idxs = np.nonzero(all_true_muts)
all_true_subs = np.take(np.concatenate([seq_to_vector(s) for s in childs]), all_mut_idxs, axis=0)

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
subacc_col=[]

print("SHMple results")
model_col.append('SHMple')
auroc_col.append(0.8741229695370192)
auprc_col.append(0.1053406710659186)
rprec_col.append(0.074874420383321)
subacc_col.append(0.4760342137823444)

print("Processing SHMoof model")
model = SHMoof()
muts, subs = model.predict_mutabilities_and_substitutions(parents, df['branch_length'].to_numpy())
seqs_mutabilities = [o.squeeze() for o in muts]
all_mutabilities = np.concatenate(seqs_mutabilities)
all_sub_rates = np.concatenate(subs, axis=0)
subs_predict = np.take(all_sub_rates, all_mut_idxs, axis=0).argmax(axis=-1).squeeze()
model_col.append('shmoof')
auroc_col.append(roc_auc_score(all_true_muts, all_mutabilities))
auprc_col.append(average_precision_score(all_true_muts, all_mutabilities))
rprec_col.append(r_precision(seqs_true_muts, seqs_mutabilities))
subacc_col.append((all_true_subs==subs_predict).sum().item()/subs_predict.shape[0])

netam_modname = "hs_cnn_4k_k13"
print(f"Processing netam model: {netam_modname}")
crepe_prefix = f"../pretrained/{netam_modname}"
assert framework.crepe_exists(crepe_prefix)
crepe = framework.load_crepe(crepe_prefix, device = my_device)
outs = crepe(parents)
seqs_mutabilities = [o[:len(p)].detach().numpy()*mf for p,o,mf in zip(parents, outs, mut_freqs)]
all_mutabilities = np.concatenate(seqs_mutabilities)
model_col.append(netam_modname)
auroc_col.append(roc_auc_score(all_true_muts, all_mutabilities))
auprc_col.append(average_precision_score(all_true_muts, all_mutabilities))
rprec_col.append(r_precision(seqs_true_muts, seqs_mutabilities))
subacc_col.append(-1)

for new_modname in new_model_names:
    print(f"Processing new model: {new_modname}")
    crepe_prefix = f"../train/trained_models/{new_modname}"
    assert framework.crepe_exists(crepe_prefix)
    crepe = framework.load_crepe(crepe_prefix, device = my_device)
    rates, csp_logits = crepe(parents)
    seqs_mutabilities = [o[:len(p)].detach().numpy()*mf for p,o,mf in zip(parents, rates, mut_freqs)]
    all_mutabilities = np.concatenate(seqs_mutabilities)

    csps = torch.softmax(csp_logits, dim=2).numpy()
    seqs_sub_rates = [c[:len(p)] for p,c in zip(parents, csps)]
    all_sub_rates = np.concatenate(seqs_sub_rates, axis=0)
    subs_predict = np.take(all_sub_rates, all_mut_idxs, axis=0).argmax(axis=-1).squeeze()

    model_col.append(new_modname)
    auroc_col.append(roc_auc_score(all_true_muts, all_mutabilities))
    auprc_col.append(average_precision_score(all_true_muts, all_mutabilities))
    rprec_col.append(r_precision(seqs_true_muts, seqs_mutabilities))
    subacc_col.append((all_true_subs==subs_predict).sum().item()/subs_predict.shape[0])

results_df['model'] = model_col
results_df['AUROC'] = auroc_col
results_df['AUPRC'] = auprc_col
results_df['r-prec'] = rprec_col
results_df['sub_acc'] = subacc_col
print(results_df)
results_df.to_csv(out_fname,index=False)
print('')
print("Results written to:", out_fname)
print('')
