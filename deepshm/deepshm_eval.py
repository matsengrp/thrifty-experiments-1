import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("deepshm_predictions.csv.gz")

mutabilities = df['mutability'].to_numpy()
branch_lengths = df['branch_length'].to_numpy()
scaled_muts = mutabilities * branch_lengths
true_muts = [0 if x == '-' else 1 for x in df['mutation']]

auroc = roc_auc_score(true_muts, scaled_muts)
auprc = average_precision_score(true_muts, scaled_muts)


subs_df = df[df['mutation']!='-']
ncorr=0
for i,row in subs_df.iterrows():
    nt = row['mutation']
    if np.argmax(row[list('ACGT')]) == 'ACGT'.index(nt):
        ncorr += 1
sub_acc = ncorr/subs_df.shape[0]


g = df.groupby('pcp_index')
pcp_rprec = []
for ipcp in g.groups.keys():
    pcp_df = g.get_group(ipcp).sort_values('mutability', ascending=False)
    nmuts = pcp_df[pcp_df['mutation']!='-'].shape[0]
    top_df = pcp_df.head(nmuts)
    nmatches = top_df[top_df['mutation']!='-'].shape[0]
    pcp_rprec.append(nmatches/nmuts)
    
rprec = np.mean(pcp_rprec)    


print('substitution accuracy:', sub_acc)
print('AUROC:', auroc)
print('AUPRC:', auprc)
print('R-precision:', rprec)
