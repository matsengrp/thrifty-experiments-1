from netam import framework
import torch
from epam.molevol import reshape_for_codons, build_mutation_matrices, codon_probs_of_mutation_matrices

# Not sure how to handle this yet, as a function input or what
site_count = 500

def trim_seqs_to_codon_boundary_and_max_len(seqs):
    """Assumes that all sequences have the same length"""
    max_len = site_count - site_count % 3
    return [seq[:min(len(seq) - len(seq) % 3, max_len)] for seq in seqs]

def prepare_pcp_df(pcp_df, crepe):
    """
    Trim the sequences to codon boundaries and add the rates and substitution probabilities.
    """
    pcp_df["parent"] = trim_seqs_to_codon_boundary_and_max_len(pcp_df["parent"])
    pcp_df["child"] = trim_seqs_to_codon_boundary_and_max_len(pcp_df["child"])
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]].reset_index(drop=True)
    ratess, cspss = framework.trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    pcp_df["rates"] = ratess
    pcp_df["subs_probs"] = cspss
    return pcp_df

def codon_probs_of_parent_scaled_rates_and_sub_probs(parent_idxs, scaled_rates, sub_probs):
    """
    Compute the probabilities of mutating to various codons for a parent sequence. 
    
    This uses the same machinery as we use for fitting the DNSM, but we stay on
    the codon level rather than moving to syn/nonsyn changes.
    """
    # This is from `aaprobs_of_parent_scaled_rates_and_sub_probs`:
    mut_probs = 1.0 - torch.exp(-scaled_rates)
    parent_codon_idxs = reshape_for_codons(parent_idxs)
    codon_mut_probs = reshape_for_codons(mut_probs)
    codon_sub_probs = reshape_for_codons(sub_probs)
    
    # This is from `aaprob_of_mut_and_sub`:
    mut_matrices = build_mutation_matrices(parent_codon_idxs, codon_mut_probs, codon_sub_probs)
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    return codon_probs