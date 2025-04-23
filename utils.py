
from rdkit.Chem import QED, Crippen, Descriptors

def encode_smiles(smiles, max_len=64):
    vec = [ord(c) for c in smiles[:max_len]]
    return vec + [0] * (max_len - len(vec))

def penalized_logp(mol):
    log_p = Crippen.MolLogP(mol)
    qed_score = QED.qed(mol)
    mol_wt = Descriptors.MolWt(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    penalty = -2.0 if (mol_wt > 500 or log_p > 5 or h_donors > 5 or h_acceptors > 10) else 0.0
    return qed_score + log_p + penalty
