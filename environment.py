
from rdkit import Chem
from rdkit.Chem import QED, Crippen, AllChem
from utils import encode_smiles, penalized_logp

fragments = ['[*]C', '[*]CC', '[*]N', '[*]O', '[*]c1ccccc1', '[*]C(=O)O']
reaction_smarts = '[*:1].[*:2]>>[*:1][*:2]'
rxn = AllChem.ReactionFromSmarts(reaction_smarts)

class DrugMDP:
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.smiles = 'C[*]'
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return encode_smiles(self.smiles)

    def apply_fragment(self, base_smiles, frag_smiles):
        try:
            mol1 = Chem.MolFromSmiles(base_smiles)
            mol2 = Chem.MolFromSmiles(frag_smiles)
            products = rxn.RunReactants((mol1, mol2))
            newmol = products[0][0]
            Chem.SanitizeMol(newmol)
            return Chem.MolToSmiles(newmol)
        except:
            return None

    def step(self, action):
        frag = fragments[action]
        next_smiles = self.apply_fragment(self.smiles, frag)
        if next_smiles is None:
            return self.get_state(), -1.0, True
        try:
            mol = Chem.MolFromSmiles(next_smiles)
            Chem.SanitizeMol(mol)
        except:
            return self.get_state(), -1.0, True
        self.smiles = next_smiles
        self.steps += 1
        reward = penalized_logp(mol)
        done = self.steps >= self.max_steps
        return self.get_state(), reward, done
