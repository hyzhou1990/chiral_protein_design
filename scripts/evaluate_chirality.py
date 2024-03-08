import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, calc_dihedral

def evaluate_chirality(structure, ref_structure):
    parser = PDBParser()
    model = parser.get_structure("design", structure)
    ref_model = parser.get_structure("reference", ref_structure)
    
    chirality_scores = []
    for chain in model:
        for residue1, residue2 in zip(chain, ref_model[chain.id]):
            if residue1.resname != residue2.resname:
                continue
            
            atoms1 = [residue1["N"], residue1["CA"], residue1["C"], residue1["O"]]
            atoms2 = [residue2["N"], residue2["CA"], residue2["C"], residue2["O"]]
            if any(atom is None for atom in atoms1 + atoms2):
                continue
            
            dihedral = calc_dihedral(
                atoms1[0].get_vector(),
                atoms1[1].get_vector(),
                atoms1[2].get_vector(),
                atoms1[3].get_vector(),
            )
            ref_dihedral = calc_dihedral(
                atoms2[0].get_vector(),
                atoms2[1].get_vector(),
                atoms2[2].get_vector(),
                atoms2[3].get_vector(),
            )
            chirality_scores.append(dihedral - ref_dihedral)
    
    chirality_scores = pd.DataFrame({"true_score": chirality_scores})
    chirality_scores.to_csv("results/chirality_scores.csv", index=False)
    
    return np.mean(chirality_scores), np.std(chirality_scores)