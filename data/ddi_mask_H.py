from rdkit import Chem
from rdkit.Chem import BRICS
import dill
import numpy as np
"""NDCList {'A01A': {'CC(=O)OC1=CC=CC=C1C(O)=O',
  '[F-].[Na+]',
  '[H][C@@]12C[C@@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'},
 'A02A': {'[MgH2]', '[OH-].[OH-].[Mg++]'},
 'A02B': {'CC1=C(OCC(F)(F)F)C=CN=C1CS(=O)C1=NC2=CC=CC=C2N1',
  'COC1=C(OC)C(CS(=O)C2=NC3=C(N2)C=C(OC(F)F)C=C3)=NC=C1',
  'COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C'},
  ....}
"""
NDCList = dill.load(open('./idx2SMILES.pkl', 'rb'))
voc = dill.load(open('./voc_final.pkl', 'rb'))
med_voc = voc['med_voc']

fraction = []
for k, v in med_voc.idx2word.items():
    tempF = set()

    for SMILES in NDCList[v]:
        try:
            m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
            for frac in m:
                tempF.add(frac)
        except:
            pass

    fraction.append(tempF)

fracSet = []
for i in fraction:
    fracSet += i
fracSet = list(set(fracSet))

ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        ddi_matrix[i, fracSet.index(frac)] = 1

dill.dump(ddi_matrix, open('ddi_mask_H.pkl', 'wb'))
dill.dump(fracSet, open('substructure_smiles.pkl', 'wb'))
