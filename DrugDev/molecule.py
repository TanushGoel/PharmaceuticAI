import plotly.express as px
import matplotlib.pyplot as plt
import py3Dmol
import networkx as nx
import numpy as np
import pandas as pd
import torch
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import SimilarityMaps, MolToImage, MolsToGridImage
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
TABLE = Chem.GetPeriodicTable()

# ZINC (https://zinc.docking.org/substances/) TODO ~ 37 billion compounds
ZINC = "data/mols/zinc.txt"

# ChEMBL (https://www.ebi.ac.uk/chembl/g/#browse/compounds) - 2,399,743 compounds
CHEMBL = "data/mols/chembl.csv" 
# PubChem (https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) - 116,073,122 compounds
PUBCHEM = "data/mols/pubchem.txt"
# Mcule (https://mcule.com/database/) - 41,566,413 compounds
MCULE = "data/mols/mcule.txt"
# total = 159,924,352 compounds --> expected = 160,039,278 compounds --> duplicates = 114,926 compounds
# TODO: MongoDB for big data queries

VOCAB_SIZE = 120
class MolDS():
    
    def __init__(self):
        self.molecules = pd.read_csv("data/mols/chembl.csv", on_bad_lines="skip", sep=";")["Smiles"].astype(str).unique()
        # self.molecules = list(filter(lambda x: len(smile) > 2 and Molecule(smile).rof(), self.molecules))
        
        # with open("data/mols/pubchem.txt", "r") as pubchem_file:
        #     smiles = [line.split("\t")[1].strip() for line in pubchem_file]

        # with open("data/mols/mcule.txt", "r") as mcule_file:
        #     smiles = [line.split("\t")[0].strip() for line in mcule_file)]
                    
        print(f"{len(self):,} molecules loaded")

    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        return self.molecules[idx]

WIDTH, HEIGHT = 1000, 500
class Molecule:
    
    def __init__(self, smiles: str):
        self.smiles = smiles
        
    def __repr__(self):
        return self.smiles
    
    def __len__(self):
        return len(self.smiles)
    
    def to_mol(self):
        return Chem.MolFromSmiles(self.smiles)
    
    def to_file(self, ftype="pdb"): # cif, sdf, pdb
        Chem.MolToMolFile(self.to_mol(), f"temp/mol.{ftype}")
    
    def to_pdbqt(self):
        self.to_file("sdf")
        subprocess.run("obabel temp/mol.sdf -O temp/mol.pdbqt", shell=True, capture_output=True, text=True)
        # centroid = Chem.rdMolTransforms.ComputeCentroid(Chem.SDMolSupplier('temp/mol.sdf', removeHs=False)[0].GetConformer())
        coords = self.get_coords()
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        return [min_coords[0], min_coords[1], min_coords[2]], [max_coords[0], max_coords[1], max_coords[2]]
    
    # Lipinski's “Rule of Five” - constraints for drug-like character
    def rof(self):
        mol = self.to_mol()
        if mol:
            donors = Descriptors.NumHDonors(mol) # number of hydrogen bond donors (NH and OH)
            acceptors = Descriptors.NumHAcceptors(mol) # number of hydrogen bond acceptors (N and O)
            weight = Descriptors.MolWt(mol) # molecular weight
            logp = Descriptors.MolLogP(mol) # octanol-water partition coefficient --> hydrophobicity
            return donors <= 5 and acceptors <= 10 and weight <= 500 and logp <= 5
        return False
    
    def prime(self): 
        mol = self.to_mol()
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(mol, maxIters=2500)
        return Chem.MolToMolBlock(mol), mol
    
    def to_graph(self):
        _, mol = self.prime()
        conf = mol.GetConformer()
        g = nx.Graph()

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            coords = conf.GetAtomPosition(idx)
            x, y, z = map(float, [coords.x, coords.y, coords.z])
            g.add_node(idx, x=x, y=y, z=z, element=atom.GetSymbol())

        for bond in mol.GetBonds():
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), weight=bond.GetBondTypeAsDouble())

        return g
    
    def get_coords(self):
        return np.array([[node['x'], node['y'], node['z']] for _, node in list(self.to_graph().nodes(data=True))])
    
    @torch.no_grad()
    def to_tensor(self):
        def scale(data):        
            col_min = data.min(dim=0).values
            col_max = data.max(dim=0).values
            scaled_data = (data - col_min) / (col_max - col_min)
            return scaled_data
        g = self.to_graph()
        edges = list(g.edges(data=True))
        edge_index = torch.tensor([(n1, n2) for n1, n2, _ in edges], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([edge['weight'] for _, _, edge in edges], dtype=torch.long)
        node_coords = torch.tensor([[node['x'], node['y'], node['z']] for _, node in list(g.nodes(data=True))], dtype=torch.float)
        node_elements = torch.tensor([TABLE.GetAtomicNumber(node['element']) for _, node in list(g.nodes(data=True))], dtype=torch.float) / 18.
        node_features = torch.cat([scale(node_coords), node_elements.unsqueeze(1)], dim=1)
        return node_features, edge_index, edge_weight
    # https://medium.com/the-modern-scientist/graph-neural-networks-series-part-3-node-embedding-36613cc967d5
    
    def plot_graph(self):
        graph = self.to_graph()
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=12, font_color='black', font_weight='bold')
        plt.axis('off')
        plt.show()

    def plot(self):
        x, y, z, elements = zip(*[(node["x"], node['y'], node['z'], node['element']) for _, node in list(self.to_graph().nodes(data=True))])
        px.scatter_3d(x=x, y=y, z=z, color=elements, opacity=0.75, width=WIDTH, height=HEIGHT).show()

    def show(self, style='stick'):
        mblock, _ = self.prime()
        view = py3Dmol.view(width=WIDTH, height=HEIGHT)
        view.addModel(mblock, 'mol')
        view.setStyle({style:{}})
        view.setBackgroundColor('#28282B') #212225
        view.zoomTo()
        view.show()
        
    def draw2(self):
        return Draw.MolToMPL(self.to_mol())
        
    def draw1(self):
        return MolToImage(self.to_mol())

    @classmethod
    def draw_list(cls, mols, mpr=5):
        return MolsToGridImage([m.to_mol() for m in mols], molsPerRow=mpr)

    def analyze(self):
        m = self.to_mol()
        print("Molecule Weight:", Chem.Descriptors.MolWt(m), "g/mol")
        print("LogP:", Chem.Descriptors.MolLogP(m))
        print("TPSA:", Chem.Descriptors.TPSA(m))

    # recreation of rdkit.DataStructs.TanimotoSimilarity
    def similarity(self, mol): # fraction of fingerprints the set of two molecules have in common
        m1 = self.to_mol()
        fp1 = set(AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=4096, bitInfo={}).GetOnBits())

        m2 = mol.to_mol()
        fp2 = set(AllChem.GetMorganFingerprintAsBitVect(m2, radius=2, nBits=4096, bitInfo={}).GetOnBits())

        common = fp1 & fp2
        combined = fp1 | fp2
        return len(common)/len(combined)

    def similarity_map(self, mol):
        m1 = self.to_mol()
        m2 = mol.to_mol()

        fingerprint = SimilarityMaps.GetMorganFingerprint
        fig1, maxweight1 = SimilarityMaps.GetSimilarityMapForFingerprint(m2, m1, fingerprint)
        fig2, maxweight2 = SimilarityMaps.GetSimilarityMapForFingerprint(m1, m2, fingerprint)
        return fig1, fig2