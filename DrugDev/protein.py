import os
import numpy as np
import plotly.express as px
import py3Dmol
import prody
prody.confProDy(verbosity='none')
import requests
import urllib
import subprocess

# PDB (https://www.rcsb.org/search) --> structures from PDB --> experimental + protein = 206,140 proteins
PDB_FILE = "data/prots/pdb.txt"

class PDB:

    def __init__(self):
        self.proteins = [Protein(id=x) for x in np.loadtxt(PDB_FILE, dtype=str, delimiter='\t')]
        print(f"{len(self):,} proteins loaded")
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        return self.proteins[idx]
    
    def __iter__(self):
        for p in self.proteins:
            yield p
            
    def get(self):
        return np.random.choice(self.proteins)


WIDTH, HEIGHT = 1000, 750
class Protein:
    
    def __init__(self, id=None, seq=None):
        self.id = id
        self.seq = seq
    
    def get_seq(self):
        if id: return requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{self.id}').json()[self.id.lower()][0]['sequence']
        else: return ""
        
    def fold(self):
        pass # TODO
    
    def get_pdb(self):
        if id: urllib.request.urlretrieve(f'http://files.rcsb.org/download/{self.id}.pdb', 'temp/prot.pdb')
        else: self.fold()
    
    def to_pdbqt(self):
        self.get_pdb()
        subprocess.run("obabel temp/prot.pdb -O temp/prot.pdbqt -p 7.4 -xr", shell=True, capture_output=True, text=True)
        structure = prody.parsePDB('temp/prot.pdb')
        min_coords = np.min(structure.getCoords(), axis=0)
        max_coords = np.max(structure.getCoords(), axis=0)
        return [min_coords[0], min_coords[1], min_coords[2]], [max_coords[0], max_coords[1], max_coords[2]]
    
    def get_view(self):
        view = py3Dmol.view(query='pdb:'+self.id, width=WIDTH, height=HEIGHT)
        view.setStyle({'cartoon':{'color':'spectrum'}})
        view.setBackgroundColor('#28282B') #212225
        return view

    def show(self):
        return self.get_view().show()

    def render(self):
        return self.get_view().render_image()

    def plot(self, type="protein"):
        # temperature factor, B factor, B value, or Debye-Waller factor - describes the degree to which the electron density is spread out
        # measured in units of Å2 - indicating the relative vibrational motion of different parts of the structure
        # atoms with low B-factors belong to a part of the structure that is well ordered
        def get_betas(protein):
            protein_betas = []
            hierview = protein.getHierView()
            for chain in hierview.iterChains():
                protein_betas += list(hierview[str(chain)[-1]].getBetas())
            return protein_betas

        protein = prody.parsePDB(self.id)
        if os.path.exists(self.id+".pdb.gz"):
            os.remove(self.id+".pdb.gz")
        else:
            os.remove(self.id+".cif")
        if type=="water":
            coords = protein.select("water").getCoords()
            color=None
        elif type=="wbetas":
            water = protein.select("water")
            coords = water.getCoords()
            color = water.getBetas()
        elif type=="pbetas":
            coords = protein.getCoords()
            color = get_betas(protein)
        elif type=="both":
            water_coords = protein.select("water").getCoords()
            protein_coords = protein.getCoords()
            coords = water_coords + protein_coords
            color = ["water"]*len(water_coords) + ["protein"]*len(protein_coords)
        else:
            coords = protein.getCoords()
            color = protein.getNames()
        x, y, z = zip(*coords)
        display(px.scatter_3d(x=x, y=y, z=z, color=color, opacity=0.75, width=WIDTH, height=HEIGHT).show())