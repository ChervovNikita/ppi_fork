import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib

import biographs as bg
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser


from transformers import AutoTokenizer, AutoModel
import re
import torch

import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# ESM-2 model loading
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

# list of 20 proteins
pro_res_table = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

# Add a standard amino acid conversion dictionary
AMINO_ACID_DICT = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

class ProteinDataset:
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        self.processed_dir = os.path.join(root, "processed_esm")
        os.makedirs(self.processed_dir, exist_ok=True)
        # Debug: Print root directory and raw paths
        print(f"Root directory: {root}")
        print(f"Full root path: {os.path.abspath(root)}")
        
        self.raw_paths = [os.path.join(root, "raw", f) for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pdb")]
        
        # Debug: Print raw paths
        # print(f"Raw paths: {self.raw_paths}")
        print(f"Number of PDB files found: {len(self.raw_paths)}")

    @property
    def raw_file_names(self):
        return [filename.name for filename in os.scandir(self.root + "/raw")]

    @property
    def processed_file_names(self):
        return [
            os.path.splitext(os.path.basename(file))[0] + ".pt"
            for file in self.raw_paths
        ]

    def _get_structure(self, file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file)
        return structure

    def _get_sequence(self, structure):
        seq = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in AMINO_ACID_DICT:
                        seq += AMINO_ACID_DICT[residue.get_resname()]
        return seq

    def _get_esm_embeddings(self, sequence):
        # Truncate sequence if too long
        max_length = 1024  # ESM-2 max sequence length
        sequence = sequence[:max_length]
        
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Average pooling of token embeddings, excluding special tokens
        embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]  # Remove [CLS] and [SEP] tokens
        return embeddings.to('cpu')  # Move back to CPU for compatibility

    def _get_adjacency(self, file):
        # Existing adjacency matrix generation logic
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file)
        
        # Create a simple distance-based adjacency matrix
        atoms = list(structure.get_atoms())
        n = len(atoms)
        adjacency_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(atoms[i].coord - atoms[j].coord)
                if distance < 8.0:  # Threshold for considering an edge
                    adjacency_mat[i, j] = adjacency_mat[j, i] = 1
        
        return torch.tensor(adjacency_mat, dtype=torch.float)

    def _get_edgeindex(self, file, adjacency_mat):
        # Convert adjacency matrix to edge index
        edge_index = torch.nonzero(adjacency_mat).t().contiguous()
        return edge_index

    def process(self):
        data_list = []
        count = 0
        for file in tqdm(self.raw_paths):
            if pathlib.Path(file).suffix == ".pdb":
                try:
                    struct = self._get_structure(file)
                    seq = self._get_sequence(struct)

                    # Node features extracted using ESM-2
                    node_feats = self._get_esm_embeddings(seq)

                    # Edge index extracted
                    mat = self._get_adjacency(file)

                    # Ensure node features and adjacency matrix are compatible
                    if mat.shape[0] >= torch.Tensor.size(node_feats)[0]:
                        edge_index = self._get_edgeindex(file, mat)

                        # Create data object
                        data = Data(x=node_feats, edge_index=edge_index)
                        count += 1
                        data_list.append(data)

                        # Save processed graph
                        torch.save(
                            data,
                            os.path.join(
                                self.processed_dir,
                                os.path.splitext(os.path.basename(file))[0] + ".pt"
                            )
                        )

                except Exception as e:
                    print(f"Error processing {file}: {e}")

        self.data_prot = data_list
        print(f"Processed {count} protein graphs")

# Create dataset
if __name__ == "__main__":
    prot_graphs = ProteinDataset("../human_features/")
    prot_graphs.process()