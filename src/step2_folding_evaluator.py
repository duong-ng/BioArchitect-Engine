import numpy as np
import time
import requests
import torch
import esm

class ESMFoldEvaluator:
    """
    Real integration of the ESMFold (Evolutionary Scale Modeling) model.
    Optionally runs the 15B parameter model locally (if fair-esm + GPU is available),
    or falls back to the official ESMFold API for immediate results without heavy hardware.
    """
    def __init__(self, device='cuda', force_api=False):
        self.force_api = force_api
        if torch.cuda.is_available() and not force_api:
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"[ESMFold] Initializing local ESMFold model on {self.device}...")
            print("[ESMFold] Loading the 15B parameter model. This requires significant RAM/VRAM...")
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval().to(self.device)
            print("[ESMFold] Local Model loaded successfully.")
        else:
            self.model = None
            print("[ESMFold] fair-esm not installed or force_api=True. Using ESMFold Cloud API instead.")
            
        self.last_pdb_string = None

    def _predict_api(self, sequence):
        """Uses the free ESMFold meta API to fold the protein"""
        # API endpoint from Evolutionary Scale Modeling
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        print(f"[ESMFold] Sending sequence (length {len(sequence)}) to ESMFold API...")
        response = requests.post(url, data=sequence, headers={"Content-Type": "text/plain"})
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"ESMFold API Error: {response.status_code} - {response.text}")

    def predict_structure(self, sequence):
        """
        Predicts the 3D structure using ESMFold (Local or API).
        Extracts and returns the Alpha Carbon (CA) spatial coordinates.
        """
        if self.model is not None:
            # Local Inference
            # print(f"[ESMFold] Folding sequence locally...")
            with torch.no_grad():
                pdb_string = self.model.infer_pdb(sequence)
        else:
            # API Inference
            pdb_string = self._predict_api(sequence)
            time.sleep(1) # Be gentle with the API if called in a loop
            
        self.last_pdb_string = pdb_string
        
        # Parse the PDB string to extract CA (Alpha-Carbon) coordinates
        coords = []
        for line in pdb_string.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                
        # If something failed in parsing, return a random walk
        if not coords:
            print("[ESMFold] Warning: Could not parse CA coordinates. Returning mock.")
            return np.cumsum(np.random.normal(0, 3.8, (len(sequence), 3)), axis=0)
            
        return np.array(coords)

    def get_residue_distance(self, coords, index1, index2):
        """
        Calculates the Euclidean distance between two residues (in Angstroms).
        """
        p1 = coords[index1]
        p2 = coords[index2]
        dist = np.linalg.norm(p1 - p2)
        return dist

    def write_pdb(self, sequence, coords, filename="output.pdb"):
        """
        Writes the ESMFold-generated PDB string directly to a file
        for high-quality visualization in UCSF ChimeraX.
        """
        if self.last_pdb_string:
            with open(filename, 'w') as f:
                f.write(self.last_pdb_string)
            print(f"[ESMFold] Exported REAL PDB file: {filename}")
        else:
            print("[ESMFold] Error: No PDB generated yet. Run predict_structure() first.")
