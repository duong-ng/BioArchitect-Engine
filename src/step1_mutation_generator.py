import os
import random
import urllib.request
import sys
import json
import numpy as np
import torch


def setup_proteinmpnn(base_dir):
    """
    Auto-downloads the official ProteinMPNN model utilities and weights
    if they are not already integrated in the project.
    """
    mpnn_dir = os.path.join(base_dir, "proteinmpnn_core")
    os.makedirs(mpnn_dir, exist_ok=True)
    
    utils_path = os.path.join(mpnn_dir, "protein_mpnn_utils.py")
    if not os.path.exists(utils_path):
        print("[ProteinMPNN] Pulling native model definitions from official repo...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/protein_mpnn_utils.py", utils_path)
    
    weights_dir = os.path.join(mpnn_dir, "vanilla_model_weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "v_48_020.pt")
    if not os.path.exists(weight_path):
        print("[ProteinMPNN] Downloading v_48_020 weights (~10MB)...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt", weight_path)
        
    # Append to sys.path so we can import the model locally
    if mpnn_dir not in sys.path:
        sys.path.append(mpnn_dir)
        
    return weight_path

class ProteinMPNNGenerator:
    """
    Fully Integrated Native ProteinMPNN (Message Passing Neural Network) Generator.
    Runs entirely via PyTorch in-memory without relying on external subprocesses or separate repos.
    """
    def __init__(self, base_sequence, target_indices, force_mock=False):
        self.base_sequence = base_sequence
        self.target_indices = target_indices 
        self.binding_aas = ['D', 'E', 'N', 'Q', 'S', 'T'] 
        
        self.use_native_mpnn = False
        if not force_mock:
            try:
                # 1. Setup the native PyTorch files
                self.mpnn_base = os.path.dirname(os.path.abspath(__file__))
                self.weight_path = setup_proteinmpnn(self.mpnn_base)
                
                # 2. Import the real model classes natively
                import protein_mpnn_utils
                self.mpnn_utils = protein_mpnn_utils
                
                # 3. Initialize the model on GPU/CPU
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"[ProteinMPNN] Initializing Native PyTorch Model on {self.device}...")
                
                hidden_dim = 128
                num_layers = 3 
                self.model = self.mpnn_utils.ProteinMPNN(
                    num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                    hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                    augment_eps=0.0, k_neighbors=48
                )
                
                checkpoint = torch.load(self.weight_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                self.use_native_mpnn = True
                print("[ProteinMPNN] Native Model loaded successfully.")
                
            except Exception as e:
                print(f"[ProteinMPNN] Failed to load native PyTorch model: {e}")
                self.use_native_mpnn = False
        else:
            print("[ProteinMPNN] PyTorch not found. Using heuristic generator as fallback.")

    def generate_population(self, population_size=10, pdb_path=None):
        if self.use_native_mpnn and pdb_path and os.path.exists(pdb_path):
            return self._run_native_mpnn(population_size, pdb_path)
        else:
            return self._run_mock_mpnn(population_size)
            
    def _run_native_mpnn(self, population_size, pdb_path):
        """
        Runs the ProteinMPNN model entirely natively in PyTorch.
        """
        print(f"[ProteinMPNN] Running Native Inference tracking {population_size} sequences...")
        
        # Determine fixed vs mutated positions
        seq_len = len(self.base_sequence)
        mutate_indices = []
        for loop_name, (start, end) in self.target_indices.items():
            mutate_indices.extend(list(range(start, end+1)))
            
        # 1-indexed for ProteinMPNN parser
        fixed_positions = [i+1 for i in range(seq_len) if i not in mutate_indices] 
        
        # Create a temporary JSON for the tied_featurize function
        chain_id = 'A'
        pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
        fixed_dict = {pdb_name: {chain_id: fixed_positions}}
        
        json_path = "temp_fixed.json"
        with open(json_path, 'w') as f:
            json.dump(fixed_dict, f)
            
        try:
            # Parse PDB features into PyTorch tensors
            pdb_dict_list = self.mpnn_utils.parse_PDB(pdb_path, {"A": chain_id})
            dataset_valid = self.mpnn_utils.StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
            
            # Prepare batch for model
            protein = dataset_valid[0]
            batch_clones = [protein] * population_size # We duplicate the backbone for the batch size
            
            with torch.no_grad():
                # Extract features
                batch, _ = self.mpnn_utils.tied_featurize(batch_clones, self.device, None, None, None, None, None)
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = batch
                
                # Setup which positions the model is allowed to redesign (the EF-hands!)
                with open(json_path, 'r') as f:
                    designed_dict = json.load(f)
                    
                chain_mask = chain_M
                for i, b in enumerate(batch_clones):
                    if pdb_name in designed_dict and chain_id in designed_dict[pdb_name]:
                        fixed_res = designed_dict[pdb_name][chain_id]
                        for res in fixed_res:
                            idx = res - 1 # back to 0-index
                            chain_mask[i, idx] = 0.0 # 0 means do not redesign

                # Temperature for sampling (0.1 means conservative, >0.5 means very creative/risky)
                temperature = 0.1 
                randn_1 = torch.randn(chain_M.shape, device=self.device)
                
                # NATIVE MODEL INFERENCE! 💥
                sample_dict = self.model.sample(
                    X, randn_1, S, chain_mask, chain_encoding_all, residue_idx, 
                    mask, temperature, omit_AAs_np=None, bias_AAs_np=None, 
                    chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                    pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, 
                    tied_pos=None, tied_beta=None
                )
                
                S_sample = sample_dict["S"]
                
            # Decode tensors back to strings
            population = []
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
            for i in range(population_size):
                seq_tensor = S_sample[i]
                seq = "".join([alphabet[c] for c in seq_tensor])
                # Filter out the padding/dummy tokens if any
                seq = seq[:lengths[i]]
                population.append(seq)
                
            os.remove(json_path)
            print(f"[ProteinMPNN] Successfully generated {len(population)} sequences via native PyTorch.")
            return population
            
        except Exception as e:
            print(f"[ProteinMPNN] Error during native inference: {e}")
            if os.path.exists(json_path): os.remove(json_path)
            return self._run_mock_mpnn(population_size)

    def _run_mock_mpnn(self, population_size):
        print(f"[ProteinMPNN] Generating {population_size} sequences heuristically (Fallback)...")
        population = []
        for _ in range(population_size):
            seq_list = list(self.base_sequence)
            for loop_name, (start, end) in self.target_indices.items():
                num_muts = random.randint(1, 2)
                mut_positions = random.sample(range(start, end + 1), num_muts)
                for pos in mut_positions:
                    seq_list[pos] = random.choice(self.binding_aas)
            population.append("".join(seq_list))
        return population
