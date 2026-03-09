import os
import random
import urllib.request
import sys
import json
import numpy as np
import torch

# Fix Windows console encoding for Unicode output (emojis, special chars)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from lanthanide_params import (
    get_binding_residue_bias,
    LANTHANIDE_DB,
    ACID_RESISTANT_RESIDUES,
    ACID_SENSITIVE_RESIDUES,
)

# Standard amino acid alphabet used by ProteinMPNN
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}


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
    Runs entirely via PyTorch in-memory without relying on external subprocesses.
    
    LanRecov Enhancement:
      - Ion-specific mutation bias based on Lanthanide parameters
      - EF-hand targeted mutation with chemical awareness
      - pH-resistant residue preference for acid-stable designs
      - Logit bias injection for native inference path
    """
    def __init__(self, base_sequence, target_indices, target_ion="La",
                 pH_resistant=False, force_mock=False):
        """
        Args:
            base_sequence (str): Wild-type protein sequence.
            target_indices (dict): EF-hand loop indices, e.g. {"EF1": (12, 23), ...}
            target_ion (str): Target Lanthanide ion (e.g. "La", "Nd", "Dy").
            pH_resistant (bool): If True, bias toward pH-resistant residues (for pH < 3.0).
            force_mock (bool): Force heuristic generator even if ProteinMPNN is available.
        """
        self.base_sequence = base_sequence
        self.target_indices = target_indices 
        self.target_ion = target_ion
        self.pH_resistant = pH_resistant
        
        # Load ion-specific binding residue preferences
        self._load_ion_preferences()
        
        # Build the set of mutable positions (0-indexed)
        self.mutate_indices = set()
        for loop_name, (start, end) in self.target_indices.items():
            self.mutate_indices.update(range(start, end + 1))
        
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
                
                checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                self.use_native_mpnn = True
                print("[ProteinMPNN] Native Model loaded successfully.")
                
            except Exception as e:
                print(f"[ProteinMPNN] Failed to load native PyTorch model: {e}")
                self.use_native_mpnn = False
        else:
            print("[ProteinMPNN] force_mock=True. Using heuristic generator as fallback.")

    def _load_ion_preferences(self):
        """Load ion-specific amino acid preferences from lanthanide database."""
        try:
            bias = get_binding_residue_bias(self.target_ion)
            self.binding_aas = bias["residues"]
            self.binding_weights = bias["weights"]
            ion_info = LANTHANIDE_DB[self.target_ion]
            print(f"[ProteinMPNN] Target ion: {ion_info['symbol']} ({ion_info['name']}) "
                  f"| Group: {ion_info['group']} | Radius: {ion_info['ionic_radius']} Å")
        except KeyError:
            print(f"[ProteinMPNN] Warning: Ion '{self.target_ion}' not in database. "
                  f"Using default binding residues.")
            self.binding_aas = ['D', 'E', 'N', 'Q', 'S', 'T']
            self.binding_weights = [0.30, 0.25, 0.18, 0.12, 0.08, 0.07]
        
        # Adjust for pH resistance if needed
        if self.pH_resistant:
            self._apply_pH_bias()

    def _apply_pH_bias(self):
        """
        Adjust amino acid weights to favor pH-resistant residues.
        At pH < 3.0, Asp (pKa 3.65) and Glu (pKa 4.25) are protonated
        and lose metal binding ability. We reduce their weight and boost
        Asn, Gln, Ser, Thr.
        """
        print("[ProteinMPNN] Applying pH < 3.0 resistance bias...")
        adjusted_weights = []
        for aa, w in zip(self.binding_aas, self.binding_weights):
            if aa in ACID_SENSITIVE_RESIDUES:
                adjusted_weights.append(w * 0.3)  # Reduce acid-sensitive
            elif aa in ACID_RESISTANT_RESIDUES:
                adjusted_weights.append(w * 2.0)  # Boost acid-resistant
            else:
                adjusted_weights.append(w)
        
        # Normalize
        total = sum(adjusted_weights)
        self.binding_weights = [w / total for w in adjusted_weights]

    def _get_ion_biased_aa(self):
        """
        Select an amino acid based on ion-specific weights.
        
        Returns:
            str: Single amino acid one-letter code.
        """
        return random.choices(self.binding_aas, weights=self.binding_weights, k=1)[0]

    def _build_bias_by_res(self, seq_len):
        """
        Build a per-residue amino acid bias tensor for ProteinMPNN.
        
        For mutable positions (EF-hand loops), we inject logit biases
        that favor ion-binding residues. For fixed positions, bias is zero.
        
        Shape: (seq_len, 21) — one row per residue, one column per amino acid.
        
        Returns:
            torch.Tensor: Bias tensor on self.device.
        """
        bias = torch.zeros(seq_len, 21, device=self.device)
        
        # Build bias vector from binding preferences
        aa_bias_vector = torch.zeros(21, device=self.device)
        for aa, weight in zip(self.binding_aas, self.binding_weights):
            if aa in AA_TO_IDX:
                # Convert weight to logit bias (log-scale, scaled up for effect)
                aa_bias_vector[AA_TO_IDX[aa]] = np.log(weight + 1e-8) + 3.0
        
        # pH resistance: penalize acid-sensitive residues at mutable positions
        if self.pH_resistant:
            for aa in ACID_SENSITIVE_RESIDUES:
                if aa in AA_TO_IDX:
                    aa_bias_vector[AA_TO_IDX[aa]] -= 2.0  # Extra penalty
            for aa in ACID_RESISTANT_RESIDUES:
                if aa in AA_TO_IDX:
                    aa_bias_vector[AA_TO_IDX[aa]] += 1.5  # Extra boost
        
        # Apply bias only to mutable (EF-hand) positions
        for idx in self.mutate_indices:
            if idx < seq_len:
                bias[idx] = aa_bias_vector
        
        return bias

    def generate_population(self, population_size=10, pdb_path=None):
        if self.use_native_mpnn and pdb_path and os.path.exists(pdb_path):
            return self._run_native_mpnn(population_size, pdb_path)
        else:
            return self._run_mock_mpnn(population_size)
            
    def _run_native_mpnn(self, population_size, pdb_path):
        """
        Runs the ProteinMPNN model entirely natively in PyTorch.
        
        Fixed issues:
          1. Correct parse_PDB call (no ca_only dict — just input_path + chain list)
          2. Build chain_M_pos mask directly from EF-hand indices (no JSON roundtrip)
          3. Validate tensor dimensions before model.sample()
          4. Inject ion-bias via bias_by_res tensor
          5. Proper sequence decoding with length truncation
        """
        print(f"[ProteinMPNN] Running Native Inference for {population_size} sequences...")
        
        seq_len = len(self.base_sequence)
        
        try:
            # ── Step 1: Parse PDB into feature dict ──────────────────────
            pdb_dict_list = self.mpnn_utils.parse_PDB(pdb_path)
            if not pdb_dict_list:
                raise ValueError(f"parse_PDB returned empty list for {pdb_path}")
            
            dataset_valid = self.mpnn_utils.StructureDatasetPDB(
                pdb_dict_list, truncate=None, max_length=20000
            )
            
            if len(dataset_valid) == 0:
                raise ValueError("StructureDatasetPDB produced no entries")
            
            protein = dataset_valid[0]
            
            # ── Step 2: Featurize the batch ──────────────────────────────
            # Replicate the single protein to create a batch
            batch_clones = [protein] * population_size
            
            with torch.no_grad():
                # tied_featurize returns a variable number of items depending
                # on ProteinMPNN version. We unpack carefully.
                featurize_result = self.mpnn_utils.tied_featurize(
                    batch_clones, self.device,
                    None, None, None, None, None
                )
                
                # The first element is always the main batch tuple
                if isinstance(featurize_result, tuple) and len(featurize_result) == 2:
                    batch = featurize_result[0]
                else:
                    batch = featurize_result
                
                # Unpack batch — handle both old (20-item) and new formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 20:
                        (X, S, mask, lengths, chain_M, chain_encoding_all,
                         chain_list_list, visible_list_list, masked_list_list,
                         masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
                         residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
                         pssm_coef, pssm_bias, pssm_log_odds_all,
                         bias_by_res_all, tied_beta) = batch[:20]
                    else:
                        raise ValueError(
                            f"tied_featurize returned {len(batch)} items, expected ≥20"
                        )
                else:
                    raise TypeError(
                        f"Unexpected featurize output type: {type(batch)}"
                    )
                
                # ── Step 3: Build mutable position mask ──────────────────
                # chain_M_pos: 1.0 = redesign, 0.0 = keep fixed
                actual_len = S.shape[1]  # Actual sequence length in tensor
                chain_M_pos = torch.zeros_like(S, dtype=torch.float32)
                
                for idx in self.mutate_indices:
                    if idx < actual_len:
                        chain_M_pos[:, idx] = 1.0
                
                # Combine with chain_M (which masks non-protein positions)
                design_mask = chain_M * chain_M_pos
                
                # ── Step 4: Build ion-bias tensor ────────────────────────
                bias_by_res = self._build_bias_by_res(actual_len)
                # Expand to batch dimension: (batch, seq_len, 21)
                bias_by_res_batch = bias_by_res.unsqueeze(0).expand(
                    population_size, -1, -1
                )
                
                # ── Step 5: Validate dimensions ─────────────────────────
                assert X.shape[0] == population_size, \
                    f"Batch size mismatch: X={X.shape[0]} vs pop={population_size}"
                assert X.shape[1] == actual_len, \
                    f"Seq length mismatch: X={X.shape[1]} vs S={actual_len}"
                
                print(f"[ProteinMPNN] Tensor shapes: X={list(X.shape)}, "
                      f"S={list(S.shape)}, mask={list(mask.shape)}, "
                      f"design_mask sum={design_mask.sum().item():.0f}")
                
                # ── Step 6: Native Model Inference ───────────────────────
                # Temperature: 0.1 = conservative, >0.5 = creative
                temperature = 0.1
                randn_1 = torch.randn(chain_M.shape, device=self.device)
                
                sample_dict = self.model.sample(
                    X, randn_1, S, design_mask, chain_encoding_all,
                    residue_idx, mask, temperature,
                    omit_AAs_np=None, bias_AAs_np=None,
                    chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                    pssm_multi=0.0, tied_pos=None, tied_beta=None,
                    bias_by_res=bias_by_res_batch,
                )
                
                S_sample = sample_dict["S"]
            
            # ── Step 7: Decode tensors back to sequences ─────────────
            population = []
            for i in range(population_size):
                seq_tensor = S_sample[i]
                # Get the actual length for this sequence
                seq_length = int(lengths[i]) if i < len(lengths) else actual_len
                seq_length = min(seq_length, actual_len)
                
                # Decode indices to amino acid characters
                seq_chars = []
                for j in range(seq_length):
                    aa_idx = int(seq_tensor[j].item())
                    if 0 <= aa_idx < len(MPNN_ALPHABET):
                        seq_chars.append(MPNN_ALPHABET[aa_idx])
                    else:
                        seq_chars.append('X')  # Unknown
                
                seq = "".join(seq_chars)
                population.append(seq)
            
            print(f"[ProteinMPNN] Successfully generated {len(population)} "
                  f"sequences via native PyTorch.")
            
            # Report mutation summary
            if population:
                wt = self.base_sequence
                sample_seq = population[0]
                n_mut = sum(1 for a, b in zip(wt[:len(sample_seq)],
                                               sample_seq[:len(wt)])
                            if a != b)
                print(f"[ProteinMPNN] Sample mutations vs WT: {n_mut} positions")
            
            return population
            
        except Exception as e:
            print(f"[ProteinMPNN] Error during native inference: {e}")
            import traceback
            traceback.print_exc()
            print("[ProteinMPNN] Falling back to ion-biased heuristic generator...")
            return self._run_mock_mpnn(population_size)

    def _run_mock_mpnn(self, population_size):
        """
        Generate mutant sequences heuristically with ion-specific bias.
        Uses weighted amino acid selection based on target Lanthanide properties.
        """
        print(f"[ProteinMPNN] Generating {population_size} sequences heuristically (Ion-biased Fallback)...")
        print(f"[ProteinMPNN] → Target: {self.target_ion} | Preferred AAs: {self.binding_aas[:3]} "
              f"| pH-resistant: {self.pH_resistant}")
        
        population = []
        for _ in range(population_size):
            seq_list = list(self.base_sequence)
            for loop_name, (start, end) in self.target_indices.items():
                num_muts = random.randint(1, 3)
                mut_positions = random.sample(range(start, end + 1), min(num_muts, end - start + 1))
                for pos in mut_positions:
                    # Use ion-biased selection instead of uniform random
                    seq_list[pos] = self._get_ion_biased_aa()
            population.append("".join(seq_list))
        return population
