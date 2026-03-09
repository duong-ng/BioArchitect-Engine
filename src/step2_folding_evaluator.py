"""
==============================================================================
 BioArchitect Engine — Step 2: ESMFold Protein Structure Evaluator
==============================================================================
 Predicts 3D protein structures using ESMFold.

 Backend priority:
   1. HuggingFace transformers EsmForProteinFolding (local GPU, ~16GB fp16)
   2. fair-esm local model (esm.pretrained.esmfold_v1)
   3. API fallback with robust retry logic

 LanRecov Enhancement:
   - Multiple backend support with automatic fallback
   - Confidence score extraction (pLDDT from B-factor column)
   - Proper error classification (410 permanent vs 503 transient)
   - float16 / int8 quantization support for low-VRAM GPUs
==============================================================================
"""

import numpy as np
import time
import sys
import os
import io
import requests
import torch

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class ESMFoldEvaluator:
    """
    Real integration of the ESMFold (Evolutionary Scale Modeling) model.
    
    Backend priority:
      1. HuggingFace transformers (EsmForProteinFolding) — local GPU
      2. fair-esm (esm.pretrained.esmfold_v1) — local GPU (legacy)
      3. HuggingFace Inference API — cloud-based
      4. Legacy ESM Atlas API — deprecated fallback
    
    LanRecov Enhancement:
      - Robust multi-backend architecture
      - float16/int8 quantization for low-VRAM GPUs
      - Confidence scores extraction (pLDDT)
      - Proper error handling with exponential backoff
    """

    # API endpoints (used only when local model is not available)
    HF_API_URL = "https://api-inference.huggingface.co/models/facebook/esmfold_v1"
    LEGACY_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    # Permanent API errors that should not be retried
    PERMANENT_ERROR_CODES = {410, 404, 401, 403}

    def __init__(self, device='cuda', force_api=False, hf_token=None,
                 quantize=None):
        """
        Args:
            device (str): 'cuda' or 'cpu' for local model.
            force_api (bool): If True, skip local model loading and use API.
            hf_token (str, optional): HuggingFace API token.
            quantize (str, optional): "float16" or "int8" for reduced VRAM.
                Only applies to transformers backend.
        """
        self.force_api = force_api
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.quantize = quantize
        self.backend = None  # Will be set during initialization
        self.model = None
        self.tokenizer = None
        
        self.last_pdb_string = None
        self.last_plddt_scores = None
        
        if force_api:
            self.backend = "api"
            print("[ESMFold] force_api=True → Using cloud API backend.")
            return
        
        # Try backends in priority order
        has_gpu = torch.cuda.is_available()
        self.device = device if has_gpu else 'cpu'
        
        if has_gpu:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"[ESMFold] GPU detected: {torch.cuda.get_device_name(0)} "
                  f"({vram_gb:.1f} GB VRAM)")
        
        # Backend 1: HuggingFace transformers (preferred)
        if self._try_load_transformers_model():
            return
        
        # Backend 2: fair-esm
        if self._try_load_fairesm_model():
            return
        
        # Backend 3: API fallback
        self.backend = "api"
        print("[ESMFold] No local model available. Using cloud API backend.")
        print("[ESMFold]   Install transformers: pip install transformers torch")
        print("[ESMFold]   Or install fair-esm:   pip install fair-esm")

    def _try_load_transformers_model(self):
        """
        Attempt to load EsmForProteinFolding from HuggingFace transformers.
        Supports float16 and int8 quantization for low-VRAM GPUs.
        
        Returns:
            bool: True if successfully loaded.
        """
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
            
            print("[ESMFold] Loading EsmForProteinFolding via HuggingFace transformers...")
            
            model_name = "facebook/esmfold_v1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.quantize == "int8":
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = EsmForProteinFolding.from_pretrained(
                        model_name,
                        quantization_config=quant_config,
                        device_map="auto",
                    )
                    print("[ESMFold] Loaded in INT8 quantization (~4GB VRAM)")
                except ImportError:
                    print("[ESMFold] bitsandbytes not available for INT8. "
                          "Trying float16...")
                    self.quantize = "float16"
                except Exception as e:
                    print(f"[ESMFold] INT8 failed: {e}. Trying float16...")
                    self.quantize = "float16"
            
            if self.quantize != "int8":
                # Load in float16 or float32
                dtype = torch.float16 if self.quantize == "float16" else torch.float32
                self.model = EsmForProteinFolding.from_pretrained(
                    model_name, torch_dtype=dtype
                )
                self.model = self.model.to(self.device)
                precision = "float16" if dtype == torch.float16 else "float32"
                print(f"[ESMFold] Loaded in {precision} on {self.device}")
            
            self.model.eval()
            
            # Check if convert_outputs_to_pdb is available
            try:
                from transformers.models.esm.openfold_utils.protein import (
                    to_pdb, Protein as OFProtein
                )
                self._has_openfold_utils = True
            except ImportError:
                self._has_openfold_utils = False
            
            self.backend = "transformers"
            print("[ESMFold] ✅ Transformers backend ready.")
            return True
            
        except ImportError:
            print("[ESMFold] transformers library not available.")
            return False
        except Exception as e:
            print(f"[ESMFold] Failed to load transformers model: {e}")
            return False

    def _try_load_fairesm_model(self):
        """
        Attempt to load esmfold_v1 from fair-esm library.
        
        Returns:
            bool: True if successfully loaded.
        """
        try:
            import esm
            
            print(f"[ESMFold] Loading fair-esm esmfold_v1 on {self.device}...")
            print("[ESMFold] This requires significant RAM/VRAM (~15GB)...")
            
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval().to(self.device)
            
            self.backend = "fairesm"
            print("[ESMFold] ✅ fair-esm backend ready.")
            return True
            
        except ImportError:
            print("[ESMFold] fair-esm library not available.")
            return False
        except Exception as e:
            print(f"[ESMFold] Failed to load fair-esm model: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    #  Prediction: Dispatch to correct backend
    # ──────────────────────────────────────────────────────────────────────
    
    def predict_structure(self, sequence):
        """
        Predicts the 3D structure using ESMFold.
        Extracts and returns the Alpha Carbon (CA) spatial coordinates.
        
        Args:
            sequence (str): Amino acid sequence.
            
        Returns:
            np.ndarray: CA coordinates, shape (seq_len, 3), in Angstroms.
            
        Raises:
            RuntimeError: If structure prediction fails.
        """
        if self.backend == "transformers":
            pdb_string = self._predict_transformers(sequence)
        elif self.backend == "fairesm":
            pdb_string = self._predict_fairesm(sequence)
        elif self.backend == "api":
            pdb_string = self._predict_api(sequence)
        else:
            raise RuntimeError(f"[ESMFold] Unknown backend: {self.backend}")
        
        self.last_pdb_string = pdb_string
        
        # Parse PDB → CA coordinates + pLDDT
        coords, plddt_scores = self._parse_pdb_string(pdb_string)
        
        if not coords:
            raise RuntimeError(
                f"[ESMFold] No CA atoms parsed from PDB output. "
                f"Backend: {self.backend}, Seq length: {len(sequence)}, "
                f"PDB length: {len(pdb_string)} chars."
            )
        
        self.last_plddt_scores = np.array(plddt_scores)
        return np.array(coords)

    def _predict_transformers(self, sequence):
        """
        Local inference via HuggingFace transformers EsmForProteinFolding.
        
        Returns:
            str: PDB format string.
        """
        print(f"[ESMFold-Transformers] Folding sequence (length {len(sequence)})...")
        
        # Tokenize
        inputs = self.tokenizer(
            [sequence], return_tensors="pt",
            add_special_tokens=False, padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert output to PDB string
        pdb_string = self._outputs_to_pdb(outputs, sequence)
        
        print(f"[ESMFold-Transformers] ✅ Structure predicted.")
        return pdb_string

    def _outputs_to_pdb(self, outputs, sequence):
        """
        Convert EsmForProteinFolding outputs to PDB format string.
        
        Handles both openfold_utils and manual conversion.
        """
        try:
            # Method 1: Use transformers built-in conversion
            from transformers.models.esm.openfold_utils.protein import (
                to_pdb, Protein as OFProtein
            )
            
            # Extract final atom positions (shape: [1, seq_len, 37, 3])
            final_atom_positions = outputs["positions"][-1]
            # atom_mask
            final_atom_mask = outputs.get(
                "atom37_atom_exists",
                torch.ones(final_atom_positions.shape[:-1],
                           device=final_atom_positions.device)
            )
            
            # pLDDT from outputs
            plddt = outputs.get("plddt", None)
            if plddt is not None:
                b_factors = plddt.unsqueeze(-1).repeat(1, 1, 37).cpu().numpy()
            else:
                b_factors = np.zeros(final_atom_mask.shape)
            
            # Build aatype from sequence
            restypes = "ARNDCQEGHILKMFPSTWYV"
            aatype = []
            for aa in sequence:
                idx = restypes.find(aa)
                aatype.append(idx if idx >= 0 else 20)
            
            protein = OFProtein(
                aatype=np.array(aatype),
                atom_positions=final_atom_positions[0].cpu().numpy(),
                atom_mask=final_atom_mask[0].cpu().numpy(),
                residue_index=np.arange(len(sequence)),
                b_factors=b_factors[0] if len(b_factors.shape) == 3 else b_factors,
                chain_index=np.zeros(len(sequence), dtype=np.int32),
            )
            
            return to_pdb(protein)
            
        except (ImportError, Exception) as e:
            print(f"[ESMFold] openfold conversion failed ({e}), "
                  f"using manual PDB generation...")
            return self._manual_pdb_from_outputs(outputs, sequence)

    def _manual_pdb_from_outputs(self, outputs, sequence):
        """
        Manual PDB string generation from model outputs.
        Fallback when openfold_utils is not available.
        """
        # positions shape: [num_layers, batch, seq_len, 37, 3]
        # We want the final layer, first (only) batch element
        final_positions = outputs["positions"][-1][0].cpu().numpy()
        
        plddt = outputs.get("plddt", None)
        if plddt is not None:
            plddt_vals = plddt[0].cpu().numpy()
        else:
            plddt_vals = np.full(len(sequence), 50.0)
        
        # Standard atom names for backbone atoms (indices 0-4 in atom37)
        # 0=N, 1=CA, 2=C, 3=O, 4=CB
        backbone_atoms = [(0, " N  "), (1, " CA "), (2, " C  "), (3, " O  ")]
        
        three_letter = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        }
        
        lines = []
        atom_serial = 1
        
        for res_idx, aa in enumerate(sequence):
            resname = three_letter.get(aa, "UNK")
            bfactor = float(plddt_vals[res_idx]) if res_idx < len(plddt_vals) else 0.0
            
            for atom_idx, atom_name in backbone_atoms:
                x, y, z = final_positions[res_idx][atom_idx]
                element = atom_name.strip()[0]
                line = (
                    f"ATOM  {atom_serial:5d} {atom_name}"
                    f"{resname:>3s} A{res_idx + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00{bfactor:6.2f}           {element:>2s}  "
                )
                lines.append(line)
                atom_serial += 1
        
        lines.append("END")
        return "\n".join(lines)

    def _predict_fairesm(self, sequence):
        """
        Local inference via fair-esm library.
        
        Returns:
            str: PDB format string.
        """
        print(f"[ESMFold-fairesm] Folding sequence (length {len(sequence)})...")
        
        with torch.no_grad():
            pdb_string = self.model.infer_pdb(sequence)
        
        print(f"[ESMFold-fairesm] ✅ Structure predicted.")
        return pdb_string

    def _predict_api(self, sequence, max_retries=3):
        """
        API-based folding with robust error handling.
        
        Error classification:
          - 410/404/401/403 → Permanent failure, skip retries
          - 503 → Model loading, wait and retry
          - Timeout → Retry with backoff
        
        Args:
            sequence (str): Protein sequence.
            max_retries (int): Maximum retry attempts per endpoint.
            
        Returns:
            str: PDB format string.
        """
        headers = {"Content-Type": "text/plain"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        # ── Try HuggingFace Inference API ───────────────────────────────
        hf_permanent_fail = False
        for attempt in range(max_retries):
            try:
                print(f"[ESMFold-API] Attempt {attempt + 1}: HuggingFace API "
                      f"(seq length {len(sequence)})...")
                response = requests.post(
                    self.HF_API_URL,
                    data=sequence,
                    headers=headers,
                    timeout=120,
                )
                if response.status_code == 200:
                    return response.text
                elif response.status_code in self.PERMANENT_ERROR_CODES:
                    print(f"[ESMFold-API] HF API returned {response.status_code} "
                          f"(permanent). Skipping retries.")
                    hf_permanent_fail = True
                    break
                elif response.status_code == 503:
                    wait_time = 2 ** attempt * 5
                    print(f"[ESMFold-API] Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ESMFold-API] HF API error: {response.status_code} "
                          f"- {response.text[:200]}")
                    break
            except requests.exceptions.Timeout:
                wait_time = 2 ** attempt * 3
                print(f"[ESMFold-API] Timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError:
                print(f"[ESMFold-API] Connection error on attempt {attempt + 1}")
                time.sleep(2 ** attempt)
        
        # ── Try legacy ESM Atlas API ────────────────────────────────────
        print("[ESMFold-API] Trying legacy ESM Atlas API...")
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.LEGACY_API_URL,
                    data=sequence,
                    headers={"Content-Type": "text/plain"},
                    timeout=120,
                )
                if response.status_code == 200:
                    return response.text
                elif response.status_code in self.PERMANENT_ERROR_CODES:
                    print(f"[ESMFold-API] Legacy API returned {response.status_code} "
                          f"(permanent). Skipping.")
                    break
                else:
                    wait_time = 2 ** attempt * 3
                    print(f"[ESMFold-API] Legacy error: {response.status_code}, "
                          f"retrying in {wait_time}s...")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"[ESMFold-API] Legacy attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        
        raise ConnectionError(
            "[ESMFold] All API endpoints failed. Options:\n"
            "  1. Install transformers + torch for local GPU inference:\n"
            "     pip install transformers torch\n"
            "  2. Install fair-esm for local inference:\n"
            "     pip install fair-esm\n"
            "  3. Set hf_token for authenticated API access:\n"
            "     ESMFoldEvaluator(hf_token='hf_...')\n"
            "  4. Check internet connectivity."
        )

    # ──────────────────────────────────────────────────────────────────────
    #  PDB Parsing & Confidence Scores
    # ──────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _parse_pdb_string(pdb_string):
        """
        Parse a PDB format string to extract CA coordinates and pLDDT.
        
        Returns:
            tuple: (coords_list, plddt_list)
        """
        coords = []
        plddt_scores = []
        for line in pdb_string.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                # B-factor column contains pLDDT in ESMFold PDB output
                try:
                    bfactor = float(line[60:66])
                    plddt_scores.append(bfactor)
                except (ValueError, IndexError):
                    plddt_scores.append(0.0)
        return coords, plddt_scores

    def get_confidence_scores(self):
        """
        Extract confidence scores from the last prediction.
        pLDDT is stored in the B-factor column of ESMFold PDB output.
        
        Returns:
            dict: {
                'plddt': np.ndarray of per-residue pLDDT scores (0–100),
                'mean_plddt': float,
                'confidence_category': str ('Very High'/'High'/'Low'/'Very Low'),
                'backend': str,
            }
        """
        if self.last_plddt_scores is None:
            raise RuntimeError("No prediction available. Run predict_structure() first.")
        
        mean_plddt = float(np.mean(self.last_plddt_scores))
        
        if mean_plddt > 90:
            category = "Very High (>90)"
        elif mean_plddt > 70:
            category = "High (70-90)"
        elif mean_plddt > 50:
            category = "Low (50-70)"
        else:
            category = "Very Low (<50)"
        
        return {
            "plddt": self.last_plddt_scores,
            "mean_plddt": mean_plddt,
            "confidence_category": category,
            "backend": self.backend,
        }

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
            print(f"[ESMFold] Exported PDB file: {filename} (backend: {self.backend})")
        else:
            print("[ESMFold] Error: No PDB generated yet. Run predict_structure() first.")
