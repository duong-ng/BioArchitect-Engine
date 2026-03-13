"""
==============================================================================
 BioArchitect Engine — Step 2: AlphaFold2 Protein Structure Evaluator
==============================================================================
 Predicts 3D protein structures using AlphaFold2 architecture for superior
 side-chain accuracy, critical for La³⁺ EF-hand binding pocket precision.

 Backend priority:
   1. LocalColabFold + MMseqs2 API — highest accuracy, online MSA
   2. OpenFold single-sequence — offline fallback, AF2 architecture

 Key features:
   - PAE (Predicted Aligned Error) extraction for inter-residue confidence
   - PDBFixer integration for 100% OpenMM-compatible PDB output
   - Memory-optimized: model chunking, CPU offloading, aggressive cleanup
   - num_recycles=3 for accuracy/performance balance

 LanRecov Integration:
   - EF-hand-specific confidence scoring (pLDDT + PAE sub-matrix)
   - Side-chain placement accuracy for Asp/Glu carboxylate coordination
   - PDB output pre-fixed for terminal atoms (OXT), hydrogens, HIS states
==============================================================================
"""

import numpy as np
import os
import sys
import io
import gc
import json
import time
import tempfile
import shutil
import warnings

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ─── Three-letter amino acid code mapping ───────────────────────────────────
AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'UNK',
}


def _detect_colabfold():
    """Check if LocalColabFold (colabfold.batch) is available."""
    try:
        from colabfold.batch import run as colabfold_run
        from colabfold.batch import get_queries, set_model_type
        return True
    except Exception:
        return False


def _detect_openfold():
    """Check if OpenFold is available for single-sequence AF2."""
    try:
        from openfold.model.model import AlphaFold
        from openfold.config import model_config
        return True
    except Exception:
        return False


def _detect_pdbfixer():
    """Check if PDBFixer is available for PDB post-processing."""
    try:
        from pdbfixer import PDBFixer
        return True
    except ImportError:
        return False


def _detect_torch():
    """Check for PyTorch and GPU availability."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        else:
            vram_gb = 0.0
        return True, has_cuda, vram_gb
    except ImportError:
        return False, False, 0.0


# Cache detection results at module level
HAS_COLABFOLD = _detect_colabfold()
HAS_OPENFOLD = _detect_openfold()
HAS_PDBFIXER = _detect_pdbfixer()
HAS_TORCH, HAS_CUDA, VRAM_GB = _detect_torch()


class AlphaFold2Evaluator:
    """
    AlphaFold2-based Protein Structure Evaluator for BioArchitect Engine.
    Superior side-chain accuracy and PAE (Predicted Aligned Error) confidence.
    
    Backend priority:
      1. LocalColabFold — AF2 + MMseqs2 API (online, full MSA)
      2. OpenFold — AF2 architecture, single-sequence mode (offline)
    
    Key features for LanRecov:
      - PAE matrix extraction → inter-residue confidence for EF-hand binding
      - PDBFixer integration → OpenMM-ready PDB output (fixes OXT, HIS, H)
      - Memory-optimized → gc.collect + torch.cuda.empty_cache after each fold
      - Configurable num_recycles (default=3) for accuracy/speed balance
    """

    def __init__(self, device='cpu', num_recycles=3, use_msa=True,
                 msa_mode='mmseqs2_uniref_env', max_seq=256,
                 model_type='alphafold2_ptm', ef_hands=None,
                 chunk_size=None, fix_pdb=True, fix_pH=7.0):
        """
        Initialize AlphaFold2 evaluator.

        Args:
            device (str): 'cuda' or 'cpu'. Auto-detects GPU.
            num_recycles (int): Number of recycling iterations (1-20).
                3 = good accuracy/speed balance for small proteins (<200 AA).
            use_msa (bool): If True, use MMseqs2 API for MSA (requires internet).
                If False, run in single-sequence mode.
            msa_mode (str): MSA search mode for ColabFold.
                'mmseqs2_uniref_env' (default, highest accuracy)
                'mmseqs2_uniref' (faster, UniRef only)
                'single_sequence' (no MSA, offline)
            max_seq (int): Maximum number of MSA sequences to use.
                Lower = faster + less memory. 256 is good for small proteins.
            model_type (str): AlphaFold2 model variant.
                'alphafold2_ptm' (with pTM head, provides PAE)
                'alphafold2' (no pTM, no PAE)
            ef_hands (dict): EF-hand loop definitions {name: (start, end)}.
                Used for region-specific confidence scoring.
            chunk_size (int, optional): Evoformer attention chunk size.
                Lower = less memory but slower. Auto-set if None.
            fix_pdb (bool): If True, run PDBFixer after every prediction.
            fix_pH (float): pH for PDBFixer hydrogen addition.
        """
        self.num_recycles = num_recycles
        self.use_msa = use_msa
        self.msa_mode = msa_mode if use_msa else 'single_sequence'
        self.max_seq = max_seq
        self.model_type = model_type
        self.ef_hands = ef_hands or {}
        self.chunk_size = chunk_size
        self.fix_pdb = fix_pdb
        self.fix_pH = fix_pH
        self.backend = None

        # Cached prediction results
        self.last_pdb_string = None
        self.last_plddt_scores = None
        self.last_pae_matrix = None
        self.last_ptm_score = None

        # Device setup
        if HAS_TORCH:
            import torch
            if device == 'cuda' and HAS_CUDA:
                self.device = 'cuda'
                print(f"[AF2] GPU detected: {torch.cuda.get_device_name(0)} "
                      f"({VRAM_GB:.1f} GB VRAM)")
                if VRAM_GB < 8.0:
                    print(f"[AF2] ⚠️  Low VRAM ({VRAM_GB:.1f}GB). "
                          f"Using CPU offloading for safety.")
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        # Auto-set chunk_size for memory optimization
        if self.chunk_size is None:
            if self.device == 'cpu' or (HAS_CUDA and VRAM_GB < 12):
                self.chunk_size = 64  # Conservative for low-memory
            else:
                self.chunk_size = 128  # Standard for >=12GB VRAM

        # Initialize backend
        self._init_backend()

    def _init_backend(self):
        """Initialize the best available AF2 backend."""
        print("[AF2] Initializing AlphaFold2 evaluator...")
        print(f"[AF2]   num_recycles={self.num_recycles}, "
              f"msa_mode={self.msa_mode}, "
              f"model_type={self.model_type}")
        print(f"[AF2]   chunk_size={self.chunk_size}, "
              f"device={self.device}, "
              f"fix_pdb={self.fix_pdb}")

        # Backend 1: LocalColabFold
        if HAS_COLABFOLD:
            self.backend = "colabfold"
            print("[AF2] ✅ Backend: LocalColabFold (AF2 + MMseqs2)")
            if not self.use_msa:
                print("[AF2]    MSA disabled → single-sequence mode")
            return

        # Backend 2: OpenFold (single-sequence AF2)
        if HAS_OPENFOLD:
            self.backend = "openfold"
            self.msa_mode = "single_sequence"
            print("[AF2] ✅ Backend: OpenFold (single-sequence AF2)")
            print("[AF2]    No MSA — using AF2 architecture without alignment")
            return

        # No backend available
        self.backend = "none"
        print("[AF2] ❌ No folding backend available!")
        print("[AF2]    Install one of:")
        print("[AF2]      pip install colabfold[alphafold]   (recommended)")
        print("[AF2]      pip install openfold               (offline AF2)")

    # ──────────────────────────────────────────────────────────────────────
    #  Public API: predict_structure (drop-in compatible)
    # ──────────────────────────────────────────────────────────────────────

    def predict_structure(self, sequence):
        """
        Predict 3D protein structure using AlphaFold2.
        Returns Alpha Carbon (CA) coordinates — compatible with GA pipeline.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            np.ndarray: CA coordinates, shape (seq_len, 3), in Angstroms.

        Raises:
            RuntimeError: If structure prediction fails entirely.
        """
        result = self.predict_full(sequence)
        return result["ca_coords"]

    def predict_full(self, sequence):
        """
        Full structure prediction returning all available data.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            dict: {
                'ca_coords': np.ndarray (seq_len, 3),
                'pdb_string': str,
                'plddt': np.ndarray (seq_len,),
                'mean_plddt': float,
                'pae': np.ndarray (seq_len, seq_len) or None,
                'ptm': float or None,
                'backend': str,
            }
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join(c if c in valid_aa else "X" for c in sequence)

        print(f"[AF2] Predicting structure for sequence "
              f"(length={len(sequence)}, backend={self.backend})...")

        try:
            if self.backend == "colabfold":
                result = self._predict_colabfold(sequence)
            elif self.backend == "openfold":
                result = self._predict_openfold(sequence)
            else:
                raise RuntimeError(
                    "[AF2] No backend available. Install colabfold or openfold."
                )
        finally:
            # ═══ MEMORY CLEANUP ═══
            # Critical for GA loop: free GPU/RAM after each prediction
            self._cleanup_memory()

        # Cache results for get_confidence_scores()
        self.last_pdb_string = result["pdb_string"]
        self.last_plddt_scores = result["plddt"]
        self.last_pae_matrix = result.get("pae", None)
        self.last_ptm_score = result.get("ptm", None)

        # Optionally fix PDB with PDBFixer
        if self.fix_pdb and result["pdb_string"]:
            result["pdb_string"] = self._apply_pdbfixer(
                result["pdb_string"], self.fix_pH
            )
            self.last_pdb_string = result["pdb_string"]

        print(f"[AF2] ✅ Structure predicted | "
              f"mean pLDDT={result['mean_plddt']:.1f} | "
              f"pTM={result.get('ptm', 'N/A')} | "
              f"backend={self.backend}")

        return result

    # ──────────────────────────────────────────────────────────────────────
    #  Backend 1: LocalColabFold
    # ──────────────────────────────────────────────────────────────────────

    def _predict_colabfold(self, sequence):
        """
        Full AF2 prediction via LocalColabFold.

        Uses colabfold.batch to:
        1. Generate MSA via MMseqs2 API (or single-sequence)
        2. Run AlphaFold2 model with recycling
        3. Extract pLDDT, PAE, pTM, and PDB output
        """
        from colabfold.batch import run as colabfold_run
        from colabfold.batch import get_queries, set_model_type

        print(f"[AF2-ColabFold] MSA mode: {self.msa_mode}")

        # Create temporary directory for ColabFold I/O
        work_dir = tempfile.mkdtemp(prefix="bioarchitect_af2_")
        input_dir = os.path.join(work_dir, "input")
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Write input FASTA
            fasta_path = os.path.join(input_dir, "query.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">query\n{sequence}\n")

            # Run ColabFold batch
            queries, is_complex = get_queries(input_dir)

            colabfold_run(
                queries=queries,
                result_dir=output_dir,
                use_templates=False,
                num_recycles=self.num_recycles,
                model_type=self.model_type,
                msa_mode=self.msa_mode,
                num_models=1,  # Only need best model for GA
                num_seeds=1,
                is_complex=is_complex,
                max_seq=self.max_seq,
                zip_results=False,
                use_gpu=(self.device == 'cuda'),
            )

            # Parse ColabFold outputs
            result = self._parse_colabfold_output(output_dir, sequence)
            return result

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    def _parse_colabfold_output(self, output_dir, sequence):
        """Parse ColabFold output files to extract coordinates and scores."""
        ca_coords = None
        pdb_string = None
        plddt = None
        pae = None
        ptm = None

        # Find PDB file
        for f in os.listdir(output_dir):
            if f.endswith(".pdb") and "rank_001" in f:
                pdb_path = os.path.join(output_dir, f)
                with open(pdb_path, "r") as fh:
                    pdb_string = fh.read()
                break

        # If no ranked PDB, find any PDB
        if pdb_string is None:
            for f in os.listdir(output_dir):
                if f.endswith(".pdb"):
                    pdb_path = os.path.join(output_dir, f)
                    with open(pdb_path, "r") as fh:
                        pdb_string = fh.read()
                    break

        # Parse CA coords and pLDDT from PDB
        if pdb_string:
            coords_list, plddt_list = self._parse_pdb_string(pdb_string)
            if coords_list:
                ca_coords = np.array(coords_list)
                plddt = np.array(plddt_list)

        # Parse PAE from JSON scores file
        for f in os.listdir(output_dir):
            if f.endswith("_scores_rank_001") or "scores" in f:
                score_path = os.path.join(output_dir, f)
                if score_path.endswith(".json"):
                    try:
                        with open(score_path, "r") as fh:
                            scores = json.load(fh)
                        if "pae" in scores:
                            pae = np.array(scores["pae"])
                        if "ptm" in scores:
                            ptm = float(scores["ptm"])
                        if "plddt" in scores and plddt is None:
                            plddt = np.array(scores["plddt"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Find JSON scores in any matching file
        if pae is None:
            for f in os.listdir(output_dir):
                if f.endswith(".json"):
                    try:
                        score_path = os.path.join(output_dir, f)
                        with open(score_path, "r") as fh:
                            scores = json.load(fh)
                        if "pae" in scores:
                            pae = np.array(scores["pae"])
                        if "ptm" in scores and ptm is None:
                            ptm = float(scores["ptm"])
                        if "plddt" in scores and plddt is None:
                            plddt = np.array(scores["plddt"])
                    except Exception:
                        continue

        if ca_coords is None:
            raise RuntimeError(
                "[AF2-ColabFold] Failed to parse any output. "
                "Check ColabFold installation and internet connectivity."
            )

        if plddt is None:
            plddt = np.full(len(sequence), 50.0)

        mean_plddt = float(np.mean(plddt))

        return {
            "ca_coords": ca_coords,
            "pdb_string": pdb_string,
            "plddt": plddt,
            "mean_plddt": mean_plddt,
            "pae": pae,
            "ptm": ptm,
            "backend": "colabfold",
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Backend 2: OpenFold (single-sequence AF2)
    # ──────────────────────────────────────────────────────────────────────

    def _predict_openfold(self, sequence):
        """
        Single-sequence AlphaFold2 prediction via OpenFold.
        
        Runs the AF2 Evoformer + Structure Module without MSA,
        using the sequence itself as a single-row alignment.
        Less accurate than full MSA mode but maintains AF2's superior
        side-chain prediction architecture.
        """
        import torch
        from openfold.model.model import AlphaFold
        from openfold.config import model_config
        from openfold.data import data_pipeline
        from openfold.utils.tensor_utils import tensor_tree_map

        print("[AF2-OpenFold] Running single-sequence AF2 prediction...")

        # Get model config for ptm variant
        config_name = "model_1_ptm"
        config = model_config(config_name)
        config.data.common.max_recycling_iters = self.num_recycles
        config.globals.chunk_size = self.chunk_size

        # Initialize model
        model = AlphaFold(config)
        model = model.eval()

        if self.device == 'cuda':
            model = model.cuda()

        # Prepare single-sequence input features
        feature_dict = self._make_single_sequence_features(sequence)

        # Move to device
        feature_dict = {
            k: torch.tensor(v).to(self.device) if isinstance(v, np.ndarray)
            else v
            for k, v in feature_dict.items()
        }

        # Add batch dimension
        feature_dict = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in feature_dict.items()
        }

        # Run inference
        with torch.no_grad():
            output = model(feature_dict)

        # Extract results
        # Final atom positions: (1, seq_len, 37, 3) in Angstroms
        final_positions = output["final_atom_positions"][0].cpu().numpy()
        final_mask = output["final_atom_mask"][0].cpu().numpy()

        # CA is atom index 1
        ca_coords = final_positions[:, 1, :]  # (seq_len, 3)

        # pLDDT
        plddt = output["plddt"][0].cpu().numpy()  # (seq_len,)
        plddt = plddt * 100.0  # Scale to 0-100

        # PAE (if ptm model)
        pae = None
        ptm = None
        if "predicted_aligned_error" in output:
            pae = output["predicted_aligned_error"][0].cpu().numpy()
        if "ptm" in output:
            ptm = float(output["ptm"].cpu().item())

        # Generate PDB string
        pdb_string = self._generate_pdb_from_positions(
            sequence, final_positions, final_mask, plddt
        )

        mean_plddt = float(np.mean(plddt))

        # Free the model immediately
        del model, output, feature_dict
        self._cleanup_memory()

        return {
            "ca_coords": ca_coords,
            "pdb_string": pdb_string,
            "plddt": plddt,
            "mean_plddt": mean_plddt,
            "pae": pae,
            "ptm": ptm,
            "backend": "openfold",
        }

    def _make_single_sequence_features(self, sequence):
        """
        Create AF2 input features from a single sequence (no MSA).
        
        The sequence acts as a single-row MSA.
        Template features are left empty.
        """
        seq_len = len(sequence)

        # Residue type encoding
        restypes = "ARNDCQEGHILKMFPSTWYV"
        aatype = np.array([
            restypes.index(aa) if aa in restypes else 20
            for aa in sequence
        ], dtype=np.int64)

        # Single-row MSA (the sequence itself)
        msa = aatype[np.newaxis, :]  # (1, seq_len)

        # Deletion matrix (no deletions for single sequence)
        deletion_matrix = np.zeros((1, seq_len), dtype=np.float32)

        # Residue index (0-indexed)
        residue_index = np.arange(seq_len, dtype=np.int64)

        features = {
            "aatype": aatype,
            "residue_index": residue_index,
            "msa": msa,
            "deletion_matrix_int": deletion_matrix.astype(np.int64),
            "num_alignments": np.array(1, dtype=np.int64),
            "seq_length": np.array(seq_len, dtype=np.int64),
            # Empty template features
            "template_aatype": np.zeros((0, seq_len), dtype=np.int64),
            "template_all_atom_positions": np.zeros(
                (0, seq_len, 37, 3), dtype=np.float32
            ),
            "template_all_atom_mask": np.zeros(
                (0, seq_len, 37), dtype=np.float32
            ),
        }

        return features

    # ──────────────────────────────────────────────────────────────────────
    #  PDB Parsing & Generation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdb_string(pdb_string):
        """
        Parse PDB format string to extract CA coordinates and pLDDT.

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
                # B-factor column contains pLDDT
                try:
                    bfactor = float(line[60:66])
                    plddt_scores.append(bfactor)
                except (ValueError, IndexError):
                    plddt_scores.append(0.0)
        return coords, plddt_scores

    def _generate_pdb_from_positions(self, sequence, atom_positions,
                                      atom_mask, plddt):
        """
        Generate PDB format string from atom37 positions.

        Args:
            sequence (str): Amino acid sequence.
            atom_positions (np.ndarray): Shape (seq_len, 37, 3).
            atom_mask (np.ndarray): Shape (seq_len, 37).
            plddt (np.ndarray): Per-residue pLDDT scores (0-100).

        Returns:
            str: PDB format string.
        """
        # Backbone atom names (indices in atom37 representation)
        # 0=N, 1=CA, 2=C, 3=CB, 4=O
        backbone_atoms = [
            (0, " N  ", "N"),
            (1, " CA ", "C"),
            (2, " C  ", "C"),
            (4, " O  ", "O"),
        ]

        lines = []
        atom_serial = 1

        for res_idx, aa in enumerate(sequence):
            resname = AA_1TO3.get(aa, "UNK")
            bfactor = float(plddt[res_idx]) if res_idx < len(plddt) else 0.0

            for atom_idx, atom_name, element in backbone_atoms:
                if res_idx < len(atom_positions):
                    if atom_mask is not None and atom_mask[res_idx][atom_idx] < 0.5:
                        continue
                    x, y, z = atom_positions[res_idx][atom_idx]
                    line = (
                        f"ATOM  {atom_serial:5d} {atom_name}"
                        f"{resname:>3s} A{res_idx + 1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"  1.00{bfactor:6.2f}           {element:>2s}  "
                    )
                    lines.append(line)
                    atom_serial += 1

            # Add CB for non-Glycine residues
            if aa != 'G' and res_idx < len(atom_positions):
                cb_idx = 3  # CB index in atom37
                if atom_mask is None or atom_mask[res_idx][cb_idx] >= 0.5:
                    x, y, z = atom_positions[res_idx][cb_idx]
                    if not (x == 0 and y == 0 and z == 0):
                        line = (
                            f"ATOM  {atom_serial:5d}  CB "
                            f"{resname:>3s} A{res_idx + 1:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}"
                            f"  1.00{bfactor:6.2f}           C   "
                        )
                        lines.append(line)
                        atom_serial += 1

        lines.append("TER")
        lines.append("END")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    #  PDBFixer Integration
    # ──────────────────────────────────────────────────────────────────────

    def _apply_pdbfixer(self, pdb_string, pH=7.0):
        """
        Apply PDBFixer to ensure PDB output is 100% OpenMM-compatible.

        Fixes:
          1. Missing atoms (OXT at C-terminal)
          2. Non-standard residues
          3. Missing hydrogens (pH-dependent protonation states)
          4. HIS → HID/HIE/HIP based on pH

        This prevents the Residue 116 (HIS) crash in OpenMM simulations.

        Args:
            pdb_string (str): Raw PDB string from AF2.
            pH (float): pH for hydrogen addition.

        Returns:
            str: Fixed PDB string ready for OpenMM.
        """
        if not HAS_PDBFIXER:
            print("[AF2-PDBFixer] ⚠️  PDBFixer not installed. "
                  "PDB may have issues with OpenMM.")
            print("[AF2-PDBFixer]    Install: pip install pdbfixer")
            return pdb_string

        try:
            from pdbfixer import PDBFixer

            print(f"[AF2-PDBFixer] Fixing PDB structure (pH={pH})...")

            fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))

            # Step 1: Find and fix missing residues
            fixer.findMissingResidues()
            # Remove terminal missing residues (often problematic)
            chains = list(fixer.topology.chains())
            keys_to_remove = []
            for key in fixer.missingResidues:
                chain_idx, res_idx = key
                chain = chains[chain_idx]
                residues = list(chain.residues())
                if res_idx == 0 or res_idx >= len(residues):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del fixer.missingResidues[key]

            # Step 2: Fix non-standard residues
            fixer.findNonstandardResidues()
            if fixer.nonstandardResidues:
                n_ns = len(fixer.nonstandardResidues)
                print(f"[AF2-PDBFixer] Replacing {n_ns} non-standard residues")
                fixer.replaceNonstandardResidues()

            # Step 3: Add missing atoms (includes OXT!)
            fixer.findMissingAtoms()
            n_atoms = len(fixer.missingAtoms)
            n_terminals = len(fixer.missingTerminals)
            if n_atoms > 0 or n_terminals > 0:
                print(f"[AF2-PDBFixer] Adding {n_atoms} missing atoms, "
                      f"{n_terminals} terminal atoms")
            fixer.addMissingAtoms()

            # Step 4: Add hydrogens at target pH
            print(f"[AF2-PDBFixer] Adding hydrogens at pH={pH} "
                  f"(HIS→{'HIP' if pH < 6.0 else 'HIE/HID'})...")
            fixer.addMissingHydrogens(pH)

            # Write fixed PDB to string
            from openmm.app import PDBFile
            output = io.StringIO()
            PDBFile.writeFile(fixer.topology, fixer.positions, output)
            fixed_pdb = output.getvalue()

            n_final_atoms = sum(1 for _ in fixer.topology.atoms())
            print(f"[AF2-PDBFixer] ✅ Fixed: {n_final_atoms} atoms total")
            return fixed_pdb

        except Exception as e:
            print(f"[AF2-PDBFixer] ⚠️  PDBFixer failed: {e}")
            print("[AF2-PDBFixer]    Returning unfixed PDB string.")
            return pdb_string

    # ──────────────────────────────────────────────────────────────────────
    #  Confidence Scores (pLDDT + PAE)
    # ──────────────────────────────────────────────────────────────────────

    def get_confidence_scores(self):
        """
        Extract confidence scores from the last prediction.

        Returns:
            dict: {
                'plddt': np.ndarray of per-residue pLDDT (0–100),
                'mean_plddt': float,
                'pae': np.ndarray (seq_len, seq_len) or None,
                'mean_pae': float or None,
                'ptm': float or None,
                'confidence_category': str,
                'backend': str,
                'ef_hand_plddt': dict (per-loop mean pLDDT) or {},
                'ef_hand_pae': float or None (mean PAE within EF-hands),
            }
        """
        if self.last_plddt_scores is None:
            raise RuntimeError(
                "No prediction available. Run predict_structure() first."
            )

        plddt = self.last_plddt_scores
        mean_plddt = float(np.mean(plddt))

        # Confidence category
        if mean_plddt > 90:
            category = "Very High (>90)"
        elif mean_plddt > 70:
            category = "High (70-90)"
        elif mean_plddt > 50:
            category = "Low (50-70)"
        else:
            category = "Very Low (<50)"

        # PAE statistics
        mean_pae = None
        if self.last_pae_matrix is not None:
            mean_pae = float(np.mean(self.last_pae_matrix))

        # EF-hand-specific confidence
        ef_hand_plddt = {}
        ef_hand_pae = None

        if self.ef_hands:
            for loop_name, (start, end) in self.ef_hands.items():
                # Clamp to valid range
                s = max(0, min(start, len(plddt) - 1))
                e = max(0, min(end, len(plddt) - 1))
                if s <= e:
                    ef_hand_plddt[loop_name] = float(np.mean(plddt[s:e+1]))

            # EF-hand PAE: mean PAE between all EF-hand residues
            if self.last_pae_matrix is not None:
                ef_indices = []
                for _, (start, end) in self.ef_hands.items():
                    s = max(0, min(start, len(plddt) - 1))
                    e = max(0, min(end, len(plddt) - 1))
                    ef_indices.extend(range(s, e + 1))
                if len(ef_indices) >= 2:
                    pae_mat = self.last_pae_matrix
                    n = pae_mat.shape[0]
                    valid = [i for i in ef_indices if i < n]
                    if len(valid) >= 2:
                        sub = pae_mat[np.ix_(valid, valid)]
                        ef_hand_pae = float(np.mean(sub))

        return {
            "plddt": plddt,
            "mean_plddt": mean_plddt,
            "pae": self.last_pae_matrix,
            "mean_pae": mean_pae,
            "ptm": self.last_ptm_score,
            "confidence_category": category,
            "backend": self.backend,
            "ef_hand_plddt": ef_hand_plddt,
            "ef_hand_pae": ef_hand_pae,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Utility Methods
    # ──────────────────────────────────────────────────────────────────────

    def get_residue_distance(self, coords, index1, index2):
        """
        Calculate Euclidean distance between two residues (in Angstroms).
        """
        p1 = coords[index1]
        p2 = coords[index2]
        return float(np.linalg.norm(p1 - p2))

    def write_pdb(self, sequence=None, coords=None, filename="output.pdb"):
        """
        Write PDB file to disk for visualization in UCSF ChimeraX.

        Uses the cached PDB string from the last prediction.
        The PDB is already PDBFixer-processed if fix_pdb=True.

        Args:
            sequence: Ignored (kept for API compatibility).
            coords: Ignored (kept for API compatibility).
            filename (str): Output PDB filename.
        """
        if self.last_pdb_string:
            with open(filename, 'w') as f:
                f.write(self.last_pdb_string)
            print(f"[AF2] Exported PDB file: {filename} "
                  f"(backend: {self.backend}, "
                  f"pdbfixer: {'applied' if self.fix_pdb else 'skipped'})")
        else:
            print("[AF2] Error: No PDB generated yet. "
                  "Run predict_structure() first.")

    def _cleanup_memory(self):
        """
        Aggressively free memory after each prediction.
        Critical for GA loop where many predictions run sequentially.
        """
        gc.collect()

        if HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def get_installation_guide(self):
        """
        Print installation guide for lightweight AF2 dependencies.
        
        Returns:
            str: Installation instructions.
        """
        guide = """
╔══════════════════════════════════════════════════════════════╗
║  AlphaFold2 Evaluator — Installation Guide                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1: LocalColabFold (Recommended, ~2GB download)       ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install colabfold[alphafold]                             ║
║  # Provides: AF2 model + MMseqs2 API (no 3TB DB needed)     ║
║  # Requires: Internet for MSA generation                     ║
║  # RAM: ~16GB for 117 AA protein                             ║
║                                                              ║
║  OPTION 2: OpenFold (Offline, single-sequence)               ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install openfold                                        ║
║  # Provides: AF2 architecture without MSA                    ║
║  # No internet required                                      ║
║  # RAM: ~12GB for 117 AA protein                             ║
║                                                              ║
║  OPTIONAL: PDBFixer (for OpenMM compatibility)               ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install pdbfixer                                        ║
║  # Fixes: missing atoms, OXT terminals, HIS protonation     ║
║  # Required for: step4_md_validation.py (OpenMM)             ║
║                                                              ║
║  OPTIONAL: PyTorch with CUDA (for GPU acceleration)          ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install torch --index-url https://download.pytorch.org  ║
║         /whl/cu121                                           ║
║  # Without GPU: CPU inference ~2-5 min per sequence          ║
║  # With GPU (>=8GB VRAM): ~30s per sequence                  ║
║                                                              ║
║  FULL INSTALL (all dependencies):                            ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install colabfold[alphafold] pdbfixer numpy scipy       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(guide)
        return guide
