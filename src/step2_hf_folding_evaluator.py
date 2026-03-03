"""
==============================================================================
 BioArchitect Engine — Step 2: HuggingFace ESMForProteinFolding Evaluator
==============================================================================
 Sử dụng 100% HuggingFace Transformers (EsmForProteinFolding + AutoTokenizer)
 thay vì fair-esm. Đầy đủ tokenizer, decode, phân tích chuỗi protein,
 fold 3D, confidence scores, và export PDB.

 Model: facebook/esmfold_v1 (ESM-2 stem + protein folding head)
 Yêu cầu: pip install transformers torch
==============================================================================
"""

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# ─── Amino Acid Lookup Table ────────────────────────────────────────────────
AMINO_ACID_NAMES = {
    'A': 'Alanine',       'R': 'Arginine',      'N': 'Asparagine',
    'D': 'Aspartate',     'C': 'Cysteine',      'E': 'Glutamate',
    'Q': 'Glutamine',     'G': 'Glycine',       'H': 'Histidine',
    'I': 'Isoleucine',    'L': 'Leucine',       'K': 'Lysine',
    'M': 'Methionine',    'F': 'Phenylalanine', 'P': 'Proline',
    'S': 'Serine',        'T': 'Threonine',     'W': 'Tryptophan',
    'Y': 'Tyrosine',      'V': 'Valine',        'X': 'Unknown',
}

# Standard amino acid one-letter codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ─── Conversion utility (atom37 → PDB format) ──────────────────────────────
# Atom names in atom37 representation (AlphaFold/OpenFold convention)
ATOM37_NAMES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]

THREE_LETTER = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'UNK',
}


class HFEsmFoldEvaluator:
    """
    HuggingFace ESMForProteinFolding Evaluator.
    
    Drop-in replacement for ESMFoldEvaluator that uses the official
    HuggingFace Transformers library instead of fair-esm.
    
    Features:
      - Tokenize / Decode protein sequences with EsmTokenizer
      - Fold proteins using EsmForProteinFolding (facebook/esmfold_v1)
      - Extract pLDDT and pTM confidence scores
      - Export 3D coordinates (CA atoms) compatible with BioArchitect GA pipeline
      - Generate PDB files for ChimeraX visualization
    """

    # ────────────────────────────────────────────────────────────────────────
    #  Initialization
    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, device='cuda', model_name="facebook/esmfold_v1",
                 half_precision=True):
        """
        Load HuggingFace EsmForProteinFolding and its tokenizer.

        Args:
            device (str): 'cuda' or 'cpu'. Falls back to CPU if no GPU.
            model_name (str): HuggingFace model identifier.
            half_precision (bool): Use float16 to reduce VRAM usage (~8GB vs ~16GB).
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.half_precision = half_precision and (self.device == 'cuda')

        print(f"[HF-ESMFold] Loading tokenizer from '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[HF-ESMFold] Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")

        print(f"[HF-ESMFold] Loading EsmForProteinFolding on {self.device}...")
        print(f"[HF-ESMFold] This requires significant RAM/VRAM. Please wait...")
        self.model = EsmForProteinFolding.from_pretrained(model_name)

        if self.half_precision:
            self.model = self.model.half()
            print("[HF-ESMFold] Using float16 (half precision) to save VRAM.")

        self.model = self.model.eval().to(self.device)
        print(f"[HF-ESMFold] ✅ Model loaded successfully on {self.device}!")

        # Cache for the most recent prediction
        self.last_output = None
        self.last_pdb_string = None
        self.last_sequence = None

    # ────────────────────────────────────────────────────────────────────────
    #  Tokenizer Functions
    # ────────────────────────────────────────────────────────────────────────
    def tokenize_sequence(self, sequence, return_tensors="pt"):
        """
        Tokenize a protein sequence using HuggingFace EsmTokenizer.

        Args:
            sequence (str): Amino acid sequence (e.g. "MLKNVQVQLV")
            return_tensors (str): Return format ("pt" for PyTorch tensors)

        Returns:
            dict: Contains 'input_ids' and 'attention_mask' tensors.
        """
        # ESMFold requires add_special_tokens=False 
        # (no <cls>/<eos> wrapping — the model handles this internally)
        tokens = self.tokenizer(
            [sequence],
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=False,
        )
        return tokens

    def tokenize_batch(self, sequences, return_tensors="pt"):
        """
        Tokenize multiple protein sequences with padding.

        Args:
            sequences (list[str]): List of amino acid sequences.
            return_tensors (str): Return format.

        Returns:
            dict: Batched 'input_ids' and 'attention_mask' tensors.
        """
        tokens = self.tokenizer(
            sequences,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        return tokens

    def decode_tokens(self, input_ids):
        """
        Decode token IDs back to amino acid sequence string.

        Args:
            input_ids (torch.Tensor or list): Token IDs from tokenizer.
                Shape: (batch_size, seq_len) or (seq_len,)

        Returns:
            str or list[str]: Decoded amino acid sequence(s).
        """
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                # Batch decode
                return [
                    self.tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
                    for ids in input_ids
                ]
            input_ids = input_ids.tolist()
        
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        # Remove spaces (ESM tokenizer adds spaces between residues)
        return decoded.replace(" ", "")

    def analyze_tokenization(self, sequence):
        """
        Detailed analysis of how a protein sequence is tokenized.
        Maps each residue to its token ID and full amino acid name.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            list[dict]: Per-residue tokenization info with keys:
                - position (int): 0-indexed position in sequence
                - residue (str): One-letter amino acid code
                - token_id (int): Tokenizer vocabulary ID
                - amino_acid_name (str): Full amino acid name
                - is_standard (bool): Whether it's a standard amino acid
        """
        tokens = self.tokenize_sequence(sequence)
        input_ids = tokens["input_ids"][0].tolist()

        analysis = []
        for i, (residue, token_id) in enumerate(zip(sequence, input_ids)):
            analysis.append({
                "position": i,
                "residue": residue,
                "token_id": token_id,
                "amino_acid_name": AMINO_ACID_NAMES.get(residue, "Unknown"),
                "is_standard": residue in STANDARD_AA,
            })
        
        return analysis

    def get_vocab_info(self):
        """
        Return tokenizer vocabulary information.
        
        Returns:
            dict: Vocabulary metadata including size, special tokens, and
                  amino acid token IDs.
        """
        vocab = self.tokenizer.get_vocab()
        special_tokens = {
            "cls_token": self.tokenizer.cls_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "mask_token": self.tokenizer.mask_token,
            "unk_token": self.tokenizer.unk_token,
        }
        
        # Map standard amino acids to their token IDs
        aa_tokens = {}
        for aa in STANDARD_AA:
            if aa in vocab:
                aa_tokens[aa] = vocab[aa]

        return {
            "vocab_size": len(vocab),
            "special_tokens": special_tokens,
            "amino_acid_token_ids": aa_tokens,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Structure Prediction (Folding)
    # ────────────────────────────────────────────────────────────────────────
    def predict_structure(self, sequence):
        """
        Predict the 3D structure of a protein sequence.
        Extracts Alpha Carbon (CA) coordinates — compatible with
        BioArchitect GA pipeline (same return type as ESMFoldEvaluator).

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            np.ndarray: CA coordinates, shape (seq_len, 3), in Angstroms.
        """
        # Tokenize
        tokens = self.tokenize_sequence(sequence)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Fold
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        self.last_output = output
        self.last_sequence = sequence

        # Extract atom37 positions: shape (batch, seq_len, 37, 3)
        # CA is atom index 1 in atom37 representation
        positions = output.positions[-1]  # Last recycling step
        ca_coords = positions[0, :, 1, :].cpu().float().numpy()  # (seq_len, 3)

        # Also generate PDB string for later export
        self._generate_pdb_string(sequence, output)

        return ca_coords

    def predict_full(self, sequence):
        """
        Full prediction returning all atom positions and metadata.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            dict: Complete prediction results including:
                - ca_coords (np.ndarray): CA coordinates (seq_len, 3)
                - all_atom_positions (np.ndarray): All atom37 positions (seq_len, 37, 3)
                - all_atom_mask (np.ndarray): Which atoms exist (seq_len, 37)
                - plddt (np.ndarray): Per-residue confidence (seq_len,)
                - ptm (float): Predicted TM-score
                - pdb_string (str): PDB format string
        """
        # Tokenize
        tokens = self.tokenize_sequence(sequence)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        self.last_output = output
        self.last_sequence = sequence

        # Extract data from output
        positions = output.positions[-1]  # (batch, seq_len, 37, 3)
        atom_mask = output.atom37_atom_exists[0].cpu().float().numpy()
        ca_coords = positions[0, :, 1, :].cpu().float().numpy()
        all_positions = positions[0].cpu().float().numpy()

        # Confidence scores
        plddt = output.plddt[0, :, 1].cpu().float().numpy()  # pLDDT for CA atoms
        ptm = output.ptm.item() if output.ptm is not None else None

        # PDB
        pdb_string = self._generate_pdb_string(sequence, output)

        return {
            "ca_coords": ca_coords,
            "all_atom_positions": all_positions,
            "all_atom_mask": atom_mask,
            "plddt": plddt,
            "ptm": ptm,
            "pdb_string": pdb_string,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  Confidence Scores
    # ────────────────────────────────────────────────────────────────────────
    def get_confidence_scores(self):
        """
        Extract per-residue pLDDT and global pTM from the last prediction.

        Returns:
            dict: {
                'plddt': np.ndarray of per-residue scores (0–100),
                'mean_plddt': float,
                'ptm': float or None,
                'confidence_category': str ('Very High'/'High'/'Low'/'Very Low')
            }
        """
        if self.last_output is None:
            raise RuntimeError("No prediction available. Run predict_structure() first.")

        plddt = self.last_output.plddt[0, :, 1].cpu().float().numpy()
        ptm = self.last_output.ptm.item() if self.last_output.ptm is not None else None
        mean_plddt = float(np.mean(plddt))

        # AlphaFold-style confidence categories
        if mean_plddt > 90:
            category = "Very High (>90)"
        elif mean_plddt > 70:
            category = "High (70-90)"
        elif mean_plddt > 50:
            category = "Low (50-70)"
        else:
            category = "Very Low (<50)"

        return {
            "plddt": plddt,
            "mean_plddt": mean_plddt,
            "ptm": ptm,
            "confidence_category": category,
        }

    # ────────────────────────────────────────────────────────────────────────
    #  PDB Generation & Export
    # ────────────────────────────────────────────────────────────────────────
    def _generate_pdb_string(self, sequence, output):
        """
        Convert model output to PDB format string using atom37 representation.

        This manually builds PDB ATOM records from the predicted coordinates,
        providing full compatibility without requiring openfold installation.
        """
        positions = output.positions[-1]  # (batch, seq_len, 37, 3)
        atom_mask = output.atom37_atom_exists  # (batch, seq_len, 37)

        coords = positions[0].cpu().float().numpy()    # (seq_len, 37, 3)
        mask = atom_mask[0].cpu().float().numpy()       # (seq_len, 37)

        # Try to get pLDDT for B-factor column
        try:
            plddt = output.plddt[0].cpu().float().numpy()  # (seq_len, 37)
        except Exception:
            plddt = np.ones_like(mask) * 50.0  # default fallback

        pdb_lines = []
        atom_serial = 1

        for res_idx in range(len(sequence)):
            residue = sequence[res_idx]
            res_name = THREE_LETTER.get(residue, "UNK")
            res_seq = res_idx + 1  # 1-indexed

            for atom_idx in range(37):
                if mask[res_idx, atom_idx] < 0.5:
                    continue  # Atom doesn't exist for this residue

                atom_name = ATOM37_NAMES[atom_idx] if atom_idx < len(ATOM37_NAMES) else f"X{atom_idx}"
                x, y, z = coords[res_idx, atom_idx]
                b_factor = float(plddt[res_idx, atom_idx]) if atom_idx < plddt.shape[-1] else 50.0

                # Standard PDB ATOM record format (columns are fixed-width)
                # ATOM serial name altLoc resName chain resSeq iCode x y z occ bfactor element
                atom_name_padded = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
                element = atom_name[0]

                pdb_lines.append(
                    f"ATOM  {atom_serial:5d} {atom_name_padded:4s} "
                    f"{res_name:>3s} A{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.0:6.2f}{b_factor:6.2f}          "
                    f"{element:>2s}  "
                )
                atom_serial += 1

        pdb_lines.append("TER")
        pdb_lines.append("END")

        pdb_string = "\n".join(pdb_lines)
        self.last_pdb_string = pdb_string
        return pdb_string

    def write_pdb(self, sequence=None, coords=None, filename="output.pdb"):
        """
        Write PDB file to disk for visualization in ChimeraX.

        If called after predict_structure(), uses the cached PDB string.
        Otherwise, folds the given sequence first.

        Args:
            sequence (str, optional): Amino acid sequence. If None, uses last prediction.
            coords: Ignored (kept for API compatibility with ESMFoldEvaluator).
            filename (str): Output PDB filename.
        """
        if sequence is not None and sequence != self.last_sequence:
            # Need to fold this sequence first
            self.predict_structure(sequence)

        if self.last_pdb_string:
            with open(filename, 'w') as f:
                f.write(self.last_pdb_string)
            print(f"[HF-ESMFold] ✅ Exported PDB file: {filename}")
        else:
            print("[HF-ESMFold] ❌ No PDB data. Run predict_structure() first.")

    # ────────────────────────────────────────────────────────────────────────
    #  Geometry Utilities
    # ────────────────────────────────────────────────────────────────────────
    def get_residue_distance(self, coords, index1, index2):
        """
        Calculate Euclidean distance between two residues (in Angstroms).
        Compatible with BioArchitect GA pipeline.

        Args:
            coords (np.ndarray): CA coordinates from predict_structure()
            index1 (int): First residue index (0-based)
            index2 (int): Second residue index (0-based)

        Returns:
            float: Distance in Angstroms.
        """
        p1 = coords[index1]
        p2 = coords[index2]
        return float(np.linalg.norm(p1 - p2))

    # ────────────────────────────────────────────────────────────────────────
    #  All-in-One Pipeline
    # ────────────────────────────────────────────────────────────────────────
    def predict_and_analyze(self, sequence, pdb_filename=None):
        """
        Complete analysis pipeline: tokenize → fold → confidence → PDB.

        Args:
            sequence (str): Amino acid sequence.
            pdb_filename (str, optional): If given, export PDB to this file.

        Returns:
            dict: Full analysis results including tokenization, coordinates,
                  confidence scores, and optionally the PDB file path.
        """
        print(f"\n{'='*60}")
        print(f"[HF-ESMFold] Full Analysis Pipeline")
        print(f"{'='*60}")

        # 1. Tokenization
        print(f"\n[Step 1] Tokenizing sequence (length={len(sequence)})...")
        tokens = self.tokenize_sequence(sequence)
        input_ids = tokens["input_ids"][0].tolist()
        print(f"  → Generated {len(input_ids)} tokens")

        decoded = self.decode_tokens(tokens["input_ids"])
        match = decoded[0] == sequence if isinstance(decoded, list) else decoded == sequence
        print(f"  → Decode verification: {'✅ MATCH' if match else '❌ MISMATCH'}")

        # 2. Folding
        print(f"\n[Step 2] Folding protein structure...")
        result = self.predict_full(sequence)
        ca_coords = result["ca_coords"]
        print(f"  → Extracted {len(ca_coords)} CA atom positions")

        # 3. Confidence
        print(f"\n[Step 3] Confidence scores:")
        confidence = self.get_confidence_scores()
        print(f"  → Mean pLDDT: {confidence['mean_plddt']:.1f}")
        print(f"  → pTM: {confidence['ptm']:.4f}" if confidence['ptm'] else "  → pTM: N/A")
        print(f"  → Category: {confidence['confidence_category']}")

        # 4. PDB Export
        pdb_path = None
        if pdb_filename:
            print(f"\n[Step 4] Exporting PDB to '{pdb_filename}'...")
            self.write_pdb(filename=pdb_filename)
            pdb_path = pdb_filename

        print(f"\n{'='*60}")
        print(f"[HF-ESMFold] ✅ Analysis Complete!")
        print(f"{'='*60}\n")

        return {
            "sequence": sequence,
            "sequence_length": len(sequence),
            "token_ids": input_ids,
            "decoded_sequence": decoded[0] if isinstance(decoded, list) else decoded,
            "ca_coords": ca_coords,
            "all_atom_positions": result["all_atom_positions"],
            "plddt": confidence["plddt"],
            "mean_plddt": confidence["mean_plddt"],
            "ptm": confidence["ptm"],
            "confidence_category": confidence["confidence_category"],
            "pdb_string": result["pdb_string"],
            "pdb_file": pdb_path,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Quick Test (when run directly)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  HuggingFace ESMForProteinFolding — Quick Self-Test")
    print("=" * 60)

    # Lanmodulin 6MI5 sequence
    LANMODULIN = (
        "PTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGR"
        "VSEADLKKLDPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIRHHHHHH"
    )

    print(f"\nSequence: {LANMODULIN}")
    print(f"Length: {len(LANMODULIN)} residues")

    try:
        evaluator = HFEsmFoldEvaluator(device='cpu', half_precision=False)

        # Test tokenization
        tokens = evaluator.tokenize_sequence(LANMODULIN)
        print(f"\nToken IDs shape: {tokens['input_ids'].shape}")

        decoded = evaluator.decode_tokens(tokens['input_ids'])
        decoded_str = decoded[0] if isinstance(decoded, list) else decoded
        print(f"Decoded back: {decoded_str}")
        print(f"Match: {decoded_str == LANMODULIN}")

        # Test full analysis
        analysis = evaluator.analyze_tokenization(LANMODULIN)
        print(f"\nFirst 10 residue tokens:")
        for entry in analysis[:10]:
            print(f"  [{entry['position']:3d}] {entry['residue']} → "
                  f"token_id={entry['token_id']:3d} ({entry['amino_acid_name']})")

        # Test folding
        print("\nAttempting structure prediction...")
        result = evaluator.predict_and_analyze(
            LANMODULIN,
            pdb_filename="lanmodulin_hf_test.pdb"
        )
        print(f"CA coords shape: {result['ca_coords'].shape}")

    except Exception as e:
        print(f"\n⚠️  Error during test: {e}")
        print("This usually means insufficient RAM/VRAM for the model.")
        print("The tokenizer functions still work without the full model.")
