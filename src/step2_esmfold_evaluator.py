"""
==============================================================================
 BioArchitect Engine — Step 2: ESMFold Protein Structure Evaluator (API)
==============================================================================
 Predicts 3D protein structures using the ESMFold v1 REST API hosted at
 ESM Metagenomic Atlas (https://api.esmatlas.com).

 Backend:
   ESMFold v1 REST API — no local GPU, no model download, no heavy deps.
   POST sequence → receive PDB string with pLDDT in B-factor column.

 Key features:
   - Drop-in replacement for AlphaFold2Evaluator (identical public API)
   - Zero local compute: all folding runs on Meta's cloud infrastructure
   - Only requires `requests` + `numpy` (no transformers, no torch)
   - pLDDT extraction from B-factor column of returned PDB
   - PDBFixer integration for 100% OpenMM-compatible PDB output
   - Retry logic with exponential backoff for API reliability

 LanRecov Integration:
   - EF-hand-specific confidence scoring (pLDDT per loop region)
   - PDB output pre-fixed for terminal atoms (OXT), hydrogens, HIS states
   - PAE returns None (ESMFold API does not provide PAE);
     the GA defaults to a neutral 0.5 confidence bonus when PAE is None.
   - pTM returns None (not provided by the public PDB endpoint)

 API Contract (identical to AlphaFold2Evaluator):
   - predict_structure(sequence) → np.ndarray (seq_len, 3) CA coords
   - predict_full(sequence) → dict with ca_coords, pdb_string, plddt, etc.
   - get_confidence_scores() → dict with plddt, pae, ptm, ef_hand scoring
   - write_pdb(sequence, coords, filename) → writes cached PDB to disk
==============================================================================
"""

import numpy as np
import os
import sys
import io
import gc
import time
import warnings

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ─── Dependency detection ───────────────────────────────────────────────────

def _detect_requests():
    """Check if the requests library is available."""
    try:
        import requests
        return True
    except ImportError:
        return False


def _detect_pdbfixer():
    """Check if PDBFixer is available for PDB post-processing."""
    try:
        from pdbfixer import PDBFixer
        return True
    except ImportError:
        return False


# Cache detection results at module level
HAS_REQUESTS = _detect_requests()
HAS_PDBFIXER = _detect_pdbfixer()


# ─── Three-letter amino acid code mapping ───────────────────────────────────
AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'UNK',
}


class ESMFoldEvaluator:
    """
    ESMFold-based Protein Structure Evaluator for BioArchitect Engine.
    Uses the ESMFold v1 REST API for cloud-based single-sequence folding.

    Drop-in replacement for AlphaFold2Evaluator with identical public API.

    Advantages:
      - Zero local compute — all folding on Meta's cloud
      - No model download (~3GB saved), no torch/transformers required
      - Only needs `requests` and `numpy`
      - Suitable for CPU-only environments

    Limitations:
      - Requires internet connection
      - API rate limits may apply for high-throughput GA loops
      - PAE and pTM not provided by the public API endpoint
      - Max sequence length ~400 AA for reliable API response

    Key features for LanRecov:
      - PDBFixer integration → OpenMM-ready PDB output (fixes OXT, HIS, H)
      - Retry logic with exponential backoff for API robustness
      - EF-hand-specific pLDDT scoring
    """

    # ESMFold public API endpoint
    API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    # API limits
    MAX_SEQUENCE_LENGTH = 400  # API recommended max for reliable responses

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2.0       # seconds
    BACKOFF_MULTIPLIER = 2.0    # exponential factor
    REQUEST_TIMEOUT = 120       # seconds per request

    def __init__(self, device='cpu', ef_hands=None, fix_pdb=True, fix_pH=7.0):
        """
        Initialize ESMFold API evaluator.

        Args:
            device (str): Ignored — API runs on Meta's cloud.
                          Kept for API compatibility with AlphaFold2Evaluator.
            ef_hands (dict): EF-hand loop definitions {name: (start, end)}.
                Used for region-specific confidence scoring.
            fix_pdb (bool): If True, run PDBFixer after every prediction.
            fix_pH (float): pH for PDBFixer hydrogen addition.
        """
        self.ef_hands = ef_hands or {}
        self.fix_pdb = fix_pdb
        self.fix_pH = fix_pH
        self.backend = "esmfold_api"

        # Cached prediction results
        self.last_pdb_string = None
        self.last_plddt_scores = None
        self.last_pae_matrix = None      # Always None for ESMFold API
        self.last_ptm_score = None       # Not provided by API

        # Report initialization
        self._report_init()

    def _report_init(self):
        """Print initialization status."""
        print("[ESMFold-API] Initializing ESMFold API evaluator...")
        print(f"[ESMFold-API]   endpoint={self.API_URL}")
        print(f"[ESMFold-API]   fix_pdb={self.fix_pdb}, "
              f"max_retries={self.MAX_RETRIES}, "
              f"timeout={self.REQUEST_TIMEOUT}s")

        if HAS_REQUESTS:
            print("[ESMFold-API] ✅ Backend: ESMFold v1 REST API")
            print("[ESMFold-API]    Cloud-based — no local GPU/model needed")
            print("[ESMFold-API]    Requires internet connection")
        else:
            print("[ESMFold-API] ❌ 'requests' library not available!")
            print("[ESMFold-API]    Install: pip install requests")

    # ──────────────────────────────────────────────────────────────────────
    #  Public API: predict_structure (drop-in compatible)
    # ──────────────────────────────────────────────────────────────────────

    def predict_structure(self, sequence):
        """
        Predict 3D protein structure using ESMFold API.
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
                'pae': None (ESMFold API does not provide PAE),
                'ptm': None (ESMFold API does not provide pTM),
                'backend': 'esmfold_api',
            }
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join(c if c in valid_aa else "X" for c in sequence)

        # Warn about long sequences
        if len(sequence) > self.MAX_SEQUENCE_LENGTH:
            print(f"[ESMFold-API] ⚠️  Sequence length {len(sequence)} exceeds "
                  f"recommended API max ({self.MAX_SEQUENCE_LENGTH}). "
                  f"Request may fail or be slow.")

        print(f"[ESMFold-API] Predicting structure for sequence "
              f"(length={len(sequence)}, backend={self.backend})...")

        result = self._predict_via_api(sequence)

        # Cache results for get_confidence_scores()
        self.last_pdb_string = result["pdb_string"]
        self.last_plddt_scores = result["plddt"]
        self.last_pae_matrix = None
        self.last_ptm_score = None

        # Optionally fix PDB with PDBFixer
        if self.fix_pdb and result["pdb_string"]:
            result["pdb_string"] = self._apply_pdbfixer(
                result["pdb_string"], self.fix_pH
            )
            self.last_pdb_string = result["pdb_string"]

        print(f"[ESMFold-API] ✅ Structure predicted | "
              f"mean pLDDT={result['mean_plddt']:.1f} | "
              f"backend={self.backend}")

        return result

    # ──────────────────────────────────────────────────────────────────────
    #  ESMFold REST API Backend
    # ──────────────────────────────────────────────────────────────────────

    def _predict_via_api(self, sequence):
        """
        Call the ESMFold v1 REST API to predict protein structure.

        Sends a POST request with the amino acid sequence as the body.
        The API returns a PDB-format string with pLDDT in the B-factor
        column.

        Implements exponential backoff retry logic for robustness against
        transient network errors and API rate limiting.

        Args:
            sequence (str): Cleaned amino acid sequence.

        Returns:
            dict: Prediction results with ca_coords, pdb_string, plddt,
                  mean_plddt, pae (None), ptm (None), and backend.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if not HAS_REQUESTS:
            raise RuntimeError(
                "[ESMFold-API] 'requests' library not available. "
                "Install: pip install requests"
            )

        import requests

        last_error = None
        backoff = self.INITIAL_BACKOFF

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"[ESMFold-API] Sending request to API "
                      f"(attempt {attempt}/{self.MAX_RETRIES})...")

                response = requests.post(
                    self.API_URL,
                    data=sequence,
                    headers={"Content-Type": "text/plain"},
                    timeout=self.REQUEST_TIMEOUT,
                    verify=True,
                )

                # Check for HTTP errors
                if response.status_code == 429:
                    # Rate limited — wait and retry
                    retry_after = float(
                        response.headers.get("Retry-After", backoff)
                    )
                    print(f"[ESMFold-API] ⚠️  Rate limited (HTTP 429). "
                          f"Waiting {retry_after:.1f}s...")
                    time.sleep(retry_after)
                    backoff *= self.BACKOFF_MULTIPLIER
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"API returned HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                pdb_string = response.text

                # Validate response — should contain ATOM records
                if "ATOM" not in pdb_string:
                    raise RuntimeError(
                        f"API returned invalid PDB (no ATOM records). "
                        f"Response preview: {pdb_string[:200]}"
                    )

                # Parse CA coordinates and pLDDT from the PDB
                coords_list, plddt_list = self._parse_pdb_string(pdb_string)

                if not coords_list:
                    raise RuntimeError(
                        "Failed to parse CA coordinates from API response."
                    )

                ca_coords = np.array(coords_list)
                plddt = np.array(plddt_list)
                mean_plddt = float(np.mean(plddt))

                print(f"[ESMFold-API] ✅ API response received | "
                      f"{len(coords_list)} residues | "
                      f"mean pLDDT={mean_plddt:.1f}")

                return {
                    "ca_coords": ca_coords,
                    "pdb_string": pdb_string,
                    "plddt": plddt,
                    "mean_plddt": mean_plddt,
                    "pae": None,
                    "ptm": None,
                    "backend": "esmfold_api",
                }

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.REQUEST_TIMEOUT}s"
                print(f"[ESMFold-API] ⚠️  {last_error}")

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                print(f"[ESMFold-API] ⚠️  {last_error}")

            except RuntimeError as e:
                last_error = str(e)
                print(f"[ESMFold-API] ⚠️  {last_error}")

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                print(f"[ESMFold-API] ⚠️  {last_error}")

            # Wait before retry (exponential backoff)
            if attempt < self.MAX_RETRIES:
                print(f"[ESMFold-API]    Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= self.BACKOFF_MULTIPLIER

        # All retries exhausted
        raise RuntimeError(
            f"[ESMFold-API] All {self.MAX_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        )

    # ──────────────────────────────────────────────────────────────────────
    #  PDB Parsing
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdb_string(pdb_string):
        """
        Parse PDB format string to extract CA coordinates and pLDDT.

        The ESMFold API returns a standard PDB file where the B-factor
        column contains the per-residue pLDDT confidence score (0–100).

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
            pdb_string (str): Raw PDB string from ESMFold API.
            pH (float): pH for hydrogen addition.

        Returns:
            str: Fixed PDB string ready for OpenMM.
        """
        if not HAS_PDBFIXER:
            print("[ESMFold-PDBFixer] ⚠️  PDBFixer not installed. "
                  "PDB may have issues with OpenMM.")
            print("[ESMFold-PDBFixer]    Install: pip install pdbfixer")
            return pdb_string

        try:
            from pdbfixer import PDBFixer

            print(f"[ESMFold-PDBFixer] Fixing PDB structure (pH={pH})...")

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
                print(f"[ESMFold-PDBFixer] Replacing {n_ns} non-standard residues")
                fixer.replaceNonstandardResidues()

            # Step 3: Add missing atoms (includes OXT!)
            fixer.findMissingAtoms()
            n_atoms = len(fixer.missingAtoms)
            n_terminals = len(fixer.missingTerminals)
            if n_atoms > 0 or n_terminals > 0:
                print(f"[ESMFold-PDBFixer] Adding {n_atoms} missing atoms, "
                      f"{n_terminals} terminal atoms")
            fixer.addMissingAtoms()

            # Step 4: Add hydrogens at target pH
            print(f"[ESMFold-PDBFixer] Adding hydrogens at pH={pH} "
                  f"(HIS→{'HIP' if pH < 6.0 else 'HIE/HID'})...")
            fixer.addMissingHydrogens(pH)

            # Write fixed PDB to string
            from openmm.app import PDBFile
            output = io.StringIO()
            PDBFile.writeFile(fixer.topology, fixer.positions, output)
            fixed_pdb = output.getvalue()

            n_final_atoms = sum(1 for _ in fixer.topology.atoms())
            print(f"[ESMFold-PDBFixer] ✅ Fixed: {n_final_atoms} atoms total")
            return fixed_pdb

        except Exception as e:
            print(f"[ESMFold-PDBFixer] ⚠️  PDBFixer failed: {e}")
            print("[ESMFold-PDBFixer]    Returning unfixed PDB string.")
            return pdb_string

    # ──────────────────────────────────────────────────────────────────────
    #  Confidence Scores (pLDDT)
    # ──────────────────────────────────────────────────────────────────────

    def get_confidence_scores(self):
        """
        Extract confidence scores from the last prediction.

        Returns:
            dict: {
                'plddt': np.ndarray of per-residue pLDDT (0–100),
                'mean_plddt': float,
                'pae': None (ESMFold API does not provide PAE),
                'mean_pae': None,
                'ptm': None (ESMFold API does not provide pTM),
                'confidence_category': str,
                'backend': 'esmfold_api',
                'ef_hand_plddt': dict (per-loop mean pLDDT) or {},
                'ef_hand_pae': None,
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

        # EF-hand-specific confidence
        ef_hand_plddt = {}

        if self.ef_hands:
            for loop_name, (start, end) in self.ef_hands.items():
                # Clamp to valid range
                s = max(0, min(start, len(plddt) - 1))
                e = max(0, min(end, len(plddt) - 1))
                if s <= e:
                    ef_hand_plddt[loop_name] = float(np.mean(plddt[s:e+1]))

        return {
            "plddt": plddt,
            "mean_plddt": mean_plddt,
            "pae": None,
            "mean_pae": None,
            "ptm": None,
            "confidence_category": category,
            "backend": self.backend,
            "ef_hand_plddt": ef_hand_plddt,
            "ef_hand_pae": None,
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
            print(f"[ESMFold-API] Exported PDB file: {filename} "
                  f"(backend: {self.backend}, "
                  f"pdbfixer: {'applied' if self.fix_pdb else 'skipped'})")
        else:
            print("[ESMFold-API] Error: No PDB generated yet. "
                  "Run predict_structure() first.")

    def _cleanup_memory(self):
        """
        Free memory after each prediction.
        Lightweight for API mode — no GPU tensors to free.
        """
        gc.collect()

    def get_installation_guide(self):
        """
        Print installation guide for ESMFold API dependencies.

        Returns:
            str: Installation instructions.
        """
        guide = """
╔══════════════════════════════════════════════════════════════╗
║  ESMFold API Evaluator — Installation Guide                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CORE: ESMFold v1 REST API (cloud-based)                     ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install requests numpy                                  ║
║  # Zero model download — all folding runs on Meta's cloud    ║
║  # Requires internet connection                              ║
║  # Typical response time: 5-30s per sequence                 ║
║                                                              ║
║  OPTIONAL: PDBFixer (for OpenMM compatibility)               ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install pdbfixer                                        ║
║  # Fixes: missing atoms, OXT terminals, HIS protonation     ║
║  # Required for: step4_md_validation.py (OpenMM)             ║
║                                                              ║
║  FULL INSTALL (all dependencies):                            ║
║  ─────────────────────────────────────────────────────────    ║
║  pip install requests numpy scipy pdbfixer                   ║
║                                                              ║
║  NOTE: No PyTorch or transformers needed!                    ║
║  The API handles all computation on Meta's servers.          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(guide)
        return guide
