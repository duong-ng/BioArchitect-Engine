"""
==============================================================================
 Optimize EF2 & EF4 Binding Pockets for Pr-Lanmodulin
==============================================================================
 Applies targeted mutations to improve Pr3+ coordination at EF2 and EF4,
 then regenerates the structure via ESMFold API, adds PR ions, and runs
 MD relaxation.

 Mutations:
   EF2: K35N, P38D, Q40D (remove charge repulsion, add coordinating O)
   EF4: R96D, A99N (remove Arg charge, add Asn coordinating O)

 Pipeline:
   1. Extract sequence from existing PDB
   2. Apply mutations
   3. ESMFold API -> new 3D structure
   4. Add PR ions at EF-hand centroids
   5. MD relaxation (implicit solvent)
   6. Compare before/after distances

 Usage:
   python optimize_ef_pockets.py
   python optimize_ef_pockets.py --input result/best_Pr_lanmodulin_variant.pdb
==============================================================================
"""

import os
import sys
import argparse

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add src/ to path for ESMFold
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 3-letter to 1-letter amino acid code
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

# Default mutations to optimize EF2 and EF4
DEFAULT_MUTATIONS = {
    # EF2 fixes
    35: ('K', 'N'),  # K35N: remove +1 charge repelling Pr3+
    38: ('P', 'D'),  # P38D: replace rigid Pro with coordinating Asp
    40: ('Q', 'D'),  # Q40D: shorter side chain for closer O
    # EF4 fixes
    96: ('R', 'D'),  # R96D: remove +1 charge, add carboxylate
    99: ('A', 'N'),  # A99N: add coordinating oxygen
}


def extract_sequence(pdb_path):
    """Extract 1-letter amino acid sequence from PDB CA atoms."""
    sequence = []
    seen = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resseq = int(line[22:26])
                if resseq not in seen:
                    seen.add(resseq)
                    resname = line[17:20].strip()
                    aa = AA_3TO1.get(resname, 'X')
                    sequence.append((resseq, aa))
    sequence.sort(key=lambda x: x[0])
    return sequence


def apply_mutations(sequence, mutations):
    """Apply mutations to sequence. Returns (new_seq_str, mutation_log)."""
    seq_dict = {resseq: aa for resseq, aa in sequence}
    log = []

    for pos, (wt, mut) in mutations.items():
        if pos in seq_dict:
            current = seq_dict[pos]
            if current == wt:
                seq_dict[pos] = mut
                log.append(f"  {AA_1TO3[wt]} {pos} -> {AA_1TO3[mut]} ({wt}{pos}{mut})")
            else:
                log.append(f"  WARNING: Expected {wt} at {pos}, found {current}. "
                           f"Mutating {current}->{mut} anyway.")
                seq_dict[pos] = mut
        else:
            log.append(f"  WARNING: Position {pos} not found in sequence!")

    new_seq = "".join(seq_dict[r] for r, _ in sequence)
    return new_seq, log


def run_pipeline(input_pdb, output_dir, mutations, ion="PR"):
    """Full pipeline: mutate -> ESMFold -> add ion -> relax."""
    print("=" * 65)
    print("  EF-Hand Pocket Optimization Pipeline")
    print("=" * 65)

    # ── 1. Extract sequence ──────────────────────────────────────────────
    print("\n[1/5] Extracting sequence from PDB...")
    sequence = extract_sequence(input_pdb)
    original_seq = "".join(aa for _, aa in sequence)
    print(f"  Length: {len(sequence)} residues")
    print(f"  Sequence: {original_seq[:50]}...")

    # ── 2. Apply mutations ───────────────────────────────────────────────
    print("\n[2/5] Applying mutations...")
    mutant_seq, log = apply_mutations(sequence, mutations)
    for line in log:
        print(line)

    n_diff = sum(1 for a, b in zip(original_seq, mutant_seq) if a != b)
    print(f"\n  Total mutations: {n_diff}")
    print(f"  Original: ...{original_seq[33:48]}... (EF2 region)")
    print(f"  Mutant:   ...{mutant_seq[33:48]}... (EF2 region)")
    print(f"  Original: ...{original_seq[88:103]}... (EF4 region)")
    print(f"  Mutant:   ...{mutant_seq[88:103]}... (EF4 region)")

    # ── 3. ESMFold prediction ────────────────────────────────────────────
    print("\n[3/5] Predicting structure via ESMFold API...")
    from step2_esmfold_evaluator import ESMFoldEvaluator

    evaluator = ESMFoldEvaluator(fix_pdb=True, fix_pH=7.0)
    result = evaluator.predict_full(mutant_seq)

    # Save the predicted PDB
    os.makedirs(output_dir, exist_ok=True)
    mutant_pdb = os.path.join(output_dir, f"best_Pr_optimized.pdb")
    evaluator.write_pdb(filename=mutant_pdb)
    print(f"  Saved: {mutant_pdb}")
    print(f"  pLDDT: {result['mean_plddt']:.1f}")

    # ── 4. Add PR ions ───────────────────────────────────────────────────
    print(f"\n[4/5] Adding {ion} ions at EF-hand centroids...")
    from add_ion_to_pdb import add_metal_ions

    ion_pdb = os.path.join(output_dir, f"best_Pr_optimized_with_{ion}.pdb")
    add_metal_ions(mutant_pdb, ion_pdb, ion_symbol=ion, element=ion)

    # ── 5. MD Relaxation ─────────────────────────────────────────────────
    print(f"\n[5/5] Running MD relaxation (implicit solvent)...")
    from md_relax_ion import run_md_relaxation

    relaxed_pdb = os.path.join(output_dir, f"best_Pr_optimized_relaxed.pdb")
    relax_result = run_md_relaxation(
        input_pdb=ion_pdb,
        output_pdb=relaxed_pdb,
        ion_symbol=ion,
        min_rounds=3,
        min_iters=2000,
    )

    if relax_result:
        print(f"\n  Final RMSD: {relax_result['rmsd']:.3f} A")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Mutations applied: {n_diff}")
    for line in log:
        print(f"  {line.strip()}")
    print(f"\n  Output files:")
    print(f"    Mutant PDB:  {mutant_pdb}")
    print(f"    With ions:   {ion_pdb}")
    print(f"    Relaxed:     {relaxed_pdb}")
    print(f"    pLDDT:       {result['mean_plddt']:.1f}")
    if relax_result:
        print(f"    RMSD:        {relax_result['rmsd']:.3f} A")
    print("=" * 65)

    # ChimeraX commands
    abs_relaxed = os.path.abspath(relaxed_pdb)
    print(f"\n  -- ChimeraX Commands --")
    print(f'  open "{abs_relaxed}"')
    print(f"  select /B; style sel sphere; color sel gold; size sel atomRadius 1.5")
    print(f"  distance #1/B:2@{ion} #1/A:37@OD1")
    print(f"  distance #1/B:2@{ion} #1/A:38@OD1")
    print(f"  distance #1/B:4@{ion} #1/A:96@OD1")
    print()

    return {
        "mutant_pdb": mutant_pdb,
        "ion_pdb": ion_pdb,
        "relaxed_pdb": relaxed_pdb,
        "plddt": result["mean_plddt"],
        "rmsd": relax_result["rmsd"] if relax_result else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize EF2/EF4 binding pockets for Pr-lanmodulin")
    parser.add_argument("--input", "-i",
        default="result/best_Pr_lanmodulin_variant.pdb",
        help="Input PDB file (Pr variant)")
    parser.add_argument("--output-dir", "-d",
        default="result",
        help="Output directory (default: result/)")
    parser.add_argument("--ion", default="PR",
        help="Ion symbol (default: PR)")
    args = parser.parse_args()

    run_pipeline(
        input_pdb=args.input,
        output_dir=args.output_dir,
        mutations=DEFAULT_MUTATIONS,
        ion=args.ion.upper(),
    )
