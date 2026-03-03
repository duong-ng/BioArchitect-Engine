"""
==============================================================================
 BioArchitect Engine — Lanmodulin HuggingFace ESMFold Demo
==============================================================================
 Ready-to-run demo script showcasing ESMForProteinFolding with the
 Lanmodulin protein (PDB: 6MI5) from Methylobacterium extorquens.
 
 Features demonstrated:
   1. Tokenize & Decode Lanmodulin sequence
   2. Detailed residue-level tokenization analysis
   3. Vocabulary inspection
   4. 3D structure prediction (folding)
   5. Confidence score extraction (pLDDT, pTM)
   6. PDB file export for ChimeraX
   7. EF-hand loop distance measurements
   8. Multi-variant tokenization comparison

 Usage:
   python run_hf_lanmodulin_demo.py
   
 Requirements:
   pip install torch transformers numpy
   
 Note: Full folding requires ~16GB RAM (CPU) or a GPU with enough VRAM.
       Tokenizer-only functions work on any machine.
==============================================================================
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step2_hf_folding_evaluator import HFEsmFoldEvaluator, AMINO_ACID_NAMES


# ═══════════════════════════════════════════════════════════════════════════
#  Lanmodulin Data (PDB: 6MI5)
# ═══════════════════════════════════════════════════════════════════════════

# Wild-type Lanmodulin from Methylobacterium extorquens AM1
LANMODULIN_WT = (
    "PTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGR"
    "VSEADLKKLDPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIRHHHHHH"
)

# EF-hand calcium/lanthanide binding loops (0-indexed residue ranges)
EF_HANDS = {
    "EF1": (12, 23),   # First EF-hand loop
    "EF2": (35, 46),   # Second EF-hand loop
    "EF3": (63, 74),   # Third EF-hand loop
    "EF4": (90, 101),  # Fourth EF-hand loop
}

# Simulated Lanmodulin variants (mutations in EF-hand regions)
LANMODULIN_VARIANTS = {
    "WT (Wild-Type)": LANMODULIN_WT,
    "EF1-Mutant (D→N)": LANMODULIN_WT[:15] + "N" + LANMODULIN_WT[16:],   # D16→N
    "EF3-Mutant (D→E)": LANMODULIN_WT[:66] + "E" + LANMODULIN_WT[67:],   # D67→E
    "His-Tag Removed": LANMODULIN_WT.rstrip("H"),                          # Remove His-tag
}


def print_header(title):
    """Pretty print a section header."""
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def print_subheader(title):
    """Pretty print a subsection header."""
    print(f"\n  ── {title} {'─'*(55 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 1: Tokenizer & Decode
# ═══════════════════════════════════════════════════════════════════════════
def demo_tokenize_decode(evaluator):
    """Demonstrate tokenization and decoding of Lanmodulin sequence."""
    print_header("Demo 1: Tokenize & Decode Lanmodulin")

    print(f"\n  Sequence ({len(LANMODULIN_WT)} residues):")
    # Print sequence in blocks of 50
    for i in range(0, len(LANMODULIN_WT), 50):
        chunk = LANMODULIN_WT[i:i+50]
        print(f"    [{i+1:3d}-{min(i+50, len(LANMODULIN_WT)):3d}] {chunk}")

    # Tokenize
    print_subheader("Tokenization")
    tokens = evaluator.tokenize_sequence(LANMODULIN_WT)
    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]

    print(f"  Input IDs shape:      {input_ids.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    print(f"  Input IDs (first 20): {input_ids[:20].tolist()}")
    print(f"  Input IDs (last 20):  {input_ids[-20:].tolist()}")

    # Decode back
    print_subheader("Decode Verification")
    decoded = evaluator.decode_tokens(tokens["input_ids"])
    decoded_str = decoded[0] if isinstance(decoded, list) else decoded

    print(f"  Original:  {LANMODULIN_WT[:60]}...")
    print(f"  Decoded:   {decoded_str[:60]}...")
    print(f"  Length:    Original={len(LANMODULIN_WT)}, Decoded={len(decoded_str)}")
    print(f"  Match:     {'✅ PERFECT MATCH' if decoded_str == LANMODULIN_WT else '❌ MISMATCH'}")

    return tokens


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 2: Detailed Residue Analysis
# ═══════════════════════════════════════════════════════════════════════════
def demo_residue_analysis(evaluator):
    """Analyze per-residue tokenization in detail."""
    print_header("Demo 2: Residue-Level Tokenization Analysis")

    analysis = evaluator.analyze_tokenization(LANMODULIN_WT)

    # Print EF-hand regions with their tokens
    for ef_name, (start, end) in EF_HANDS.items():
        print_subheader(f"{ef_name} Loop (residues {start+1}–{end+1})")
        print(f"  {'Pos':>4s} │ {'AA':>2s} │ {'Token ID':>8s} │ {'Amino Acid Name':<16s} │ {'Standard'}")
        print(f"  {'─'*4:>4s}─┼─{'─'*2:>2s}─┼─{'─'*8:>8s}─┼─{'─'*16:<16s}─┼─{'─'*8}")
        for entry in analysis[start:end+1]:
            std = "✅" if entry['is_standard'] else "⚠️"
            print(f"  {entry['position']+1:4d} │ {entry['residue']:>2s} │ "
                  f"{entry['token_id']:8d} │ {entry['amino_acid_name']:<16s} │ {std}")

    # Key residue statistics
    print_subheader("Sequence Composition")
    from collections import Counter
    aa_counts = Counter(LANMODULIN_WT)
    total = len(LANMODULIN_WT)

    # Sort by frequency
    sorted_aa = sorted(aa_counts.items(), key=lambda x: -x[1])
    print(f"\n  {'AA':>2s} │ {'Count':>5s} │ {'Freq':>5s} │ {'Name':<16s} │ Bar")
    print(f"  {'─'*2:>2s}─┼─{'─'*5:>5s}─┼─{'─'*5:>5s}─┼─{'─'*16:<16s}─┼─{'─'*20}")
    for aa, count in sorted_aa:
        freq = count / total * 100
        bar = "█" * int(freq / 2)
        name = AMINO_ACID_NAMES.get(aa, "Unknown")
        print(f"  {aa:>2s} │ {count:5d} │ {freq:4.1f}% │ {name:<16s} │ {bar}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 3: Vocabulary Inspection
# ═══════════════════════════════════════════════════════════════════════════
def demo_vocab_info(evaluator):
    """Inspect the ESM tokenizer vocabulary."""
    print_header("Demo 3: ESM Tokenizer Vocabulary")

    vocab_info = evaluator.get_vocab_info()

    print(f"\n  Total vocabulary size: {vocab_info['vocab_size']}")

    print_subheader("Special Tokens")
    for name, token in vocab_info['special_tokens'].items():
        print(f"  {name:<15s}: '{token}'")

    print_subheader("Amino Acid → Token ID Mapping")
    aa_tokens = vocab_info['amino_acid_token_ids']
    for aa in sorted(aa_tokens.keys()):
        name = AMINO_ACID_NAMES.get(aa, "?")
        print(f"  {aa} ({name:<14s}) → token_id = {aa_tokens[aa]}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 4: Structure Prediction (Folding)
# ═══════════════════════════════════════════════════════════════════════════
def demo_fold_structure(evaluator):
    """Fold Lanmodulin and extract 3D coordinates."""
    print_header("Demo 4: 3D Structure Prediction (ESMFold)")

    print(f"\n  Folding Lanmodulin ({len(LANMODULIN_WT)} residues)...")
    print(f"  This may take 30–120 seconds depending on hardware...\n")

    t0 = time.time()
    result = evaluator.predict_and_analyze(
        LANMODULIN_WT,
        pdb_filename="lanmodulin_hf_folded.pdb"
    )
    elapsed = time.time() - t0

    print(f"\n  ⏱️  Folding time: {elapsed:.1f} seconds")

    # Print coordinate summary
    ca_coords = result["ca_coords"]
    print_subheader("CA Coordinate Statistics")
    print(f"  Shape: {ca_coords.shape}")
    print(f"  X range: [{ca_coords[:, 0].min():.2f}, {ca_coords[:, 0].max():.2f}] Å")
    print(f"  Y range: [{ca_coords[:, 1].min():.2f}, {ca_coords[:, 1].max():.2f}] Å")
    print(f"  Z range: [{ca_coords[:, 2].min():.2f}, {ca_coords[:, 2].max():.2f}] Å")

    # Print per-residue pLDDT for EF-hand regions
    plddt = result["plddt"]
    print_subheader("pLDDT Confidence by EF-Hand Region")
    for ef_name, (start, end) in EF_HANDS.items():
        region_plddt = plddt[start:end+1]
        mean_p = np.mean(region_plddt)
        min_p = np.min(region_plddt)
        max_p = np.max(region_plddt)
        print(f"  {ef_name}: mean={mean_p:.1f}, min={min_p:.1f}, max={max_p:.1f}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 5: EF-Hand Distance Measurements
# ═══════════════════════════════════════════════════════════════════════════
def demo_ef_hand_distances(evaluator, ca_coords):
    """Measure inter-residue distances in EF-hand binding loops."""
    print_header("Demo 5: EF-Hand Binding Loop Distances")

    print(f"\n  Measuring key distances in the lanthanide binding pockets:\n")
    print(f"  {'Loop Pair':<20s} │ {'Res1':>5s} │ {'Res2':>5s} │ {'Distance (Å)':>12s}")
    print(f"  {'─'*20:─<20s}─┼─{'─'*5:─>5s}─┼─{'─'*5:─>5s}─┼─{'─'*12:─>12s}")

    for ef_name, (start, end) in EF_HANDS.items():
        # Measure distance between first and last residue of each EF-hand
        dist = evaluator.get_residue_distance(ca_coords, start, end)
        print(f"  {ef_name} (start→end)   │ {start+1:5d} │ {end+1:5d} │ {dist:12.3f}")

    # Cross-EF-hand distances
    print_subheader("Cross EF-Hand Distances")
    ef_centers = {}
    for ef_name, (start, end) in EF_HANDS.items():
        center = (start + end) // 2
        ef_centers[ef_name] = center

    pairs = [("EF1", "EF2"), ("EF2", "EF3"), ("EF3", "EF4"),
             ("EF1", "EF3"), ("EF1", "EF4"), ("EF2", "EF4")]

    for ef_a, ef_b in pairs:
        idx_a = ef_centers[ef_a]
        idx_b = ef_centers[ef_b]
        dist = evaluator.get_residue_distance(ca_coords, idx_a, idx_b)
        print(f"  {ef_a}↔{ef_b} (centers)  │ {idx_a+1:5d} │ {idx_b+1:5d} │ {dist:12.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo 6: Multi-Variant Tokenization Comparison
# ═══════════════════════════════════════════════════════════════════════════
def demo_variant_comparison(evaluator):
    """Tokenize and compare multiple Lanmodulin variants."""
    print_header("Demo 6: Lanmodulin Variant Tokenization")

    for variant_name, sequence in LANMODULIN_VARIANTS.items():
        print_subheader(variant_name)
        tokens = evaluator.tokenize_sequence(sequence)
        input_ids = tokens["input_ids"][0].tolist()

        # Decode back
        decoded = evaluator.decode_tokens(tokens["input_ids"])
        decoded_str = decoded[0] if isinstance(decoded, list) else decoded

        print(f"  Length:   {len(sequence)} residues → {len(input_ids)} tokens")
        print(f"  Decode:  {'✅ Match' if decoded_str == sequence else '❌ Mismatch'}")

        # Show mutation site if different from WT
        if sequence != LANMODULIN_WT:
            diffs = []
            for i, (a, b) in enumerate(zip(LANMODULIN_WT, sequence)):
                if a != b:
                    diffs.append(f"pos {i+1}: {a}→{b}")
            if len(sequence) != len(LANMODULIN_WT):
                diffs.append(f"length: {len(LANMODULIN_WT)}→{len(sequence)}")
            print(f"  Changes: {', '.join(diffs)}")

    # Batch tokenization
    print_subheader("Batch Tokenization")
    all_seqs = list(LANMODULIN_VARIANTS.values())
    batch_tokens = evaluator.tokenize_batch(all_seqs)
    print(f"  Batch input_ids shape:      {batch_tokens['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch_tokens['attention_mask'].shape}")

    # Batch decode
    batch_decoded = evaluator.decode_tokens(batch_tokens['input_ids'])
    for i, (name, orig_seq) in enumerate(LANMODULIN_VARIANTS.items()):
        dec = batch_decoded[i]
        # Batch decode may include padding, strip it
        dec_stripped = dec.rstrip()
        match = orig_seq in dec_stripped or dec_stripped.startswith(orig_seq)
        print(f"  [{i}] {name[:25]:<25s}: decode {'✅' if match else '❌'}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  🧬 BioArchitect Engine — HuggingFace ESMFold Lanmodulin Demo   █")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\n  Protein: Lanmodulin (PDB: 6MI5)")
    print(f"  Organism: Methylobacterium extorquens AM1")
    print(f"  Sequence length: {len(LANMODULIN_WT)} residues")
    print(f"  Model: facebook/esmfold_v1 (HuggingFace Transformers)")

    # ── Load Model ─────────────────────────────────────────────────────────
    print_header("Loading HuggingFace ESMFold Model")

    try:
        evaluator = HFEsmFoldEvaluator(
            device='cuda',       # Will fallback to CPU if no GPU
            half_precision=True,  # Use FP16 on GPU to save VRAM
        )
    except Exception as e:
        print(f"\n  ⚠️  Failed to load full model: {e}")
        print(f"  Attempting CPU-only load with FP32...")
        evaluator = HFEsmFoldEvaluator(device='cpu', half_precision=False)

    # ── Run Demos ──────────────────────────────────────────────────────────
    # These demos work on any machine (tokenizer only)
    demo_tokenize_decode(evaluator)
    demo_residue_analysis(evaluator)
    demo_vocab_info(evaluator)
    demo_variant_comparison(evaluator)

    # These demos require model inference (need RAM/VRAM)
    try:
        result = demo_fold_structure(evaluator)
        demo_ef_hand_distances(evaluator, result["ca_coords"])
    except Exception as e:
        print_header("⚠️  Structure Prediction Skipped")
        print(f"\n  Could not run folding: {e}")
        print(f"  This usually means insufficient RAM/VRAM.")
        print(f"  All tokenizer demos completed successfully above.")

    # ── Summary ────────────────────────────────────────────────────────────
    print_header("🏁 Demo Complete!")
    print(f"""
  Files created:
    • step2_hf_folding_evaluator.py  — HuggingFace ESMFold module
    • run_hf_lanmodulin_demo.py      — This demo script
    • lanmodulin_hf_folded.pdb       — Folded structure (if folding ran)

  Next steps:
    • Open .pdb in UCSF ChimeraX for 3D visualization
    • Import HFEsmFoldEvaluator into main_optimizer.py
    • Run the full BioArchitect optimization pipeline
""")


if __name__ == "__main__":
    main()
