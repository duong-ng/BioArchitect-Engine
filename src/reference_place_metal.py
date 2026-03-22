"""
==============================================================================
 Reference-Guided Metal Placement (PDB 6MI5) — Per-EF-Hand Local Alignment
==============================================================================
 For each EF-hand, locally superimposes the loop backbone onto the
 corresponding region in PDB 6MI5, then transfers the experimental
 metal position.

 PDB 6MI5: Lanmodulin NMR structure (model 1) with 3 Y³⁺ ions
   Chain X, residues 22–139
   Metals (YT3): res 200, 201, 202

 Wild-type lanmodulin has 4 EF-hands but 6MI5 only resolves 3 metal sites.
 We map them based on proximity to EF-hand loops.

 Usage:
   python reference_place_metal.py --input result/best_Pr_lanmodulin_variant.pdb
==============================================================================
"""

import os
import sys
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Our variant EF-hands
EF_HANDS = {
    "EF1": (12, 23),
    "EF2": (35, 46),
    "EF3": (63, 74),
    "EF4": (90, 101),
}

COORD_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "ASN": ["OD1"],
    "GLN": ["OE1"],
    "SER": ["OG"],
    "THR": ["OG1"],
}


def parse_pdb_atoms(pdb_path, model=1):
    """Parse all atoms from model 1 of a PDB file."""
    atoms = {}  # resseq -> {atom_name: (x,y,z)}
    ca_list = []
    in_model = (model is None)
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                m = int(line[6:].strip())
                in_model = (m == model) if model else True
            if line.startswith("ENDMDL"):
                if in_model and model:
                    break
            if not in_model:
                continue
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                resseq = int(line[22:26])
                resname = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if resseq not in atoms:
                    atoms[resseq] = {"resname": resname}
                atoms[resseq][atom_name] = np.array([x, y, z])

                if atom_name == "CA":
                    ca_list.append({
                        "resseq": resseq, "resname": resname,
                        "pos": np.array([x, y, z])
                    })
    return atoms, ca_list


def parse_metals(pdb_path, model=1):
    """Parse HETATM metal positions."""
    metals = []
    in_model = (model is None)
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                m = int(line[6:].strip())
                in_model = (m == model) if model else True
            if line.startswith("ENDMDL"):
                if in_model and model:
                    break
            if not in_model:
                continue
            if line.startswith("HETATM"):
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                metals.append({
                    "resseq": resseq,
                    "pos": np.array([x, y, z])
                })
    return metals


def kabsch(P, Q):
    """Kabsch: returns R, t such that P @ R + t ≈ Q."""
    cP = np.mean(P, axis=0)
    cQ = np.mean(Q, axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_m = np.eye(3)
    sign_m[2, 2] = np.sign(d)
    R = Vt.T @ sign_m @ U.T
    t = cQ - cP @ R
    return R, t


def find_offset(ref_ca, target_ca):
    """Find residue number offset between ref and target."""
    ref_names = {c["resseq"]: c["resname"] for c in ref_ca}
    tgt_names = {c["resseq"]: c["resname"] for c in target_ca}
    best_off, best_n = 0, 0
    for off in range(-50, 150):
        n = sum(1 for r in tgt_names
                if r+off in ref_names and tgt_names[r] == ref_names[r+off])
        if n > best_n:
            best_n = n
            best_off = off
    return best_off, best_n


def local_align_ef(ref_ca_dict, target_ca_dict, ef_start, ef_end, offset,
                   extend=5):
    """
    Locally align the EF-hand region ± extend residues.
    Returns R, t, rmsd for this local alignment.
    """
    P_list, Q_list = [], []
    for tgt_res in range(ef_start - extend, ef_end + extend + 1):
        ref_res = tgt_res + offset
        if tgt_res in target_ca_dict and ref_res in ref_ca_dict:
            P_list.append(target_ca_dict[tgt_res])
            Q_list.append(ref_ca_dict[ref_res])

    if len(P_list) < 5:
        return None, None, None

    P = np.array(P_list)
    Q = np.array(Q_list)
    R, t = kabsch(P, Q)

    P_aligned = P @ R + t
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))))
    return R, t, rmsd


def assign_metals_to_ef(metals, ref_ca_dict, ef_hands_ref):
    """Assign each metal to the nearest EF-hand based on CA center."""
    assignments = {}
    for m in metals:
        best_ef = None
        best_d = 1e9
        for ef_name, (s, e) in ef_hands_ref.items():
            positions = [ref_ca_dict[r] for r in range(s, e+1)
                         if r in ref_ca_dict]
            if positions:
                center = np.mean(positions, axis=0)
                d = np.linalg.norm(m["pos"] - center)
                if d < best_d:
                    best_d = d
                    best_ef = ef_name
        if best_ef:
            assignments[best_ef] = m["pos"]
    return assignments


def run(input_pdb, ref_pdb, output_pdb, ion="PR"):
    print("=" * 65)
    print("  Reference-Guided Metal Placement (Per-EF-Hand Local Alignment)")
    print("=" * 65)

    # ── 1. Parse structures ──────────────────────────────────────────────
    print("\n[1/4] Parsing structures...")
    ref_atoms, ref_ca = parse_pdb_atoms(ref_pdb, model=1)
    target_atoms, target_ca = parse_pdb_atoms(input_pdb, model=None)
    metals = parse_metals(ref_pdb, model=1)

    ref_ca_dict = {c["resseq"]: c["pos"] for c in ref_ca}
    target_ca_dict = {c["resseq"]: c["pos"] for c in target_ca}

    print(f"  Reference: {len(ref_ca)} CA, {len(metals)} metals")
    print(f"  Target:    {len(target_ca)} CA")

    # ── 2. Alignment offset ──────────────────────────────────────────────
    print("\n[2/4] Finding residue alignment...")
    offset, n_match = find_offset(ref_ca, target_ca)
    print(f"  Offset: target + {offset} = reference ({n_match} matches)")

    # Map reference EF-hands (with offset applied)
    ef_hands_ref = {
        name: (s + offset, e + offset)
        for name, (s, e) in EF_HANDS.items()
    }

    # Assign metals to EF-hands in reference frame
    metal_assignments = assign_metals_to_ef(metals, ref_ca_dict, ef_hands_ref)
    print(f"  Metals assigned to EF-hands: {list(metal_assignments.keys())}")

    # ── 3. Per-EF-hand local alignment + metal transfer ──────────────────
    print("\n[3/4] Per-EF-hand local alignment...")

    transferred_metals = {}  # ef_name -> position in target frame

    for ef_name, (tgt_start, tgt_end) in EF_HANDS.items():
        R, t, rmsd = local_align_ef(
            ref_ca_dict, target_ca_dict,
            tgt_start, tgt_end, offset, extend=5
        )

        if R is None:
            print(f"  {ef_name}: SKIP (insufficient paired residues)")
            continue

        print(f"  {ef_name} (res {tgt_start}-{tgt_end}): "
              f"local RMSD = {rmsd:.3f} A")

        if ef_name in metal_assignments:
            ref_metal = metal_assignments[ef_name]
            # Transform: from ref frame, we need to go back to target frame
            # target @ R + t = ref  =>  target = (ref - t) @ R.T
            target_metal = (ref_metal - t) @ R.T
            transferred_metals[ef_name] = target_metal
            print(f"    Metal transferred: ({target_metal[0]:.2f}, "
                  f"{target_metal[1]:.2f}, {target_metal[2]:.2f})")
        else:
            print(f"    No metal in reference for this EF-hand")
            # Fall back to centroid for EF-hands without reference metal
            oxygens = []
            for resseq in range(tgt_start, tgt_end + 1):
                if resseq in target_atoms:
                    resname = target_atoms[resseq]["resname"]
                    if resname in COORD_ATOMS:
                        for aname in COORD_ATOMS[resname]:
                            if aname in target_atoms[resseq]:
                                oxygens.append(target_atoms[resseq][aname])
            if oxygens:
                centroid = np.mean(oxygens, axis=0)
                transferred_metals[ef_name] = centroid
                print(f"    Fallback centroid: ({centroid[0]:.2f}, "
                      f"{centroid[1]:.2f}, {centroid[2]:.2f})")

    # ── 4. Write output + distances ──────────────────────────────────────
    print(f"\n[4/4] Writing output...")

    atom_lines = []
    with open(input_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_lines.append(line)

    max_serial = max(int(l[6:11]) for l in atom_lines) if atom_lines else 0

    with open(output_pdb, "w") as f:
        f.write("REMARK   1 Reference-guided placement from PDB 6MI5\n")
        f.write("REMARK   2 Per-EF-hand local Kabsch alignment\n")
        for line in atom_lines:
            f.write(line)
        f.write("TER\n")

        metal_idx = 0
        for ef_name in ["EF1", "EF2", "EF3", "EF4"]:
            if ef_name not in transferred_metals:
                continue
            metal_idx += 1
            max_serial += 1
            pos = transferred_metals[ef_name]
            f.write(
                f"HETATM{max_serial:5d} {ion:<4s} {ion:>3s} B"
                f"{metal_idx:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {ion:>2s}  \n"
            )
        f.write("END\n")

    print(f"  Output: {output_pdb}")

    # ── Report distances ─────────────────────────────────────────────────
    print(f"\n  {ion}-O Coordination Distances (reference-guided)")
    print("  " + "-" * 60)

    for ef_name in ["EF1", "EF2", "EF3", "EF4"]:
        if ef_name not in transferred_metals:
            continue
        pos = transferred_metals[ef_name]
        start, end = EF_HANDS[ef_name]

        source = "crystal" if ef_name in metal_assignments else "centroid"
        print(f"\n  [{ef_name}] {ion} at ({pos[0]:.2f}, {pos[1]:.2f}, "
              f"{pos[2]:.2f}) [{source}]")

        dists = []
        for resseq in range(start, end + 1):
            if resseq in target_atoms:
                resname = target_atoms[resseq]["resname"]
                if resname in COORD_ATOMS:
                    for aname in COORD_ATOMS[resname]:
                        if aname in target_atoms[resseq]:
                            d = float(np.linalg.norm(
                                pos - target_atoms[resseq][aname]
                            ))
                            dists.append((resname, resseq, aname, d))

        dists.sort(key=lambda x: x[3])
        n_coord = 0
        for resname, resseq, aname, d in dists:
            tag = ""
            if d < 2.7:
                tag = " ** INNER SPHERE (Ln-O) **"
                n_coord += 1
            elif d < 3.5:
                tag = " * SECOND SHELL *"
                n_coord += 1
            print(f"    {resname:>3s} {resseq:>3d} {aname:<4s}  "
                  f"{d:.3f} A{tag}")

        if dists:
            avg = np.mean([d for _, _, _, d in dists])
            mn = min(d for _, _, _, d in dists)
            # Count within proper Ln coordination range (2.3-2.7 A)
            n_inner = sum(1 for _, _, _, d in dists if d < 2.7)
            print(f"    --- Avg: {avg:.3f} A | Min: {mn:.3f} A | "
                  f"Inner sphere (<2.7A): {n_inner}/{len(dists)} | "
                  f"<3.5A: {n_coord}/{len(dists)}")

    print(f"\n  Output: {output_pdb}")
    print("=" * 65)
    abs_out = os.path.abspath(output_pdb)
    print(f"\n  -- ChimeraX --")
    print(f'  open "{abs_out}"')
    print(f"  select /B; style sel sphere; color sel gold; size sel atomRadius 1.5")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reference-guided metal placement from PDB 6MI5")
    parser.add_argument("--input", "-i",
        default="result/best_Pr_lanmodulin_variant.pdb")
    parser.add_argument("--reference", "-r", default="reference/6MI5.pdb")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--ion", default="PR")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_ref_placed{ext}"

    run(args.input, args.reference, args.output, args.ion.upper())
