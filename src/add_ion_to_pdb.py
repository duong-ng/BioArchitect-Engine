"""
Add Pr3+ ions to lanmodulin PDB at EF-hand binding pocket centroids.

Reads the protein-only PDB, identifies coordinating oxygens (OD1/OD2 from
Asp, OE1/OE2 from Glu, OD1 from Asn) within each EF-hand loop, computes
the centroid of those oxygens, and writes a HETATM record for Pr3+ at
each centroid.  The output PDB can be opened directly in ChimeraX.

Usage:
    python add_pr_to_pdb.py
    python add_pr_to_pdb.py --input my_structure.pdb --ion La
"""

import argparse
import math
import os

# ── EF-hand loop definitions (1-indexed residue numbers) ─────────────────
EF_HANDS = {
    "EF1": (12, 23),
    "EF2": (35, 46),
    "EF3": (63, 74),
    "EF4": (90, 101),
}

# Atoms that coordinate the lanthanide ion
COORDINATING_ATOMS = {
    "ASP": ["OD1", "OD2"],      # bidentate carboxylate
    "GLU": ["OE1", "OE2"],      # bidentate carboxylate
    "ASN": ["OD1"],             # amide oxygen
    "GLN": ["OE1"],             # amide oxygen
    "SER": ["OG"],              # hydroxyl
    "THR": ["OG1"],             # hydroxyl
}


def parse_pdb_atoms(pdb_path):
    """Parse ATOM/HETATM records from a PDB file."""
    atoms = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom = {
                    "record":  line[:6].strip(),
                    "serial":  int(line[6:11]),
                    "name":    line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain":   line[21],
                    "resseq":  int(line[22:26]),
                    "x":       float(line[30:38]),
                    "y":       float(line[38:46]),
                    "z":       float(line[46:54]),
                    "line":    line,
                }
                atoms.append(atom)
    return atoms


def find_coordinating_oxygens(atoms, ef_start, ef_end):
    """Find coordinating oxygen atoms within an EF-hand loop."""
    oxygens = []
    for atom in atoms:
        if ef_start <= atom["resseq"] <= ef_end:
            if atom["resname"] in COORDINATING_ATOMS:
                if atom["name"] in COORDINATING_ATOMS[atom["resname"]]:
                    oxygens.append(atom)
    return oxygens


def compute_centroid(atoms_list):
    """Compute the centroid (x, y, z) of a list of atoms."""
    n = len(atoms_list)
    if n == 0:
        return None
    cx = sum(a["x"] for a in atoms_list) / n
    cy = sum(a["y"] for a in atoms_list) / n
    cz = sum(a["z"] for a in atoms_list) / n
    return (cx, cy, cz)


def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def make_hetatm_line(serial, atom_name, resname, chain, resseq, x, y, z, element):
    """Format a HETATM line in PDB format."""
    return (
        f"HETATM{serial:5d} {atom_name:<4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  \n"
    )


def add_metal_ions(input_pdb, output_pdb, ion_symbol="ND", element="ND",
                   ef_hands=None):
    """
    Add metal ions at EF-hand binding pocket centroids.

    Returns a report dict with per-site distances.
    """
    if ef_hands is None:
        ef_hands = EF_HANDS

    atoms = parse_pdb_atoms(input_pdb)
    report = {}

    # Read original file lines
    with open(input_pdb, "r") as f:
        lines = f.readlines()

    # Remove trailing END/TER so we can append HETATM before them
    protein_lines = []
    footer_lines = []
    for line in lines:
        if line.startswith("END") or (line.startswith("TER") and not protein_lines):
            footer_lines.append(line)
        elif line.startswith("TER") and protein_lines:
            protein_lines.append(line)
        else:
            protein_lines.append(line)

    # Separate last TER + END
    final_footer = []
    rebuilt = []
    for line in reversed(protein_lines + footer_lines):
        stripped = line.strip()
        if stripped in ("END", "") or stripped.startswith("TER"):
            final_footer.insert(0, line)
        else:
            rebuilt.insert(0, line)
            break
    # Grab everything up to and including the last real atom line
    last_real_idx = 0
    all_lines = protein_lines + footer_lines
    for i, line in enumerate(all_lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            last_real_idx = i
    before_end = all_lines[:last_real_idx + 1]

    # Find max serial number
    max_serial = max(a["serial"] for a in atoms) if atoms else 0

    hetatm_lines = []
    metal_chain = "B"
    metal_resseq = 1

    print("=" * 65)
    print(f"  Adding {ion_symbol}3+ ions to EF-hand binding pockets")
    print("=" * 65)

    for loop_name, (start, end) in ef_hands.items():
        oxygens = find_coordinating_oxygens(atoms, start, end)

        if not oxygens:
            print(f"\n  [{loop_name}] (res {start}-{end}): No coordinating oxygens found")
            report[loop_name] = {"status": "no_oxygens"}
            continue

        centroid = compute_centroid(oxygens)
        max_serial += 1

        hetatm = make_hetatm_line(
            serial=max_serial,
            atom_name=ion_symbol,
            resname=ion_symbol,
            chain=metal_chain,
            resseq=metal_resseq,
            x=centroid[0], y=centroid[1], z=centroid[2],
            element=element,
        )
        hetatm_lines.append(hetatm)

        # Compute distances to each coordinating oxygen
        distances = []
        print(f"\n  [{loop_name}] (res {start}-{end}):")
        print(f"    Metal position: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        print(f"    Coordinating oxygens ({len(oxygens)}):")

        for oxy in oxygens:
            oxy_pos = (oxy["x"], oxy["y"], oxy["z"])
            d = distance(centroid, oxy_pos)
            distances.append({
                "residue": f"{oxy['resname']} {oxy['resseq']}",
                "atom": oxy["name"],
                "distance": d,
            })
            print(f"      {oxy['resname']:>3s} {oxy['resseq']:>3d} {oxy['name']:<4s}  ->  {d:.3f} A")

        avg_dist = sum(dd["distance"] for dd in distances) / len(distances)
        print(f"    Average Pr-O distance: {avg_dist:.3f} A")

        report[loop_name] = {
            "centroid": centroid,
            "num_oxygens": len(oxygens),
            "distances": distances,
            "avg_distance": avg_dist,
        }
        metal_resseq += 1

    # Write output PDB
    with open(output_pdb, "w") as f:
        for line in before_end:
            f.write(line)
        # TER after protein
        f.write("TER\n")
        # HETATM lines for metal ions
        for hl in hetatm_lines:
            f.write(hl)
        f.write("END\n")

    print(f"\n  Output PDB: {output_pdb}")
    print(f"  Metal ions added: {len(hetatm_lines)}")
    print("=" * 65)

    # Print ChimeraX commands
    print("\n  -- ChimeraX Commands --------------------------------------")
    print(f"  open {os.path.abspath(output_pdb)}")
    print(f"  select /{metal_chain}")
    print(f"  style sel sphere")
    print(f"  color sel gold")
    print(f"  size sel atomRadius 1.2")
    print(f"  select :ASP,GLU,ASN & :{','.join(str(r) for h in ef_hands.values() for r in range(h[0], h[1]+1))}")
    print(f"  show sel atoms")
    print(f"  # Measure distances: Tools -> Structure Analysis -> Distances")
    print(f"  # Or use: distance /{metal_chain}:1 :13@OD1")
    print("  --------------------------------------------------------\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add lanthanide ion to PDB at EF-hand binding centroids"
    )
    parser.add_argument(
        "--input", "-i",
        default="best_Pr_lanmodulin.pdb",
        help="Input PDB file (default: best_Pr_lanmodulin.pdb)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output PDB file (default: <input>_with_<ion>.pdb)",
    )
    parser.add_argument(
        "--ion",
        default="PR",
        help="Ion symbol, e.g. ND, LA, DY (default: ND)",
    )
    args = parser.parse_args()

    ion = args.ion.upper()
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_with_{ion}{ext}"
    else:
        output = args.output

    add_metal_ions(args.input, output, ion_symbol=ion, element=ion)
