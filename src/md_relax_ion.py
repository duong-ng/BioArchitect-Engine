"""
==============================================================================
 MD Relaxation of Lanthanide-Lanmodulin Complex
==============================================================================
 Uses OpenMM energy minimization with implicit solvent (OBC2) and harmonic
 restraints to relax EF-hand side chains around lanthanide binding sites.

 Supports any lanthanide ion (ND, PR, LA, DY, etc.) via --ion parameter.

 Usage:
   python md_relax_ion.py --input best_Pr_with_PR.pdb --ion PR
   python md_relax_ion.py --input result/best_Nd_with_ND.pdb --ion ND
==============================================================================
"""

import os
import io
import sys
import math
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── EF-hand definitions ─────────────────────────────────────────────────────
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

# Optimal metal-O coordination distances (Angstroms)
OPTIMAL_DIST = {
    "ND": 2.49, "PR": 2.48, "LA": 2.54, "CE": 2.51,
    "SM": 2.46, "EU": 2.45, "GD": 2.44, "TB": 2.43,
    "DY": 2.42, "HO": 2.41, "ER": 2.39, "TM": 2.38,
    "YB": 2.37, "LU": 2.36,
}
DEFAULT_OPT_DIST = 2.49


def parse_pdb(pdb_path):
    """Parse PDB, separate protein ATOM and metal HETATM records."""
    protein_lines = []
    metal_positions = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resseq = int(line[22:26])
                    metal_positions.append({"resseq": resseq, "x": x, "y": y, "z": z})
                except (ValueError, IndexError):
                    pass
            elif line.startswith("ATOM"):
                protein_lines.append(line)
    return protein_lines, metal_positions


def find_coord_atom_indices(topology, ef_hands):
    """Find indices of coordinating oxygen atoms in the topology."""
    coord_indices = {}
    for ef_name, (start, end) in ef_hands.items():
        indices = []
        for atom in topology.atoms():
            try:
                resnum = int(atom.residue.id)
            except ValueError:
                continue
            if start <= resnum <= end:
                if atom.residue.name in COORD_ATOMS:
                    if atom.name in COORD_ATOMS[atom.residue.name]:
                        indices.append(atom.index)
        coord_indices[ef_name] = indices
    return coord_indices


def run_md_relaxation(input_pdb, output_pdb, ion_symbol="ND",
                      min_rounds=3, min_iters=2000):
    """
    Run OpenMM energy minimization with implicit solvent and restraints.
    Uses multiple rounds with progressively softer backbone restraints.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer

    opt_dist = OPTIMAL_DIST.get(ion_symbol.upper(), DEFAULT_OPT_DIST)
    print("=" * 65)
    print(f"  MD Relaxation of {ion_symbol}-Lanmodulin (Implicit Solvent)")
    print("=" * 65)

    # ── 1. Parse PDB ─────────────────────────────────────────────────────
    print("\n[1/7] Parsing PDB file...")
    protein_lines, metal_positions = parse_pdb(input_pdb)
    print(f"  Protein lines: {len(protein_lines)}, Metal sites: {len(metal_positions)}")
    if not metal_positions:
        print("  ERROR: No HETATM metal records found!")
        return None

    ef_metal_map = {}
    for i, ef_name in enumerate(EF_HANDS):
        if i < len(metal_positions):
            m = metal_positions[i]
            ef_metal_map[ef_name] = np.array([m["x"], m["y"], m["z"]])
            print(f"    {ef_name} -> {ion_symbol} at ({m['x']:.2f}, {m['y']:.2f}, {m['z']:.2f})")

    # ── 2. PDBFixer ──────────────────────────────────────────────────────
    print("\n[2/7] PDBFixer preprocessing...")
    pdb_str = "".join(protein_lines) + "\nTER\nEND\n"
    fixer = PDBFixer(pdbfile=io.StringIO(pdb_str))
    fixer.findMissingResidues()
    # Remove terminal missing residues
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues.keys()):
        ci, ri = key
        residues = list(chains[ci].residues())
        if ri == 0 or ri >= len(residues):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    topology = fixer.topology
    positions = fixer.positions
    n_atoms = sum(1 for _ in topology.atoms())
    print(f"  Fixed: {n_atoms} atoms")

    # ── 3. Setup implicit solvent system ─────────────────────────────────
    print("\n[3/7] Setting up implicit solvent (OBC2) system...")
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')

    # Add hydrogens via modeller for AMBER compatibility
    try:
        modeller = app.Modeller(topology, positions)
        modeller.addHydrogens(forcefield)
        topology = modeller.topology
        positions = modeller.positions
        n_atoms = topology.getNumAtoms()
        print(f"  After H addition: {n_atoms} atoms")
    except Exception as e:
        print(f"  Warning: {e}")

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    print(f"  System created: {system.getNumParticles()} particles")

    # ── 4. Find coordinating oxygens ─────────────────────────────────────
    print("\n[4/7] Identifying coordinating oxygens...")
    coord_indices = find_coord_atom_indices(topology, EF_HANDS)
    for ef_name, indices in coord_indices.items():
        print(f"  {ef_name}: {len(indices)} oxygens")

    # ── 5. Add restraint forces ──────────────────────────────────────────
    print("\n[5/7] Adding restraint forces...")

    # Backbone restraints (strong, keeps fold)
    bb_force = openmm.CustomExternalForce(
        "0.5*k_bb*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    bb_force.addGlobalParameter("k_bb", 5000.0)  # kJ/mol/nm^2
    bb_force.addPerParticleParameter("x0")
    bb_force.addPerParticleParameter("y0")
    bb_force.addPerParticleParameter("z0")

    n_bb = 0
    for atom in topology.atoms():
        if atom.name in ('CA', 'C', 'N'):
            pos = positions[atom.index]
            p = pos.value_in_unit(unit.nanometers)
            bb_force.addParticle(atom.index, [p[0], p[1], p[2]])
            n_bb += 1
    system.addForce(bb_force)
    print(f"  Backbone restraints: {n_bb} atoms (k=5000)")

    # Coordination restraints (pull oxygens toward optimal distance from Nd)
    coord_force = openmm.CustomExternalForce(
        "0.5*k_coord*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    coord_force.addGlobalParameter("k_coord", 500.0)  # kJ/mol/nm^2
    coord_force.addPerParticleParameter("x0")
    coord_force.addPerParticleParameter("y0")
    coord_force.addPerParticleParameter("z0")

    n_coord = 0
    for ef_name, indices in coord_indices.items():
        if ef_name not in ef_metal_map:
            continue
        metal = ef_metal_map[ef_name] * 0.1  # Angstrom -> nm
        opt_nm = opt_dist * 0.1  # nm

        for idx in indices:
            pos = positions[idx].value_in_unit(unit.nanometers)
            # Vector from atom to metal
            dx, dy, dz = metal[0]-pos[0], metal[1]-pos[1], metal[2]-pos[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 1e-6:
                # Target = point at opt_nm from metal along atom-metal line
                frac = (dist - opt_nm) / dist
                tx = pos[0] + dx * frac
                ty = pos[1] + dy * frac
                tz = pos[2] + dz * frac
            else:
                tx, ty, tz = pos[0], pos[1], pos[2]

            coord_force.addParticle(idx, [tx, ty, tz])
            n_coord += 1

    system.addForce(coord_force)
    print(f"  Coordination restraints: {n_coord} atoms (k=500)")

    # ── 6. Multi-round energy minimization ───────────────────────────────
    print(f"\n[6/7] Energy minimization ({min_rounds} rounds, "
          f"{min_iters} iter each)...")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  Initial energy: {e0:.1f} kJ/mol")

    # Round 1: strong backbone (5000) + coordination (500)
    bb_k_schedule = [5000.0, 1000.0, 100.0]
    coord_k_schedule = [500.0, 1000.0, 2000.0]

    for round_i in range(min_rounds):
        k_bb = bb_k_schedule[min(round_i, len(bb_k_schedule)-1)]
        k_co = coord_k_schedule[min(round_i, len(coord_k_schedule)-1)]
        simulation.context.setParameter("k_bb", k_bb)
        simulation.context.setParameter("k_coord", k_co)

        simulation.minimizeEnergy(maxIterations=min_iters)
        state = simulation.context.getState(getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        print(f"  Round {round_i+1}: k_bb={k_bb:.0f}, k_coord={k_co:.0f}"
              f" -> E={e:.1f} kJ/mol")

    state = simulation.context.getState(getEnergy=True)
    ef = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  Final energy: {ef:.1f} kJ/mol (reduction: {e0 - ef:.1f})")

    # ── 7. Extract and write results ─────────────────────────────────────
    print(f"\n[7/7] Writing relaxed structure...")
    final_pos = simulation.context.getState(
        getPositions=True
    ).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    # Compute RMSD (protein only)
    init_pos = np.array([
        positions[i].value_in_unit(unit.angstrom)
        for i in range(len(positions))
    ])
    delta = final_pos - init_pos
    rmsd = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    print(f"  RMSD from initial: {rmsd:.3f} A")

    # Write PDB
    with open(output_pdb, "w") as f:
        f.write(f"REMARK   1 MD-RELAXED (implicit solvent, {min_rounds}x{min_iters})\n")
        f.write(f"REMARK   2 RMSD: {rmsd:.3f} A\n")

        serial = 0
        atom_list = list(topology.atoms())
        for atom in atom_list:
            serial += 1
            pos = final_pos[atom.index]
            chain_id = atom.residue.chain.id or 'A'
            resname = atom.residue.name
            try:
                resseq = int(atom.residue.id)
            except ValueError:
                resseq = 0
            element = atom.element.symbol if atom.element else atom.name[0]
            f.write(
                f"ATOM  {serial:5d} {atom.name:<4s} {resname:>3s} "
                f"{chain_id}{resseq:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}  \n"
            )
        f.write("TER\n")

        # Add metal at relaxed centroids
        ion = ion_symbol.upper()
        print(f"\n  Relaxed {ion}-O distances:")
        print("  " + "-" * 60)
        metal_resseq = 1
        for ef_name, indices in coord_indices.items():
            if not indices:
                continue
            oxy_pos = [final_pos[i] for i in indices]
            centroid = np.mean(oxy_pos, axis=0)
            serial += 1
            f.write(
                f"HETATM{serial:5d} {ion:<4s} {ion:>3s} B{metal_resseq:4d}    "
                f"{centroid[0]:8.3f}{centroid[1]:8.3f}{centroid[2]:8.3f}"
                f"  1.00  0.00          {ion:>2s}  \n"
            )

            # Distances
            print(f"\n  [{ef_name}] {ion} at "
                  f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
            dists = []
            for idx in indices:
                d = float(np.linalg.norm(centroid - final_pos[idx]))
                dists.append(d)
                atom = atom_list[idx]
                tag = " ** COORD **" if d < 3.5 else ""
                print(f"    {atom.residue.name:>3s} {atom.residue.id:>3s} "
                      f"{atom.name:<4s}  {d:.3f} A{tag}")
            if dists:
                avg = np.mean(dists)
                mn = np.min(dists)
                n3 = sum(1 for d in dists if d < 3.5)
                print(f"    --- Avg: {avg:.3f} A | Min: {mn:.3f} A | "
                      f"<3.5A: {n3}/{len(dists)}")

            metal_resseq += 1

        f.write("END\n")

    print(f"\n  Output: {output_pdb}")
    print(f"  RMSD:   {rmsd:.3f} A")
    print("=" * 65)

    abs_out = os.path.abspath(output_pdb)
    print(f"\n  -- ChimeraX --")
    print(f'  open "{abs_out}"')
    print(f"  select /B; style sel sphere; color sel gold; size sel atomRadius 1.5")
    print()
    return {"output_pdb": output_pdb, "rmsd": rmsd}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MD relaxation of lanthanide-lanmodulin complex")
    parser.add_argument("--input", "-i",
        default="result/best_Nd_lanmodulin_with_ND.pdb",
        help="Input PDB with metal HETATM records")
    parser.add_argument("--output", "-o", default=None,
        help="Output relaxed PDB (default: <input>_relaxed.pdb)")
    parser.add_argument("--ion", default="ND",
        help="Ion symbol, e.g. ND, PR, LA (default: ND)")
    parser.add_argument("--rounds", type=int, default=3,
        help="Number of minimization rounds (default: 3)")
    parser.add_argument("--iters", type=int, default=2000,
        help="Iterations per round (default: 2000)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_relaxed{ext}"

    run_md_relaxation(
        input_pdb=args.input,
        output_pdb=args.output,
        ion_symbol=args.ion.upper(),
        min_rounds=args.rounds,
        min_iters=args.iters,
    )
