"""
==============================================================================
 Direct PDB Mutation + Local Relaxation
==============================================================================
 Mutates residues directly in the PDB file (preserving backbone), uses
 PDBFixer to rebuild missing sidechain atoms, then runs local energy
 minimization + adds metal ions.

 This preserves the existing fold (unlike ESMFold re-prediction) while
 only changing the specified sidechain atoms.

 Usage:
   python mutate_pdb.py --input result/best_Pr_lanmodulin_variant.pdb
   python mutate_pdb.py --mutations "K35N,P38D,Q40D,R96D,A99N" --ion PR
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

# Backbone atoms to keep during mutation (shared across all residues)
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'CB', 'H', 'HA'}

# Default mutations for EF2/EF4 optimization
DEFAULT_MUTATIONS = "K35N,P38D,Q40D,R96D,A99N"


def parse_mutation_string(mut_str):
    """Parse 'K35N,P38D,Q40D' -> {35: ('K','N'), 38: ('P','D'), ...}"""
    mutations = {}
    for m in mut_str.split(","):
        m = m.strip()
        if len(m) < 3:
            continue
        wt = m[0]
        mut = m[-1]
        pos = int(m[1:-1])
        mutations[pos] = (wt, mut)
    return mutations


def mutate_pdb_inplace(pdb_path, mutations, output_path):
    """
    Mutate residues in PDB by:
    1. Changing residue name in ATOM records
    2. Removing sidechain atoms beyond CB (they'll be rebuilt by PDBFixer)
    3. Keeping backbone atoms intact
    """
    lines_out = []
    mutated = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    resseq = int(line[22:26])
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                except (ValueError, IndexError):
                    lines_out.append(line)
                    continue

                if resseq in mutations:
                    wt_1, mut_1 = mutations[resseq]
                    new_resname = AA_1TO3[mut_1]

                    # Skip sidechain atoms (they'll be rebuilt by PDBFixer)
                    # Keep backbone: N, CA, C, O, CB, H, HA
                    if atom_name not in BACKBONE_ATOMS:
                        continue  # Drop this atom

                    # Replace residue name
                    line = line[:17] + f"{new_resname:>3s}" + line[20:]
                    mutated.add(resseq)

            lines_out.append(line)

    with open(output_path, "w") as f:
        f.writelines(lines_out)

    return mutated


def rebuild_sidechains(pdb_path, output_path, pH=7.0):
    """Use PDBFixer to rebuild missing sidechain atoms."""
    from pdbfixer import PDBFixer
    import openmm.app as app

    print("  Rebuilding sidechains with PDBFixer...")
    fixer = PDBFixer(filename=pdb_path)

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

    n_missing = len(fixer.missingAtoms)
    n_terminals = len(fixer.missingTerminals)
    print(f"  Missing atoms to add: {n_missing} + {n_terminals} terminal")

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)

    n_atoms = sum(1 for _ in fixer.topology.atoms())
    print(f"  Rebuilt structure: {n_atoms} atoms")

    # Write
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return fixer.topology, fixer.positions


def local_minimize(pdb_path, output_path, mutation_positions,
                   min_rounds=3, min_iters=2000):
    """
    Energy minimize with strong backbone restraints.
    Only sidechain atoms at mutation sites are free to move.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    print("  Setting up local energy minimization...")
    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Strong backbone restraints — hold everything except mutated sidechains
    bb_force = openmm.CustomExternalForce(
        "0.5*k_bb*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    bb_force.addGlobalParameter("k_bb", 5000.0)
    bb_force.addPerParticleParameter("x0")
    bb_force.addPerParticleParameter("y0")
    bb_force.addPerParticleParameter("z0")

    n_restrained = 0
    for atom in pdb.topology.atoms():
        try:
            resseq = int(atom.residue.id)
        except ValueError:
            continue

        # Restrain all atoms EXCEPT sidechain atoms at mutation sites
        is_mutation_site = resseq in mutation_positions
        is_sidechain = atom.name not in BACKBONE_ATOMS

        if not (is_mutation_site and is_sidechain):
            pos = pdb.positions[atom.index]
            p = pos.value_in_unit(unit.nanometers)
            bb_force.addParticle(atom.index, [p[0], p[1], p[2]])
            n_restrained += 1

    system.addForce(bb_force)
    n_free = system.getNumParticles() - n_restrained
    print(f"  Restrained: {n_restrained} atoms, Free: {n_free} atoms")

    # Minimize
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  Initial energy: {e0:.1f} kJ/mol")

    bb_k_schedule = [5000.0, 2000.0, 500.0]
    for i in range(min_rounds):
        k = bb_k_schedule[min(i, len(bb_k_schedule)-1)]
        simulation.context.setParameter("k_bb", k)
        simulation.minimizeEnergy(maxIterations=min_iters)
        state = simulation.context.getState(getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        print(f"  Round {i+1}: k_bb={k:.0f} -> E={e:.1f} kJ/mol")

    # Get final positions
    final_pos = simulation.context.getState(
        getPositions=True
    ).getPositions()

    # Compute RMSD
    init_pos = np.array([
        pdb.positions[i].value_in_unit(unit.angstrom)
        for i in range(len(pdb.positions))
    ])
    final_np = np.array([
        final_pos[i].value_in_unit(unit.angstrom)
        for i in range(len(final_pos))
    ])
    rmsd = float(np.sqrt(np.mean(np.sum((final_np - init_pos)**2, axis=1))))
    print(f"  RMSD: {rmsd:.3f} A")

    # Write output
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(pdb.topology, final_pos, f, keepIds=True)

    return rmsd


def run_pipeline(input_pdb, output_dir, mutations_str, ion="PR"):
    """Full pipeline: mutate -> rebuild -> minimize -> add ion -> relax."""
    print("=" * 65)
    print("  Direct PDB Mutation + Local Relaxation")
    print("=" * 65)

    mutations = parse_mutation_string(mutations_str)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Mutate PDB ───────────────────────────────────────────────────
    print(f"\n[1/5] Applying {len(mutations)} mutations in-place...")
    for pos, (wt, mut) in sorted(mutations.items()):
        print(f"  {AA_1TO3.get(wt, '???')} {pos} -> {AA_1TO3.get(mut, '???')} ({wt}{pos}{mut})")

    mutated_pdb = os.path.join(output_dir, "Pr_mutated_raw.pdb")
    mutated_positions = mutate_pdb_inplace(input_pdb, mutations, mutated_pdb)
    print(f"  Mutated {len(mutated_positions)} residues")

    # ── 2. Rebuild sidechains ────────────────────────────────────────────
    print(f"\n[2/5] Rebuilding sidechains...")
    rebuilt_pdb = os.path.join(output_dir, "Pr_mutated_rebuilt.pdb")
    rebuild_sidechains(mutated_pdb, rebuilt_pdb)

    # ── 3. Local energy minimization ─────────────────────────────────────
    print(f"\n[3/5] Local energy minimization...")
    minimized_pdb = os.path.join(output_dir, "Pr_mutated_minimized.pdb")
    rmsd = local_minimize(
        rebuilt_pdb, minimized_pdb,
        mutation_positions=set(mutations.keys()),
    )

    # ── 4. Add metal ions ────────────────────────────────────────────────
    print(f"\n[4/5] Adding {ion} ions at EF-hand centroids...")
    from add_ion_to_pdb import add_metal_ions
    ion_pdb = os.path.join(output_dir, f"Pr_mutated_with_{ion}.pdb")
    add_metal_ions(minimized_pdb, ion_pdb, ion_symbol=ion, element=ion)

    # ── 5. MD Relaxation ─────────────────────────────────────────────────
    print(f"\n[5/5] MD relaxation (implicit solvent)...")
    from md_relax_ion import run_md_relaxation
    relaxed_pdb = os.path.join(output_dir, f"Pr_mutated_relaxed.pdb")
    result = run_md_relaxation(
        input_pdb=ion_pdb,
        output_pdb=relaxed_pdb,
        ion_symbol=ion,
        min_rounds=3,
        min_iters=2000,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    for pos, (wt, mut) in sorted(mutations.items()):
        print(f"  {wt}{pos}{mut}")
    print(f"\n  Local min RMSD:    {rmsd:.3f} A")
    if result:
        print(f"  MD relax RMSD:     {result['rmsd']:.3f} A")
    print(f"\n  Output: {relaxed_pdb}")
    print("=" * 65)

    abs_out = os.path.abspath(relaxed_pdb)
    print(f"\n  -- ChimeraX --")
    print(f'  open "{abs_out}"')
    print(f"  select /B; style sel sphere; color sel gold; size sel atomRadius 1.5")
    print(f"  distance #1/B:2@{ion} #1/A:38@OD1")
    print(f"  distance #1/B:4@{ion} #1/A:96@OD1")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct PDB mutation + local relaxation")
    parser.add_argument("--input", "-i",
        default="result/best_Pr_lanmodulin_variant.pdb",
        help="Input PDB file")
    parser.add_argument("--output-dir", "-d", default="result",
        help="Output directory")
    parser.add_argument("--mutations", "-m", default=DEFAULT_MUTATIONS,
        help="Comma-separated mutations (e.g. K35N,P38D)")
    parser.add_argument("--ion", default="PR",
        help="Ion symbol (default: PR)")
    args = parser.parse_args()

    run_pipeline(
        input_pdb=args.input,
        output_dir=args.output_dir,
        mutations_str=args.mutations,
        ion=args.ion.upper(),
    )
