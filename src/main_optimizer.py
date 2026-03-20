"""
==============================================================================
 BioArchitect Engine — Main Optimizer Pipeline
==============================================================================
 LanRecov: Multi-ion optimization pipeline for selective Lanthanide recovery.
 
 Pipeline Steps:
   1. ProteinMPNN — Ion-biased mutation generation in EF-hand regions
   2. AlphaFold2 — High-accuracy 3D structure prediction (PAE + pLDDT)
   3. Genetic Algorithm — LanRecov fitness (binding + selectivity + stability + confidence)
   4. Molecular Dynamics — Structural validation (OpenMM/ASE/mock)
   5. PDB Export — ChimeraX visualization
   
 Usage:
   python main_optimizer.py                    # Default: La³⁺ optimization
   python main_optimizer.py --ion Nd           # Optimize for Nd³⁺
   python main_optimizer.py --multi La,Nd,Dy   # Multi-ion screening
==============================================================================
"""

import sys
import time
import argparse

# Fix Windows console encoding for Unicode output (emojis, special chars)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from step1_mutation_generator import ProteinMPNNGenerator
from step4_md_validation import MDValidator
from lanthanide_params import (
    LANTHANIDE_DB,
    get_ion_params,
    get_distance_range,
)

# ═══════════════════════════════════════════════════════════════════════════
#  AlphaFold2 Backend (ColabFold / OpenFold)
# ═══════════════════════════════════════════════════════════════════════════
from step2_af2_folding_evaluator import AlphaFold2Evaluator
from step3_genetic_optimizer import BioArchitectGA
print("[Config] Using AlphaFold2 backend (ColabFold/OpenFold)")

# 6MI5 FASTA SEQUENCE
base_fasta = "PTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADLKKLDPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIRHHHHHH"

# Define EF hand loops
ef_hands = {
    "EF1": (12, 23),
    "EF2": (35, 46),
    "EF3": (63, 74),
    "EF4": (90, 101)
}

# Path to the base Lanmodulin structure (used by ProteinMPNN Native)
base_pdb = r"d:\BioArchitect Engine\data\6MI5.pdb"


def run_bioarchitect_pipeline(target_ion="Nd", pH=7.0, generations=20,
                               population_size=40, mutation_rate=0.15,
                               pH_resistant=False):
    """
    Run the full BioArchitect Engine optimization pipeline for a specific ion.
    
    Args:
        target_ion (str): Target Lanthanide (e.g. "La", "Nd", "Dy").
        pH (float): Solution pH for MD validation.
        generations (int): Number of GA generations.
        population_size (int): Population size per generation.
        pH_resistant (bool): If True, bias mutations toward pH-stable residues.
    
    Returns:
        dict: Pipeline results including best candidate and validation.
    """
    ion_params = get_ion_params(target_ion)
    target_distance = get_distance_range(target_ion)
    
    print("=" * 60)
    print("🧬 BIO-ARCHITECT ENGINE — LanRecov PIPELINE")
    print("=" * 60)
    print(f"  Target Ion:      {ion_params['symbol']} ({ion_params['name']})")
    print(f"  Ionic Radius:    {ion_params['ionic_radius']} Å")
    print(f"  Coord Distance:  {target_distance[0]:.2f} – {target_distance[1]:.2f} Å")
    print(f"  Group:           {ion_params['group']} REE")
    print(f"  pH:              {pH}")
    print(f"  Generations:     {generations}")
    print(f"  Population:      {population_size}")
    print("=" * 60)
    
    # Init modules with ion-specific parameters
    generator = ProteinMPNNGenerator(
        base_fasta, ef_hands,
        target_ion=target_ion,
        pH_resistant=pH_resistant,
    )
    
    # Initialize AlphaFold2 evaluator with LanRecov config
    evaluator = AlphaFold2Evaluator(
        device='cpu',
        num_recycles=3,
        ef_hands=ef_hands,
        fix_pdb=True,
        fix_pH=pH,
    )
    
    ga_optimizer = BioArchitectGA(
        target_ion=target_ion,
        ef_hands=ef_hands,
    )
    validator = MDValidator(target_ion=target_ion)

    # Pipeline Step 1: Initial Population via ProteinMPNN
    print(f"\n[Step 1] Generating Ion-Optimized Mutant Population...")
    population = generator.generate_population(population_size=population_size, pdb_path=base_pdb)
    
    # Define binding site residues from EF4 loop
    target_res_1 = ef_hands["EF4"][0] + 2  # e.g., an Aspartate
    target_res_2 = ef_hands["EF4"][1] - 1  # e.g., a Glutamate
    
    # Collect all EF-hand binding indices
    binding_indices = []
    for loop_name, (start, end) in ef_hands.items():
        binding_indices.extend(range(start, end + 1))

    best_candidate = None
    best_fitness = -1
    best_breakdown = None

    for gen in range(generations):
        print(f"\n{'─'*50}")
        print(f"  Generation {gen+1}/{generations}")
        print(f"{'─'*50}")
        scored_population = []
        
        for idx, seq in enumerate(population):
            # Pipeline Step 2: AlphaFold2 structure prediction
            coords = evaluator.predict_structure(seq)
            
            # Get confidence scores if available (AF2 provides PAE)
            confidence_scores = None
            try:
                confidence_scores = evaluator.get_confidence_scores()
            except (RuntimeError, AttributeError):
                pass
            
            # Pipeline Step 3: LanRecov Fitness Evaluation
            fitness, dist, breakdown = ga_optimizer.calculate_fitness(
                coords, target_res_1, target_res_2,
                sequence=seq,
                binding_indices=binding_indices,
                confidence_scores=confidence_scores,
            )
            scored_population.append((fitness, dist, breakdown, seq, coords))
            
        # Sort by fitness descending
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        top_fitness, top_dist, top_breakdown, top_seq, top_coords = scored_population[0]
        print(f"\n  🥇 Best in Gen {gen+1}:")
        print(f"     Distance:    {top_dist:.3f} Å")
        print(f"     Fitness:     {top_fitness:.4f}")
        print(f"     Scores:      Geo={top_breakdown['geometric']:.3f} "
              f"Bind={top_breakdown['binding_energy']:.3f} "
              f"Stab={top_breakdown['stability']:.3f} "
              f"Sel={top_breakdown['selectivity']:.3f} "
              f"Coord={top_breakdown['coordination']:.3f} "
              f"Conf={top_breakdown.get('confidence', 0.5):.3f}")
        
        if top_fitness > best_fitness:
            best_fitness = top_fitness
            best_candidate = (top_seq, top_coords, top_dist)
            best_breakdown = top_breakdown

        # Evolve population for next generation
        population = ga_optimizer.evolve_population(
            scored_population, population_size,
            elitism=3, mutation_rate=mutation_rate,
            generation=gen, total_generations=generations,
        )

    print("\n" + "=" * 60)
    print("🔥 LanRecov OPTIMIZATION COMPLETE 🔥")
    print(f"  Target:   {ion_params['symbol']}")
    print(f"  Distance: {best_candidate[2]:.3f} Å "
          f"(target: {target_distance[0]:.2f}–{target_distance[1]:.2f})")
    print(f"  Fitness:  {best_fitness:.4f}")
    print("=" * 60)
    
    # Pipeline Step 4: Molecular Dynamics Validation
    print("\n[Step 4] Running MD Validation on Best Candidate...")
    md_result = validator.run_simulation(
        best_candidate[0], best_candidate[1],
        metal=target_ion, steps=10000,
        pH=pH,
        pdb_string=getattr(evaluator, 'last_pdb_string', None),
    )
    
    # pH Stability Analysis
    print("\n[Step 4b] pH Stability Profile:")
    pH_profile = validator.validate_pH_stability(best_candidate[0])
    for pH_val, info in pH_profile.items():
        icon = "✅" if info["stable"] else "❌"
        print(f"  pH {pH_val:.1f}: {icon} (protonated: {info['protonation_fraction']:.0%}, "
              f"{info['protonated_count']}/{info['total_binding_residues']} residues)")
    
    # Ion Selectivity Analysis
    print("\n[Step 4c] Ion Selectivity Analysis:")
    selectivity = validator.check_ion_selectivity(
        best_candidate[1], best_candidate[0], binding_indices
    )
    
    # Pipeline Step 5: Export to PDB for ChimeraX
    output_filename = f"best_{target_ion}_lanmodulin_variant.pdb"
    print(f"\n[Step 5] Exporting best structure: {output_filename}")
    evaluator.write_pdb(best_candidate[0], best_candidate[1], filename=output_filename)
    
    # Final Summary
    print("\n" + "═" * 60)
    if md_result["stable"]:
        print(f"  ✅ SUCCESS: {ion_params['symbol']} binding complex is stable!")
        print(f"     RMSD: {md_result['rmsd']:.2f} Å")
        print(f"     Metal retained: {md_result['metal_retained']}")
        print(f"     pH stable: {md_result['pH_stable']}")
        print(f"     Ion selective: {selectivity['selective']}")
        print(f"     MD engine: {md_result['engine']}")
    else:
        print(f"  ❌ UNSTABLE: Complex destabilized during MD simulation.")
        print(f"     RMSD: {md_result['rmsd']:.2f} Å")
    print("═" * 60)
    
    return {
        "target_ion": target_ion,
        "best_sequence": best_candidate[0],
        "best_coords": best_candidate[1],
        "best_distance": best_candidate[2],
        "best_fitness": best_fitness,
        "fitness_breakdown": best_breakdown,
        "md_result": md_result,
        "pH_profile": pH_profile,
        "selectivity": selectivity,
        "output_pdb": output_filename,
    }


def run_multi_ion_pipeline(target_ions, **kwargs):
    """
    Run the LanRecov pipeline for multiple Lanthanide ions.
    Generates separate optimized variants for each ion.
    
    Args:
        target_ions (list[str]): List of ion names (e.g. ["La", "Nd", "Dy"]).
        **kwargs: Additional arguments passed to run_bioarchitect_pipeline.
    
    Returns:
        dict: {ion_name → pipeline_results}
    """
    print("█" * 60)
    print("█  BIO-ARCHITECT ENGINE — MULTI-ION LanRecov SCREENING  █")
    print("█" * 60)
    print(f"\n  Screening ions: {target_ions}")
    print(f"  Total runs: {len(target_ions)}")
    
    all_results = {}
    
    for i, ion in enumerate(target_ions):
        print(f"\n\n{'▓' * 60}")
        print(f"  RUN {i+1}/{len(target_ions)}: {LANTHANIDE_DB.get(ion, {}).get('symbol', ion)}")
        print(f"{'▓' * 60}")
        
        result = run_bioarchitect_pipeline(target_ion=ion, **kwargs)
        all_results[ion] = result
    
    # Summary
    print("\n\n" + "█" * 60)
    print("█  MULTI-ION SCREENING SUMMARY")
    print("█" * 60)
    print(f"\n  {'Ion':<8s} │ {'Fitness':>8s} │ {'Dist (Å)':>8s} │ {'RMSD':>6s} │ {'Stable':>6s} │ {'Selective':>9s}")
    print(f"  {'─'*8} ┼ {'─'*8} ┼ {'─'*8} ┼ {'─'*6} ┼ {'─'*6} ┼ {'─'*9}")
    
    for ion, result in all_results.items():
        params = LANTHANIDE_DB.get(ion, {})
        symbol = params.get("symbol", ion)
        print(f"  {symbol:<8s} │ {result['best_fitness']:8.4f} │ "
              f"{result['best_distance']:8.3f} │ "
              f"{result['md_result']['rmsd']:6.2f} │ "
              f"{'✅' if result['md_result']['stable'] else '❌':>6s} │ "
              f"{'✅' if result['selectivity']['selective'] else '❌':>9s}")
    
    # Best overall
    best_ion = max(all_results.items(), key=lambda x: x[1]["best_fitness"])
    print(f"\n  🏆 Best ion: {LANTHANIDE_DB.get(best_ion[0], {}).get('symbol', best_ion[0])} "
          f"(fitness: {best_ion[1]['best_fitness']:.4f})")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioArchitect Engine — LanRecov Pipeline")
    parser.add_argument("--ion", type=str, default="La",
                        help="Target Lanthanide ion (e.g. La, Nd, Dy)")
    parser.add_argument("--multi", type=str, default=None,
                        help="Comma-separated list of ions for multi-ion screening (e.g. La,Nd,Dy)")
    parser.add_argument("--pH", type=float, default=7.0,
                        help="Solution pH for MD validation (default: 7.0)")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of GA generations (default: 20)")
    parser.add_argument("--population", type=int, default=30,
                        help="Population size per generation (default: 30)")
    parser.add_argument("--mutation-rate", type=float, default=0.15,
                        help="GA mutation rate (default: 0.15, adaptive decay to 0.05)")
    parser.add_argument("--pH-resistant", action="store_true",
                        help="Bias mutations toward pH-resistant residues")
    
    args = parser.parse_args()
    
    if args.multi:
        ions = [ion.strip() for ion in args.multi.split(",")]
        run_multi_ion_pipeline(
            ions,
            pH=args.pH,
            generations=args.generations,
            population_size=args.population,
            pH_resistant=args.pH_resistant,
        )
    else:
        run_bioarchitect_pipeline(
            target_ion=args.ion,
            pH=args.pH,
            generations=args.generations,
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            pH_resistant=args.pH_resistant,
        )
