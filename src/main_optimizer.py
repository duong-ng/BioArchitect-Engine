import time
from step1_mutation_generator import ProteinMPNNGenerator
from step2_folding_evaluator import ESMFoldEvaluator
from step3_genetic_optimizer import BioArchitectGA
from step4_md_validation import MDValidator

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

def run_bioarchitect_pipeline():
    print("="*50)
    print("🧬 BIO-ARCHITECT ENGINE PIPELINE STARTED")
    print("="*50)
    
    # Init modules
    generator = ProteinMPNNGenerator(base_fasta, ef_hands)
    evaluator = ESMFoldEvaluator(device='cpu')
    ga_optimizer = BioArchitectGA(target_distance=(0.7, 1.1))
    validator = MDValidator()

    # Pipeline Step 1: Initial Population via ProteinMPNN
    print(f"\n[Step 1] Initializing Population.")
    population = generator.generate_population(population_size=10, pdb_path=base_pdb)
    
    # Assume we are targeting the interaction between Asp(8) and Glu(9) within EF4
    # Real indices derived from total sequence
    target_res_1 = ef_hands["EF4"][0] + 2 # e.g., an Aspartate
    target_res_2 = ef_hands["EF4"][1] - 1 # e.g., a Glutamate

    generations = 3
    best_candidate = None
    best_fitness = -1

    for gen in range(generations):
        print(f"\n--- Generation {gen+1} ---")
        scored_population = []
        
        for idx, seq in enumerate(population):
            # Pipeline Step 2: Ultra-fast Folding using ESMFold
            coords = evaluator.predict_structure(seq)
            
            # Pipeline Step 3: Evaluate Fitness (Matrix Exponentiation + GA)
            fitness, dist = ga_optimizer.calculate_fitness(coords, target_res_1, target_res_2)
            scored_population.append((fitness, dist, seq, coords))
            
        # Sort by fitness descending
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        top_fitness, top_dist, top_seq, top_coords = scored_population[0]
        print(f"🥇 Best in Gen {gen+1} | Dist: {top_dist:.3f}A | Fitness: {top_fitness:.4f}")
        
        if top_fitness > best_fitness:
            best_fitness = top_fitness
            best_candidate = (top_seq, top_coords, top_dist)

        # Crossover & Mutate for next generation (simplified)
        next_gen = []
        for i in range(len(population)):
            parent1 = scored_population[i % 5][2] # Top 5 elitism
            parent2 = scored_population[(i+1) % 5][2]
            child = ga_optimizer.crossover(parent1, parent2)
            next_gen.append(child)
        population = next_gen

    print("\n" + "="*50)
    print("🔥 PIPELINE OPTIMIZATION COMPLETE 🔥")
    print(f"Optimal Sequence Selected with pocket distance: {best_candidate[2]:.3f}A")
    
    # Pipeline Step 4: Molecular Dynamics Validation
    print("\n[Running Final MD Validation on Best Candidate]")
    md_result = validator.run_simulation(best_candidate[0], best_candidate[1], metal="La", steps=10000)
    
    # Pipeline Step 5: Export to PDB for ChimeraX
    print("\n[Exporting best structure for 3D Visualization]")
    evaluator.write_pdb(best_candidate[0], best_candidate[1], filename="best_lanmodulin_variant.pdb")
    
    if md_result["stable"]:
        print("\n✅ SUCCESS: Metal-binding complex is stable in aqueous environment.")
        print(f"Final RMSD: {md_result['rmsd']:.2f}")
    else:
        print("\n❌ FAILED: Complex destabilized during MD simulation.")

if __name__ == "__main__":
    run_bioarchitect_pipeline()

