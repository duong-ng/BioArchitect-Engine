"""
==============================================================================
 BioArchitect Engine — Step 3: Genetic Optimizer with LanRecov Scoring
==============================================================================
 Genetic Algorithm integrated with Matrix Exponentiation for evaluating
 structural fitness of Lanmodulin mutants.
 
 LanRecov Enhancements:
   - CLIFF PENALTY for r > 3.0 Å — forces convergence to binding range
   - Ion-specific fitness scoring based on Lanthanide parameters
   - Binding energy estimation (Lennard-Jones-like potential)
   - Ion competition penalty (Fe³⁺, Ca²⁺, Al³⁺ rejection)
   - Enhanced Matrix Exponentiation (spectral radius coupling metric)
   - Adaptive mutation rate with generation-dependent decay
   - Tournament selection with crowding distance for diversity
==============================================================================
"""

import random
import numpy as np
from scipy.linalg import expm

from lanthanide_params import (
    get_ion_params,
    get_distance_range,
    estimate_binding_energy,
    calculate_selectivity_score,
    get_binding_residue_bias,
    LANTHANIDE_DB,
    COMPETING_IONS,
)


class BioArchitectGA:
    """
    Genetic Algorithm integrated with Matrix Exponentiation
    for evaluating structural fitness of Lanmodulin mutants.
    
    LanRecov: Ion-specific fitness with cliff penalty + binding energy
    + matrix exponential stability scoring.
    """
    
    # ── Recommended Hyperparameters ──────────────────────────────────────
    # These are tuned for La³⁺ recovery from acid mine drainage.
    RECOMMENDED_PARAMS = {
        "generations": 20,
        "population_size": 30,
        "mutation_rate_initial": 0.15,
        "mutation_rate_final": 0.05,
        "elitism": 3,
        "tournament_size": 5,
        "crossover_mode": "ef_hand_aware",
    }
    
    def __init__(self, target_ion="La", target_distance=None,
                 ef_hands=None, competing_ions=None):
        """
        Args:
            target_ion (str): Target Lanthanide ion (e.g. "La", "Nd", "Dy").
            target_distance (tuple, optional): (min, max) coordination distance in Å.
                If None, auto-set from lanthanide database.
            ef_hands (dict, optional): EF-hand loop definitions.
            competing_ions (list, optional): List of competing ion names.
        """
        self.target_ion = target_ion
        self.ef_hands = ef_hands or {}
        self.competing_ions = competing_ions or ["Fe3", "Ca2", "Al3"]
        
        # Auto-set target distance from ion database
        if target_distance is not None:
            self.target_distance = target_distance
        else:
            self.target_distance = get_distance_range(target_ion)
        
        # Load ion parameters
        self.ion_params = get_ion_params(target_ion)
        
        # ── Fitness weights (REBALANCED with folding confidence) ──────
        # Geometry dominates to force pocket convergence first
        # w_confidence added for ESMFold pLDDT/pTM integration
        self.w_geometry = 0.40     # Geometric distance score
        self.w_binding = 0.18     # Binding energy score
        self.w_stability = 0.15   # Network stability - matrix exp
        self.w_selectivity = 0.10 # Ion selectivity vs competitors
        self.w_coordination = 0.07 # Coordination geometry quality
        self.w_confidence = 0.10  # ESMFold confidence bonus (pLDDT + pTM)
        
        # Cliff penalty threshold: any distance beyond this gets
        # exponentially crushed
        self.cliff_threshold = 3.0  # Å
        
        print(f"[GA] Target: {self.ion_params['symbol']} | "
              f"Distance range: {self.target_distance[0]:.2f}–{self.target_distance[1]:.2f} Å | "
              f"Cliff penalty at r > {self.cliff_threshold:.1f} Å | "
              f"Competitors: {self.competing_ions}")
        
    def construct_adjacency_matrix(self, coords, distance_threshold=8.0):
        """
        Converts 3D coordinates into an Adjacency Matrix representing
        residue-residue interaction networks.
        
        Uses vectorized distance computation for performance.
        """
        L = len(coords)
        coords_arr = np.array(coords)
        
        # Vectorized pairwise distance computation
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Build adjacency: 1/distance for close contacts, 0 otherwise
        A = np.zeros((L, L))
        mask = (dist_matrix < distance_threshold) & (dist_matrix > 0)
        A[mask] = 1.0 / dist_matrix[mask]
        
        return A

    def matrix_exponentiation_score(self, A, mutation_indices, t=0.5):
        """
        Uses Matrix Exponentiation (e^At) to measure allosteric network robustness.
        
        Enhanced: Uses spectral radius of e^(At) sub-matrix as the coupling
        metric, which better captures long-range communication pathways
        through the protein network.
        
        Args:
            A (np.ndarray): Adjacency matrix.
            mutation_indices (list[int]): Indices of mutated residues.
            t (float): Time parameter (higher = longer-range effects).
        
        Returns:
            float: Coupling score (higher = better connected network).
        """
        n = len(A)
        if n == 0 or len(mutation_indices) == 0:
            return 0.0
        
        # Clamp indices to valid range
        valid_indices = [i for i in mutation_indices if 0 <= i < n]
        if not valid_indices:
            return 0.0
        
        # Calculate the matrix exponential e^(A * t)
        try:
            expA = expm(A * t)
        except Exception:
            return 0.0
        
        # Method 1: Average coupling strength between mutated and all residues
        coupling_rows = expA[valid_indices, :]
        avg_coupling = np.mean(coupling_rows)
        
        # Method 2: Spectral radius of the binding-site sub-network
        # This captures how well the binding site residues communicate
        # with each other through the protein network
        if len(valid_indices) >= 2:
            sub_matrix = expA[np.ix_(valid_indices, valid_indices)]
            try:
                eigenvalues = np.linalg.eigvals(sub_matrix)
                spectral_radius = np.max(np.abs(eigenvalues))
            except np.linalg.LinAlgError:
                spectral_radius = avg_coupling
        else:
            spectral_radius = avg_coupling
        
        # Combine: avg coupling (global connectivity) + spectral radius (local)
        combined = 0.5 * avg_coupling + 0.5 * (spectral_radius / max(1.0, n))
        
        return float(combined)

    def _geometric_score(self, distance):
        """
        Score based on distance to target coordination range.
        
        CLIFF PENALTY: For r > 3.0 Å, fitness drops exponentially.
        This forces the GA to converge to the binding range quickly.
        
        Returns:
            float: Score between 0 and 1 (1 = perfect fit).
        """
        low, high = self.target_distance
        
        if low <= distance <= high:
            # Perfect: within target range
            center = (low + high) / 2
            half_width = (high - low) / 2
            deviation = abs(distance - center) / half_width if half_width > 0 else 0
            return 1.0 - 0.1 * deviation  # 0.90–1.00
        
        elif distance > self.cliff_threshold:
            # ═══ CLIFF PENALTY ═══
            # Exponential decay beyond 3.0 Å — kills fitness of bad geometries
            # At r=4.0: score ≈ 0.135
            # At r=7.0: score ≈ 0.0003
            # At r=12.7: score ≈ 2.8e-9  (effectively zero)
            excess = distance - self.cliff_threshold
            return float(np.exp(-2.0 * excess))
        
        elif distance < low:
            # Too close: gentle quadratic penalty
            deficit = low - distance
            return max(0.01, 1.0 / (1.0 + 3.0 * deficit ** 2))
        
        else:
            # Between high and cliff_threshold: moderate penalty
            excess = distance - high
            return max(0.01, 1.0 / (1.0 + 5.0 * excess ** 2))

    def _binding_energy_score(self, distance, coordinating_residues=None):
        """
        Score based on estimated binding energy with the target ion.
        
        Args:
            distance (float): Measured coordination distance.
            coordinating_residues (list[str], optional): Residue types at binding site.
        
        Returns:
            float: Normalized score (0–1), higher = stronger binding.
        """
        if coordinating_residues is None:
            coordinating_residues = ["D"]  # Default: Aspartate
        
        # Average binding energy over coordinating residues
        energies = []
        for res in coordinating_residues:
            energy = estimate_binding_energy(distance, self.target_ion, res)
            energies.append(energy)
        
        avg_energy = np.mean(energies)
        
        # Normalize: map energy to 0–1 range
        # More negative = stronger binding = higher score
        score = 1.0 / (1.0 + np.exp(avg_energy + 2))  # Sigmoid normalization
        return float(score)

    def _selectivity_score(self, distance):
        """
        Penalize designs that would also bind competing ions well.
        We want high selectivity for the target Lanthanide.
        
        Returns:
            float: Score (0–1), higher = better selectivity.
        """
        result = calculate_selectivity_score(
            distance, self.target_ion, self.competing_ions
        )
        
        selectivity = result["selectivity_score"]
        
        # Normalize to 0–1
        score = min(1.0, selectivity / 5.0)
        return float(score)

    def _coordination_score(self, coords, binding_indices):
        """
        Evaluate quality of coordination geometry around the binding site.
        Good coordination: residues arranged symmetrically around a central point.
        
        Args:
            coords (np.ndarray): CA coordinates.
            binding_indices (list[int]): Indices of coordinating residues.
        
        Returns:
            float: Score (0–1), higher = better geometry.
        """
        if len(binding_indices) < 2:
            return 0.5
        
        # Clamp indices to valid range
        valid_indices = [i for i in binding_indices if i < len(coords)]
        if len(valid_indices) < 2:
            return 0.5
        
        # Get binding site coordinates
        binding_coords = coords[valid_indices]
        
        # Calculate centroid
        centroid = np.mean(binding_coords, axis=0)
        
        # Calculate distances from centroid
        dists_to_center = np.linalg.norm(binding_coords - centroid, axis=1)
        
        # Good coordination = uniform distances (low variance)
        mean_dist = np.mean(dists_to_center)
        if mean_dist == 0:
            return 0.0
        
        cv = np.std(dists_to_center) / mean_dist  # Coefficient of variation
        
        # Lower CV = more symmetric = better
        score = 1.0 / (1.0 + cv * 3)
        
        # Also check if CN matches expected coordination number
        expected_cn = self.ion_params.get("coordination_number", 9)
        cn_match = 1.0 - abs(len(valid_indices) - expected_cn) / expected_cn
        cn_match = max(0.0, cn_match)
        
        return float(0.6 * score + 0.4 * cn_match)

    def _confidence_bonus(self, confidence_scores, binding_indices=None):
        """
        Compute confidence bonus from folding evaluator pLDDT and PAE scores.
        
        Higher pLDDT in EF-hand regions = more trustworthy pocket geometry.
        Lower PAE between binding residues = higher inter-residue accuracy.
        
        ConfidenceBonus = ef_hand_plddt_norm * (1 - ef_hand_pae_norm)
        
        When confidence_scores is None (no PAE available),
        returns a neutral score of 0.5 so it does not penalize or boost.
        
        Args:
            confidence_scores (dict or None): From evaluator.get_confidence_scores().
                Expected keys: 'plddt' (np.ndarray), 'mean_plddt' (float),
                               'pae' (np.ndarray or None), 'ef_hand_plddt' (dict).
            binding_indices (list[int], optional): Binding site residue indices.
        
        Returns:
            float: Score (0–1), higher = more confident prediction.
        """
        if confidence_scores is None:
            return 0.5  # Neutral — no confidence data available
        
        # ── pLDDT component ──────────────────────────────────────────
        # Use EF-hand-specific pLDDT if available, else global mean
        ef_hand_plddt = confidence_scores.get("ef_hand_plddt", {})
        if ef_hand_plddt:
            mean_ef_plddt = float(np.mean(list(ef_hand_plddt.values())))
        elif binding_indices is not None:
            plddt = confidence_scores.get("plddt", None)
            if plddt is not None and len(plddt) > 0:
                valid_idx = [i for i in binding_indices if i < len(plddt)]
                mean_ef_plddt = float(np.mean(plddt[valid_idx])) if valid_idx else 50.0
            else:
                mean_ef_plddt = confidence_scores.get("mean_plddt", 50.0)
        else:
            mean_ef_plddt = confidence_scores.get("mean_plddt", 50.0)
        
        # Normalize pLDDT to 0–1 (scores are 0–100)
        plddt_norm = min(1.0, max(0.0, mean_ef_plddt / 100.0))
        
        # ── PAE component ────────────────────────────────────────────
        pae = confidence_scores.get("pae", None)
        ef_hand_pae = confidence_scores.get("ef_hand_pae", None)
        
        if ef_hand_pae is not None:
            # Lower PAE = better. Typical range: 0–30 Å
            pae_norm = min(1.0, max(0.0, ef_hand_pae / 30.0))
        elif pae is not None and binding_indices is not None:
            # Compute PAE between binding site residues
            n = pae.shape[0] if hasattr(pae, 'shape') else len(pae)
            valid_idx = [i for i in binding_indices if i < n]
            if len(valid_idx) >= 2:
                pae_arr = np.array(pae)
                sub = pae_arr[np.ix_(valid_idx, valid_idx)]
                mean_binding_pae = float(np.mean(sub))
                pae_norm = min(1.0, max(0.0, mean_binding_pae / 30.0))
            else:
                pae_norm = 0.5  # Neutral
        else:
            pae_norm = 0.5  # Neutral — no PAE data
        
        # ── Combined score ───────────────────────────────────────────
        # High pLDDT (close to 1) AND low PAE (close to 0) = high bonus
        bonus = plddt_norm * (1.0 - pae_norm)
        
        return float(bonus)

    def calculate_fitness(self, coords, target_res_1, target_res_2,
                          sequence=None, binding_indices=None,
                          confidence_scores=None):
        """
        LanRecov Fitness Function with CLIFF PENALTY + Folding Confidence:
        
            F = w₁·Geometric(cliff) + w₂·BindingEnergy + w₃·Stability + 
                w₄·Selectivity + w₅·Coordination + w₆·ConfidenceBonus
        
        The geometric term uses an exponential cliff penalty for r > 3.0 Å.
        The confidence bonus uses ESMFold pLDDT and PAE to weight 
        how much we trust the predicted pocket geometry.
        
        Args:
            coords (np.ndarray): CA coordinates from ESMFold.
            target_res_1 (int): First target residue index.
            target_res_2 (int): Second target residue index.
            sequence (str, optional): Protein sequence for residue type analysis.
            binding_indices (list[int], optional): All binding site residue indices.
            confidence_scores (dict, optional): From evaluator.get_confidence_scores().
                Contains 'plddt', 'pae', 'ef_hand_plddt', 'ef_hand_pae'.
                When None, confidence bonus defaults to 0.5 (neutral).
            
        Returns:
            tuple: (fitness, distance, score_breakdown)
        """
        # Validate indices
        max_idx = len(coords) - 1
        target_res_1 = min(target_res_1, max_idx)
        target_res_2 = min(target_res_2, max_idx)
        
        # 1. Geometric distance with CLIFF PENALTY
        p1 = coords[target_res_1]
        p2 = coords[target_res_2]
        r = float(np.linalg.norm(p1 - p2))
        
        geo_score = self._geometric_score(r)
        
        # 2. Binding energy
        coord_residues = ["D"]
        if sequence:
            coord_residues = []
            check_indices = binding_indices if binding_indices else [target_res_1, target_res_2]
            for idx in check_indices:
                if idx < len(sequence):
                    coord_residues.append(sequence[idx])
            if not coord_residues:
                coord_residues = ["D"]
        
        bind_score = self._binding_energy_score(r, coord_residues)
        
        # 3. Network Stability (Enhanced Matrix Exponentiation)
        A = self.construct_adjacency_matrix(coords)
        mut_indices = binding_indices if binding_indices else [target_res_1, target_res_2]
        stability_score = self.matrix_exponentiation_score(A, mut_indices, t=0.5)
        # Normalize stability to 0–1 range
        stability_score = min(1.0, stability_score / 0.5)
        
        # 4. Selectivity vs competing ions
        select_score = self._selectivity_score(r)
        
        # 5. Coordination geometry
        coord_indices = binding_indices if binding_indices else [target_res_1, target_res_2]
        coord_score = self._coordination_score(coords, coord_indices)
        
        # 6. ESMFold Confidence bonus (pLDDT + pTM)
        conf_score = self._confidence_bonus(confidence_scores, coord_indices)
        
        # ── Weighted total fitness ──────────────────────────────────────
        # The cliff penalty in geo_score already crushes bad candidates,
        # so the weighted sum naturally produces near-zero fitness for
        # r > 3.0 Å structures.
        fitness = (
            self.w_geometry * geo_score +
            self.w_binding * bind_score +
            self.w_stability * stability_score +
            self.w_selectivity * select_score +
            self.w_coordination * coord_score +
            self.w_confidence * conf_score
        )
        
        # Additional penalty multiplier: if distance is extremely bad
        # (>5 Å), apply an overall multiplier to crush all sub-scores
        if r > 5.0:
            distance_multiplier = float(np.exp(-0.5 * (r - 5.0)))
            fitness *= distance_multiplier
        
        score_breakdown = {
            "geometric": geo_score,
            "binding_energy": bind_score,
            "stability": stability_score,
            "selectivity": select_score,
            "coordination": coord_score,
            "confidence": conf_score,
            "total": fitness,
        }
        
        return fitness, r, score_breakdown

    def get_adaptive_mutation_rate(self, generation, total_generations,
                                   initial_rate=0.15, final_rate=0.05):
        """
        Adaptive mutation rate: high initially (explore), low later (exploit).
        Uses cosine annealing for smooth decay.
        
        Args:
            generation (int): Current generation (0-indexed).
            total_generations (int): Total number of generations.
            initial_rate (float): Starting mutation rate.
            final_rate (float): Ending mutation rate.
        
        Returns:
            float: Mutation rate for this generation.
        """
        if total_generations <= 1:
            return initial_rate
        
        progress = generation / (total_generations - 1)
        # Cosine annealing
        rate = final_rate + 0.5 * (initial_rate - final_rate) * (
            1 + np.cos(np.pi * progress)
        )
        return float(rate)

    def mutate(self, sequence, mutation_rate=0.05):
        """
        Point mutation operator: randomly mutate residues in EF-hand regions
        using ion-specific amino acid preferences.
        
        Args:
            sequence (str): Protein sequence.
            mutation_rate (float): Probability of mutation per EF-hand residue.
        
        Returns:
            str: Mutated sequence.
        """
        seq_list = list(sequence)
        
        try:
            bias = get_binding_residue_bias(self.target_ion)
            mutation_aas = bias["residues"]
            mutation_weights = bias["weights"]
        except KeyError:
            mutation_aas = list("DENQST")
            mutation_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        
        # Only mutate within EF-hand regions
        if self.ef_hands:
            mutable_positions = []
            for loop_name, (start, end) in self.ef_hands.items():
                mutable_positions.extend(range(start, end + 1))
        else:
            # If no EF-hands defined, allow full sequence mutation
            mutable_positions = list(range(len(sequence)))
        
        for pos in mutable_positions:
            if random.random() < mutation_rate and pos < len(seq_list):
                seq_list[pos] = random.choices(mutation_aas, weights=mutation_weights, k=1)[0]
        
        return "".join(seq_list)

    def crossover(self, seq1, seq2, mode="ef_hand_aware"):
        """
        Crossover operator.
        
        Args:
            seq1 (str): Parent 1 sequence.
            seq2 (str): Parent 2 sequence.
            mode (str): "single_point", "multi_point", or "ef_hand_aware".
        
        Returns:
            str: Child sequence.
        """
        if mode == "single_point":
            pt = len(seq1) // 2
            return seq1[:pt] + seq2[pt:]
        
        elif mode == "multi_point":
            # Two-point crossover
            pts = sorted(random.sample(range(1, len(seq1)), 2))
            return seq1[:pts[0]] + seq2[pts[0]:pts[1]] + seq1[pts[1]:]
        
        elif mode == "ef_hand_aware" and self.ef_hands:
            # Crossover at EF-hand boundaries
            # Child inherits entire EF-hand loops from either parent
            child_list = list(seq1)  # Start with parent 1
            for loop_name, (start, end) in self.ef_hands.items():
                if random.random() > 0.5:
                    # Take this EF-hand from parent 2
                    for i in range(start, min(end + 1, len(seq2))):
                        if i < len(child_list):
                            child_list[i] = seq2[i]
            return "".join(child_list)
        
        else:
            # Default: single point
            pt = len(seq1) // 2
            return seq1[:pt] + seq2[pt:]

    def select_tournament(self, scored_population, tournament_size=5):
        """
        Tournament selection: pick best individual from random subset.
        Increased tournament_size (5 vs 3) for stronger selection pressure.
        
        Args:
            scored_population (list): List of (fitness, distance, breakdown, seq, coords) tuples.
            tournament_size (int): Number of candidates per tournament.
        
        Returns:
            tuple: Selected individual.
        """
        tournament = random.sample(scored_population, min(tournament_size, len(scored_population)))
        return max(tournament, key=lambda x: x[0])

    def _crowding_distance(self, scored_population):
        """
        Calculate crowding distance to preserve diversity.
        Individuals with higher crowding distance are in sparser regions
        of the fitness landscape and should be preferred for diversity.
        
        Args:
            scored_population (list): List of (fitness, distance, breakdown, seq, coords).
        
        Returns:
            list[float]: Crowding distances for each individual.
        """
        n = len(scored_population)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        
        # Sort by fitness
        indices_by_fitness = sorted(range(n), key=lambda i: scored_population[i][0])
        distances[indices_by_fitness[0]] = float('inf')
        distances[indices_by_fitness[-1]] = float('inf')
        
        f_range = (scored_population[indices_by_fitness[-1]][0] -
                   scored_population[indices_by_fitness[0]][0])
        if f_range > 0:
            for i in range(1, n - 1):
                idx = indices_by_fitness[i]
                d = (scored_population[indices_by_fitness[i + 1]][0] -
                     scored_population[indices_by_fitness[i - 1]][0])
                distances[idx] += d / f_range
        
        # Sort by distance
        indices_by_dist = sorted(range(n), key=lambda i: scored_population[i][1])
        distances[indices_by_dist[0]] = float('inf')
        distances[indices_by_dist[-1]] = float('inf')
        
        d_range = (scored_population[indices_by_dist[-1]][1] -
                   scored_population[indices_by_dist[0]][1])
        if d_range > 0:
            for i in range(1, n - 1):
                idx = indices_by_dist[i]
                d = (scored_population[indices_by_dist[i + 1]][1] -
                     scored_population[indices_by_dist[i - 1]][1])
                distances[idx] += d / d_range
        
        return distances

    def evolve_population(self, scored_population, population_size,
                          elitism=3, mutation_rate=0.15,
                          generation=None, total_generations=None):
        """
        Create next generation using selection, crossover, and mutation.
        
        Supports adaptive mutation rate when generation info is provided.
        
        Args:
            scored_population (list): List of (fitness, dist, breakdown, seq, coords).
            population_size (int): Desired population size.
            elitism (int): Number of top individuals to carry forward unchanged.
            mutation_rate (float): Per-residue mutation probability (base rate).
            generation (int, optional): Current generation for adaptive rate.
            total_generations (int, optional): Total generations for adaptive rate.
        
        Returns:
            list[str]: New population of sequences.
        """
        # Sort by fitness (descending)
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # Adaptive mutation rate
        if generation is not None and total_generations is not None:
            actual_rate = self.get_adaptive_mutation_rate(
                generation, total_generations,
                initial_rate=mutation_rate,
                final_rate=0.05,
            )
        else:
            actual_rate = mutation_rate
        
        next_gen = []
        
        # Elitism: carry top individuals unchanged
        for i in range(min(elitism, len(scored_population))):
            next_gen.append(scored_population[i][3])  # 3 = seq index
        
        # Calculate crowding distances for diversity-aware selection
        crowd_dists = self._crowding_distance(scored_population)
        
        # Fill rest with crossover + mutation
        while len(next_gen) < population_size:
            parent1 = self.select_tournament(scored_population)
            parent2 = self.select_tournament(scored_population)
            
            child = self.crossover(parent1[3], parent2[3])
            child = self.mutate(child, actual_rate)
            next_gen.append(child)
        
        return next_gen[:population_size]
