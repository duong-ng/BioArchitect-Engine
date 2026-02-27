import numpy as np
from scipy.linalg import expm

class BioArchitectGA:
    """
    Genetic Algorithm integrated with Matrix Exponentiation
    for evaluating structural fitness of Lanmodulin mutants.
    """
    def __init__(self, target_distance=(0.7, 1.1)):
        self.target_distance = target_distance
        
    def construct_adjacency_matrix(self, coords, distance_threshold=8.0):
        """
        Converts 3D coordinates into an Adjacency Matrix representing
        residue-residue interaction networks.
        """
        L = len(coords)
        A = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < distance_threshold:
                        # simple interaction strength
                        A[i, j] = 1.0 / dist 
        return A

    def matrix_exponentiation_score(self, A, mutation_indices, t=0.5):
        """
        Uses Matrix Exponentiation (e^At) to measure allosteric network robustness.
        We check if the target target mutation areas disrupt communication 
        across the protein.
        """
        # Calculate the matrix exponential e^(A * t)
        # expA[i, j] represents the transmission of 'information' or dynamic coupling 
        # from residue i to residue j over structural distance.
        expA = expm(A * t)
        
        # We want to measure how well the mutated residues (pocket) 
        # still communicate with the rest of the protein (global stability).
        # A good mutation maintains strong coupling to the scaffold.
        
        coupling_score = 0
        for idx in mutation_indices:
            # Sum of information flow from this mutated residue to all other residues
            coupling_score += np.sum(expA[idx, :])
            # Normalize by protein length
        
        # We average the coupling
        avg_coupling = coupling_score / (len(mutation_indices) * len(A))
        
        return avg_coupling

    def calculate_fitness(self, coords, target_res_1, target_res_2):
        """
        Fitness = Geometric Penalty + 
                  Matrix Exponentiation Stability
        """
        # 1. Geometric distance (Must be within 0.7 - 1.1 Angstroms)
        p1 = coords[target_res_1]
        p2 = coords[target_res_2]
        r = np.linalg.norm(p1 - p2)
        
        penalty = 0
        if r < self.target_distance[0]:
            penalty = self.target_distance[0] - r
        elif r > self.target_distance[1]:
            penalty = r - self.target_distance[1]
            
        geometric_score = 1.0 / (1.0 + penalty)
        
        # 2. Network Stability
        A = self.construct_adjacency_matrix(coords)
        stability_score = self.matrix_exponentiation_score(A, [target_res_1, target_res_2])
        
        # Total fitness
        fitness = (geometric_score * 0.7) + (stability_score * 0.3)
        return fitness, r

    def crossover(self, seq1, seq2):
        """Single point crossover"""
        pt = len(seq1) // 2
        return seq1[:pt] + seq2[pt:]
