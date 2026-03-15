<p align="center">
  <h1 align="center">🧬 BioArchitect Engine</h1>
  <p align="center">
    <strong>AI-Powered Protein Engineering for Selective Rare Earth Element Recovery</strong>
  </p>
  <p align="center">
    <a href="#pipeline-overview">Pipeline</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-reference">API</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Overview](#pipeline-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Data Files](#data-files)
- [License](#license)

---

## Overview

**BioArchitect Engine** is a computational protein engineering platform that optimizes the structure of [**Lanmodulin**](https://www.rcsb.org/structure/6MI5) — a naturally occurring Lanthanide-binding protein — to selectively recover **Rare Earth Elements (REEs)** from acid mine drainage (AMD).

The engine combines four state-of-the-art AI/computational methods into a single end-to-end pipeline:

| Step | Method | Purpose |
|------|--------|---------|
| **1** | **ProteinMPNN** | Ion-biased mutation generation in EF-hand binding regions |
| **2** | **AlphaFold2** | High-accuracy 3D structure prediction with confidence scoring |
| **3** | **Genetic Algorithm + Matrix Exponentiation** | Multi-objective fitness optimization with cliff penalty |
| **4** | **Molecular Dynamics** | Structural stability validation (OpenMM / ASE / analytical) |

The **LanRecov** enhancement extends the pipeline to support **ion-specific optimization** for individual Lanthanides (La³⁺, Nd³⁺, Dy³⁺, etc.) with pH resistance, competitive ion selectivity, and binding energy scoring.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   BioArchitect Engine Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Step 1       │    │  Step 2       │    │  Step 3           │   │
│  │  ProteinMPNN  │───▶│  AlphaFold2   │───▶│  Genetic          │   │
│  │  Mutation     │    │  Structure    │    │  Algorithm +      │   │
│  │  Generator    │    │  Prediction   │    │  Matrix Exp.      │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                                           ┌────────▼─────────┐   │
│                 ┌──────────────┐           │  Step 4           │   │
│                 │  Step 5       │◀──────────│  Molecular       │   │
│                 │  PDB Export   │           │  Dynamics         │   │
│                 │  (ChimeraX)   │           │  Validation       │   │
│                 └──────────────┘           └──────────────────┘   │
│                                                                 │
│  Loop: Population evolves over N generations (GA selection,     │
│        crossover, adaptive mutation) until convergence.         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🔬 Core Capabilities

- **ProteinMPNN Native Integration** — Runs the ProteinMPNN neural network entirely in PyTorch for mutation generation, with fallback to heuristic ion-biased sampling
- **AlphaFold2 Structure Prediction** — Supports LocalColabFold (MSA-based) and OpenFold (single-sequence) backends with PAE and pLDDT confidence scoring
- **Multi-Objective Genetic Algorithm** — 6-component fitness function: geometry, binding energy, stability, selectivity, coordination quality, and AF2 confidence
- **Molecular Dynamics Validation**h — OpenMM (explicit solvent), ASE (ML potentials), or enhanced physics-based mock engine

### 🧪 LanRecov Enhancements

- **Ion-Specific Optimization** — Tailored parameters for all 15 Lanthanides (La → Lu) from a comprehensive database
- **Multi-Ion Screening** — Parallel optimization for multiple target ions in a single run
- **pH Resistance** — Mutation bias toward pH-stable residues for operation in acidic conditions (pH < 3.0)
- **Competitive Ion Selectivity** — Penalty scoring against common interfering ions (Fe³⁺, Ca²⁺, Al³⁺, Mg²⁺, Cu²⁺, Pb²⁺)
- **Cliff Penalty** — Exponential fitness penalty for coordination distances > 3.0 Å, forcing rapid convergence

### 📊 Analysis & Output

- **pH Stability Profiling** — Protonation state analysis across pH 1.0 – 7.0
- **Ion Selectivity Reports** — Binding energy comparison between target and competing ions
- **PDB Export** — Direct export for UCSF ChimeraX visualization
- **Detailed Score Breakdown** — Per-generation reporting of all fitness components

---

## Project Structure

```
BioArchitect Engine/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── data/
│   ├── 6MI5.pdb                    # Wild-type Lanmodulin crystal structure
│   ├── 6MI5.cif                    # CIF format structure
│   └── rcsb_pdb_6MI5.fasta         # FASTA sequence (Methylobacterium extorquens)
├── docs/
│   ├── Bio-Architect.txt           # Project overview (Vietnamese)
│   ├── Bio-Architect-updates.txt   # LanRecov enhancement details
│   ├── Bio-Architect.docx          # Full documentation
│   ├── Bio-Architect-updates.docx  # Update documentation
│   └── prompt.txt                  # AI design prompts
├── src/
│   ├── main_optimizer.py           # Main pipeline orchestrator
│   ├── lanthanide_params.py        # Lanthanide ion parameter database
│   ├── step1_mutation_generator.py # ProteinMPNN mutation generator
│   ├── step2_af2_folding_evaluator.py  # AlphaFold2 structure predictor
│   ├── step3_genetic_optimizer.py  # Genetic Algorithm + Matrix Exp.
│   ├── step4_md_validation.py      # Molecular Dynamics validation
│   ├── _verify_all.py              # Module verification script
│   └── proteinmpnn_core/           # ProteinMPNN model weights & utils
│       ├── protein_mpnn_utils.py
│       └── vanilla_model_weights/
│           └── v_48_020.pt
└── best_*_lanmodulin_variant.pdb   # Output: optimized structures
```

---

## Installation

### Prerequisites

- **Python** 3.10+
- **PyTorch** 1.12+ (CUDA recommended for GPU acceleration)
- **NumPy**, **SciPy**

### Required Dependencies

```bash
pip install torch numpy scipy
```

### Optional Dependencies (for full functionality)

```bash
# AlphaFold2 backends
pip install colabfold[alphafold2]   # LocalColabFold (recommended)
pip install openfold                # OpenFold (single-sequence fallback)

# Molecular Dynamics engines
pip install openmm                  # Full explicit-solvent MD
pip install ase                     # ASE with ML potentials (MACE-OFF23, ANI-2x)

# PDB preprocessing
pip install pdbfixer                # Fix missing atoms, protonation states
```

> **Note:** The engine gracefully degrades when optional dependencies are missing. Without ColabFold/OpenFold, the AF2 evaluator uses a CA-only mock predictor. Without OpenMM/ASE, the MD validator uses an enhanced physics-based estimator.

---

## Usage

### Single-Ion Optimization (Default: La³⁺)

```bash
cd src
python main_optimizer.py
```

### Specify Target Ion

```bash
python main_optimizer.py --ion Nd
python main_optimizer.py --ion Dy --pH 3.0 --pH-resistant
```

### Multi-Ion Screening

```bash
python main_optimizer.py --multi La,Nd,Dy --generations 30 --population 50
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ion` | `La` | Target Lanthanide ion (e.g., `La`, `Nd`, `Dy`, `Eu`, `Tb`) |
| `--multi` | — | Comma-separated ion list for multi-ion screening |
| `--pH` | `7.0` | Solution pH for MD validation |
| `--generations` | `20` | Number of GA evolution generations |
| `--population` | `30` | Population size per generation |
| `--mutation-rate` | `0.15` | Initial GA mutation rate (cosine annealing to 0.05) |
| `--pH-resistant` | `false` | Bias mutations toward pH-stable residues |

### Programmatic Usage

```python
from main_optimizer import run_bioarchitect_pipeline, run_multi_ion_pipeline

# Single ion
result = run_bioarchitect_pipeline(
    target_ion="La",
    pH=7.0,
    generations=20,
    population_size=30,
)

# Access results
print(f"Best fitness: {result['best_fitness']:.4f}")
print(f"Best distance: {result['best_distance']:.3f} Å")
print(f"MD stable: {result['md_result']['stable']}")
print(f"Output PDB: {result['output_pdb']}")

# Multi-ion screening
results = run_multi_ion_pipeline(
    target_ions=["La", "Nd", "Dy"],
    pH=3.0,
    pH_resistant=True,
)
```

---

## Architecture

### Fitness Function (LanRecov)

The Genetic Algorithm uses a weighted 6-component fitness function:

```
F = w₁·Geometric(cliff) + w₂·BindingEnergy + w₃·Stability +
    w₄·Selectivity + w₅·Coordination + w₆·ConfidenceBonus
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Geometric** | `w₁` | Distance to optimal coordination range (2.3–2.6 Å) with exponential cliff penalty |
| **Binding Energy** | `w₂` | Lennard-Jones-like metal-ligand interaction potential |
| **Stability** | `w₃` | Matrix Exponentiation allosteric network robustness score |
| **Selectivity** | `w₄` | Preference for target ion over competing ions (Fe³⁺, Ca²⁺, etc.) |
| **Coordination** | `w₅` | Symmetry quality of coordinating residues around binding site |
| **Confidence** | `w₆` | AlphaFold2 pLDDT and PAE confidence bonus in EF-hand regions |

### EF-Hand Loop Definitions

Lanmodulin contains 4 EF-hand calcium-binding motifs that are repurposed for Lanthanide binding:

| Loop | Residues | Position |
|------|----------|----------|
| EF1 | 12 – 23 | N-terminal |
| EF2 | 35 – 46 | Central |
| EF3 | 63 – 74 | Central |
| EF4 | 90 – 101 | C-terminal |

### Lanthanide Database

The `lanthanide_params.py` module provides comprehensive parameters for all 15 Lanthanides plus 6 competing ions:

- **Ionic radius** and **optimal coordination distance** (Å)
- **Coordination number** and **formal charge**
- **Electronegativity** (Pauling) and **HSAB hardness** (eV)
- **Group classification** (light / middle / heavy REE)
- **EF-hand binding residue preferences** with weighted amino acid profiles
- **Competing ions**: Ca²⁺, Mg²⁺, Fe³⁺, Al³⁺, Cu²⁺, Pb²⁺

---

## Module Reference

### `main_optimizer.py` — Pipeline Orchestrator

The central module that coordinates all pipeline steps.

| Function | Description |
|----------|-------------|
| `run_bioarchitect_pipeline()` | Run full optimization for a single target ion |
| `run_multi_ion_pipeline()` | Run optimization for multiple ions with comparative summary |

### `step1_mutation_generator.py` — ProteinMPNN Generator

Generates mutant protein sequences targeting EF-hand binding regions.

| Class / Function | Description |
|-----------------|-------------|
| `ProteinMPNNGenerator` | Main generator class with native PyTorch ProteinMPNN integration |
| `ProteinMPNNGenerator.generate_population()` | Generate a population of mutant sequences |
| `setup_proteinmpnn()` | Auto-download ProteinMPNN model weights |

### `step2_af2_folding_evaluator.py` — AlphaFold2 Evaluator

Predicts 3D protein structure with confidence scoring.

| Class / Function | Description |
|-----------------|-------------|
| `AlphaFold2Evaluator` | Main evaluator with ColabFold/OpenFold backend auto-detection |
| `AlphaFold2Evaluator.predict_structure()` | Predict CA coordinates from sequence |
| `AlphaFold2Evaluator.predict_full()` | Full prediction with pLDDT, PAE, pTM scores |
| `AlphaFold2Evaluator.write_pdb()` | Export structure to PDB file |

### `step3_genetic_optimizer.py` — Genetic Algorithm

Multi-objective optimization with Matrix Exponentiation.

| Class / Function | Description |
|-----------------|-------------|
| `BioArchitectGA` | GA with LanRecov ion-specific fitness function |
| `BioArchitectGA.calculate_fitness()` | 6-component fitness evaluation |
| `BioArchitectGA.matrix_exponentiation_score()` | Allosteric network coupling via e^(At) |
| `BioArchitectGA.evolve_population()` | Selection, crossover, and adaptive mutation |
| `BioArchitectGA.mutate()` | Ion-biased point mutation in EF-hand regions |
| `BioArchitectGA.crossover()` | EF-hand-aware crossover operator |

### `step4_md_validation.py` — Molecular Dynamics Validator

Validates structural stability through simulation.

| Class / Function | Description |
|-----------------|-------------|
| `MDValidator` | MD validator with OpenMM/ASE/mock engine auto-selection |
| `MDValidator.run_simulation()` | Run MD simulation with metal ion placement |
| `MDValidator.validate_pH_stability()` | Test stability across pH range |
| `MDValidator.check_ion_selectivity()` | Compare binding affinity for competing ions |

### `lanthanide_params.py` — Ion Database

Comprehensive Lanthanide and competing ion parameter database.

| Export | Description |
|--------|-------------|
| `LANTHANIDE_DB` | Dictionary of all 15 Lanthanide parameters |
| `COMPETING_IONS` | Dictionary of 6 common interfering ion parameters |
| `get_ion_params()` | Retrieve parameters for any ion |
| `get_distance_range()` | Get acceptable coordination distance range |
| `get_binding_residue_bias()` | Get preferred amino acids for a target ion |
| `estimate_binding_energy()` | Lennard-Jones-like binding energy estimation |
| `calculate_selectivity_score()` | Compare binding preference vs. competitors |

---

## Data Files

| File | Description |
|------|-------------|
| `data/6MI5.pdb` | Wild-type Lanmodulin crystal structure from [RCSB PDB](https://www.rcsb.org/structure/6MI5) |
| `data/6MI5.cif` | CIF format of the same structure |
| `data/rcsb_pdb_6MI5.fasta` | FASTA sequence — *Methylobacterium extorquens* AM1 (114 residues + His-tag) |

The base sequence used by the pipeline:
```
PTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADLKKL
DPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIRHHHHHH
```

---

## Verification

Run the built-in verification script to check all modules:

```bash
cd src
python _verify_all.py
```

This performs:
1. **Syntax check** — Compiles all Python modules
2. **Import test** — Verifies module loading without GPU dependencies
3. **GA cliff penalty test** — Validates fitness function ordering (good > mid > bad distances)

---

## Context: Bio-Architect System

BioArchitect Engine is the **AI in-silico protein design** component of a larger two-stage Rare Earth Element recovery system:

1. **Stage 1 (Green Membrane)** — Chitosan/Biochar membrane pre-filtration removes 80–90% of interfering heavy metals (Fe, Al, Cu, Pb) from acid mine drainage
2. **Stage 2 (Bio-Architect)** — AI-optimized Lanmodulin protein selectively captures target Lanthanide ions from the pre-filtered solution

The full system integrates IoT monitoring (ESP32) for real-time process control, with pH/TDS sensors and adaptive flow rate management.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 duong-ng
```
