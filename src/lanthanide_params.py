"""
==============================================================================
 BioArchitect Engine — Lanthanide Ion Parameter Database
==============================================================================
 Database tham số hóa lý cho 15 Lanthanide (Ln3+) và các ion cạnh tranh.
 Dùng trong LanRecov pipeline để tối ưu hóa chọn lọc protein theo từng ion.

 Nguồn dữ liệu:
   - Shannon (1976) ionic radii (CN=9, 3+ oxidation state)
   - Cotton & Wilkinson, Advanced Inorganic Chemistry
   - Cotruvo et al. (2018) Lanmodulin characterization
==============================================================================
"""

# ─── Lanthanide Database ────────────────────────────────────────────────────
# Mỗi entry chứa:
#   ionic_radius        : bán kính ion (Å) ở coordination number = 9
#   optimal_coord_dist  : khoảng cách tối ưu Ln3+ – O(carboxylate) (Å)
#   coordination_number : số phối trí ưu tiên
#   charge              : điện tích ion
#   electronegativity   : độ âm điện Pauling
#   hardness            : độ cứng HSAB (eV) – quan trọng cho selectivity
#   group               : light / middle / heavy REE
#   symbol              : ký hiệu hóa học

LANTHANIDE_DB = {
    "La": {
        "symbol": "La3+", "name": "Lanthanum",
        "ionic_radius": 1.216, "optimal_coord_dist": 2.55,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.10, "hardness": 15.4,
        "group": "light",
    },
    "Ce": {
        "symbol": "Ce3+", "name": "Cerium",
        "ionic_radius": 1.196, "optimal_coord_dist": 2.53,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.12, "hardness": 15.7,
        "group": "light",
    },
    "Pr": {
        "symbol": "Pr3+", "name": "Praseodymium",
        "ionic_radius": 1.179, "optimal_coord_dist": 2.51,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.13, "hardness": 16.0,
        "group": "light",
    },
    "Nd": {
        "symbol": "Nd3+", "name": "Neodymium",
        "ionic_radius": 1.163, "optimal_coord_dist": 2.49,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.14, "hardness": 16.3,
        "group": "light",
    },
    "Pm": {
        "symbol": "Pm3+", "name": "Promethium",
        "ionic_radius": 1.144, "optimal_coord_dist": 2.47,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.13, "hardness": 16.5,
        "group": "light",
    },
    "Sm": {
        "symbol": "Sm3+", "name": "Samarium",
        "ionic_radius": 1.132, "optimal_coord_dist": 2.46,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.17, "hardness": 16.8,
        "group": "light",
    },
    "Eu": {
        "symbol": "Eu3+", "name": "Europium",
        "ionic_radius": 1.120, "optimal_coord_dist": 2.45,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.20, "hardness": 17.0,
        "group": "middle",
    },
    "Gd": {
        "symbol": "Gd3+", "name": "Gadolinium",
        "ionic_radius": 1.107, "optimal_coord_dist": 2.44,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.20, "hardness": 17.2,
        "group": "middle",
    },
    "Tb": {
        "symbol": "Tb3+", "name": "Terbium",
        "ionic_radius": 1.095, "optimal_coord_dist": 2.42,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.10, "hardness": 17.5,
        "group": "middle",
    },
    "Dy": {
        "symbol": "Dy3+", "name": "Dysprosium",
        "ionic_radius": 1.083, "optimal_coord_dist": 2.41,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.22, "hardness": 17.7,
        "group": "heavy",
    },
    "Ho": {
        "symbol": "Ho3+", "name": "Holmium",
        "ionic_radius": 1.072, "optimal_coord_dist": 2.40,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.23, "hardness": 17.9,
        "group": "heavy",
    },
    "Er": {
        "symbol": "Er3+", "name": "Erbium",
        "ionic_radius": 1.062, "optimal_coord_dist": 2.39,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.24, "hardness": 18.1,
        "group": "heavy",
    },
    "Tm": {
        "symbol": "Tm3+", "name": "Thulium",
        "ionic_radius": 1.052, "optimal_coord_dist": 2.38,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.25, "hardness": 18.3,
        "group": "heavy",
    },
    "Yb": {
        "symbol": "Yb3+", "name": "Ytterbium",
        "ionic_radius": 1.042, "optimal_coord_dist": 2.37,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.10, "hardness": 18.5,
        "group": "heavy",
    },
    "Lu": {
        "symbol": "Lu3+", "name": "Lutetium",
        "ionic_radius": 1.032, "optimal_coord_dist": 2.36,
        "coordination_number": 9, "charge": 3,
        "electronegativity": 1.27, "hardness": 18.7,
        "group": "heavy",
    },
}

# ─── Competing Ions ─────────────────────────────────────────────────────────
# Các ion thường có trong nước thải mỏ AMD, cạnh tranh binding site với Ln3+.

COMPETING_IONS = {
    "Fe3": {
        "symbol": "Fe3+", "name": "Iron(III)",
        "ionic_radius": 0.645, "optimal_coord_dist": 2.01,
        "coordination_number": 6, "charge": 3,
        "electronegativity": 1.83, "hardness": 13.1,
    },
    "Ca2": {
        "symbol": "Ca2+", "name": "Calcium(II)",
        "ionic_radius": 1.180, "optimal_coord_dist": 2.40,
        "coordination_number": 8, "charge": 2,
        "electronegativity": 1.00, "hardness": 19.5,
    },
    "Al3": {
        "symbol": "Al3+", "name": "Aluminum(III)",
        "ionic_radius": 0.535, "optimal_coord_dist": 1.89,
        "coordination_number": 6, "charge": 3,
        "electronegativity": 1.61, "hardness": 45.8,
    },
    "Cu2": {
        "symbol": "Cu2+", "name": "Copper(II)",
        "ionic_radius": 0.730, "optimal_coord_dist": 2.00,
        "coordination_number": 6, "charge": 2,
        "electronegativity": 1.90, "hardness": 8.3,
    },
    "Pb2": {
        "symbol": "Pb2+", "name": "Lead(II)",
        "ionic_radius": 1.290, "optimal_coord_dist": 2.63,
        "coordination_number": 8, "charge": 2,
        "electronegativity": 2.33, "hardness": 8.5,
    },
}

# ─── EF-Hand Binding Residue Preferences ────────────────────────────────────
# Amino acid ưu tiên trong vùng binding site theo nhóm Lanthanide.
# Dựa trên phân tích PDB structures của EF-hand proteins.

EF_HAND_BINDING_RESIDUES = {
    "light": {
        # Light REEs (La, Ce, Pr, Nd, Pm, Sm) — bán kính lớn, ưu tiên CN=9
        # Cần nhiều Asp/Glu (bidentate) để tạo cavity lớn
        "preferred": ["D", "E", "N"],       # Asp, Glu, Asn
        "allowed":   ["Q", "S", "T"],       # Gln, Ser, Thr
        "weights":   [0.35, 0.30, 0.15, 0.08, 0.07, 0.05],
    },
    "middle": {
        # Middle REEs (Eu, Gd, Tb) — cân bằng giữa bidentate và monodentate
        "preferred": ["D", "E", "N"],
        "allowed":   ["S", "T", "Q"],
        "weights":   [0.30, 0.28, 0.18, 0.10, 0.08, 0.06],
    },
    "heavy": {
        # Heavy REEs (Dy, Ho, Er, Tm, Yb, Lu) — bán kính nhỏ, CN=8
        # Ưu tiên monodentate ligands, cavity nhỏ và chặt hơn
        "preferred": ["D", "N", "S"],       # Asp, Asn, Ser
        "allowed":   ["E", "T", "Q"],       # Glu, Thr, Gln
        "weights":   [0.30, 0.22, 0.18, 0.12, 0.10, 0.08],
    },
}

# ─── pH Stability Parameters ────────────────────────────────────────────────
# pKa values for amino acid side chains relevant to metal binding

RESIDUE_PKA = {
    "D": 3.65,   # Aspartate (β-carboxyl)
    "E": 4.25,   # Glutamate (γ-carboxyl)
    "H": 6.00,   # Histidine (imidazole)
    "C": 8.18,   # Cysteine (thiol)
    "Y": 10.07,  # Tyrosine (phenol)
    "K": 10.53,  # Lysine (ε-amino)
    "R": 12.48,  # Arginine (guanidinium)
}

# At pH < 3.0, Asp and Glu will be protonated → lose metal binding ability
# This is critical for LanRecov design
ACID_RESISTANT_RESIDUES = ["N", "Q", "S", "T"]  # Not affected by pH < 3.0
ACID_SENSITIVE_RESIDUES = ["D", "E", "H"]        # Protonated below their pKa


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_ion_params(ion_name):
    """
    Get parameters for any ion (Lanthanide or competing).
    
    Args:
        ion_name (str): e.g. "La", "Nd", "Fe3", "Ca2"
    
    Returns:
        dict: Ion parameters. Raises KeyError if not found.
    """
    if ion_name in LANTHANIDE_DB:
        return LANTHANIDE_DB[ion_name]
    elif ion_name in COMPETING_IONS:
        return COMPETING_IONS[ion_name]
    else:
        raise KeyError(f"Unknown ion: '{ion_name}'. "
                       f"Available: {list(LANTHANIDE_DB.keys()) + list(COMPETING_IONS.keys())}")


def get_optimal_distance(ion_name):
    """
    Get optimal coordination distance (Å) for an ion.
    
    Args:
        ion_name (str): e.g. "La", "Dy", "Fe3"
    
    Returns:
        float: Optimal distance in Angstroms.
    """
    params = get_ion_params(ion_name)
    return params["optimal_coord_dist"]


def get_distance_range(ion_name, tolerance=0.15):
    """
    Get acceptable coordination distance range for an ion.
    
    Args:
        ion_name (str): Ion identifier.
        tolerance (float): ± tolerance in Angstroms. Default ±0.15 Å.
    
    Returns:
        tuple: (min_dist, max_dist) in Angstroms.
    """
    optimal = get_optimal_distance(ion_name)
    return (optimal - tolerance, optimal + tolerance)


def get_binding_residue_bias(ion_name):
    """
    Get preferred binding residues and their weights for a given ion.
    
    Args:
        ion_name (str): Lanthanide identifier (e.g. "La", "Dy").
    
    Returns:
        dict: {"residues": list[str], "weights": list[float]}
    """
    if ion_name not in LANTHANIDE_DB:
        raise KeyError(f"'{ion_name}' is not a Lanthanide. Available: {list(LANTHANIDE_DB.keys())}")
    
    group = LANTHANIDE_DB[ion_name]["group"]
    prefs = EF_HAND_BINDING_RESIDUES[group]
    residues = prefs["preferred"] + prefs["allowed"]
    weights = prefs["weights"]
    
    return {"residues": residues, "weights": weights}


def estimate_binding_energy(distance, ion_name, coordinating_residue="D"):
    """
    Estimate binding energy using a simplified Lennard-Jones-like potential
    adapted for metal-ligand interactions.
    
    E(r) = ε * [(r₀/r)^12 - 2*(r₀/r)^6]
    
    Where:
        r₀ = optimal coordination distance
        ε  = well depth (proportional to charge²/radius)
    
    Args:
        distance (float): Actual distance in Angstroms.
        ion_name (str): Ion identifier.
        coordinating_residue (str): One-letter amino acid code.
    
    Returns:
        float: Estimated binding energy (arbitrary units, more negative = stronger).
    """
    params = get_ion_params(ion_name)
    r0 = params["optimal_coord_dist"]
    charge = params["charge"]
    radius = params["ionic_radius"]
    
    # Well depth: stronger for higher charge and smaller radius
    epsilon = (charge ** 2) / radius
    
    # Residue-specific scaling
    residue_affinity = {
        "D": 1.0,   # Aspartate: bidentate carboxylate, strongest
        "E": 0.95,  # Glutamate: bidentate but more flexible
        "N": 0.70,  # Asparagine: amide oxygen, moderate
        "Q": 0.65,  # Glutamine: amide oxygen
        "S": 0.50,  # Serine: hydroxyl
        "T": 0.50,  # Threonine: hydroxyl
    }
    affinity_scale = residue_affinity.get(coordinating_residue, 0.3)
    epsilon *= affinity_scale
    
    # Lennard-Jones potential
    if distance <= 0:
        return float('inf')  # Physically impossible
    
    ratio = r0 / distance
    energy = epsilon * (ratio**12 - 2 * ratio**6)
    
    return energy


def calculate_selectivity_score(distance, target_ion, competing_ions=None):
    """
    Calculate selectivity score: how much better does the binding pocket
    fit the target ion compared to competing ions.
    
    Score > 1.0 means selective for target. Higher is better.
    
    Args:
        distance (float): Measured pocket distance in Angstroms.
        target_ion (str): Target Lanthanide (e.g. "La").
        competing_ions (list[str]): List of competing ion names.
    
    Returns:
        dict: {
            "selectivity_score": float,
            "target_energy": float,
            "competitor_energies": dict,
        }
    """
    if competing_ions is None:
        competing_ions = ["Fe3", "Ca2", "Al3"]
    
    target_energy = estimate_binding_energy(distance, target_ion)
    
    competitor_energies = {}
    for comp in competing_ions:
        try:
            competitor_energies[comp] = estimate_binding_energy(distance, comp)
        except KeyError:
            continue
    
    # Selectivity: ratio of target binding vs best competitor
    if competitor_energies:
        best_competitor_energy = min(competitor_energies.values())
        # More negative energy = stronger binding
        # We want target to be much more negative than competitors
        if best_competitor_energy < 0 and target_energy < 0:
            selectivity = target_energy / best_competitor_energy
        elif target_energy < 0:
            selectivity = 10.0  # Target binds, competitors don't
        else:
            selectivity = 0.1  # Target doesn't bind well
    else:
        selectivity = 1.0
    
    return {
        "selectivity_score": selectivity,
        "target_energy": target_energy,
        "competitor_energies": competitor_energies,
    }
