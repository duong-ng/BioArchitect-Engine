"""Quick verification script for all BioArchitect Engine modules."""
import py_compile
import sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = [
    "lanthanide_params.py",
    "step1_mutation_generator.py",
    "step2_af2_folding_evaluator.py",
    "step3_genetic_optimizer.py",
    "step4_md_validation.py",
    "main_optimizer.py",
]

print("=" * 50)
print("  SYNTAX CHECK")
print("=" * 50)
all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  OK  {f}")
    except py_compile.PyCompileError as e:
        print(f"  FAIL {f}: {e}")
        all_ok = False

print()
print("=" * 50)
print("  IMPORT TEST")
print("=" * 50)

# Test imports that don't require GPU/heavy libs
try:
    from lanthanide_params import LANTHANIDE_DB, get_ion_params, get_distance_range
    print(f"  OK  lanthanide_params ({len(LANTHANIDE_DB)} ions)")
except Exception as e:
    print(f"  FAIL lanthanide_params: {e}")
    all_ok = False

try:
    from step3_genetic_optimizer import BioArchitectGA
    print("  OK  step3_genetic_optimizer")
except Exception as e:
    print(f"  FAIL step3_genetic_optimizer: {e}")
    all_ok = False

try:
    from step2_af2_folding_evaluator import AlphaFold2Evaluator
    print("  OK  step2_af2_folding_evaluator")
except Exception as e:
    print(f"  FAIL step2_af2_folding_evaluator: {e}")
    all_ok = False

# Test GA cliff penalty
print()
print("=" * 50)
print("  GA CLIFF PENALTY TEST")
print("=" * 50)
try:
    import numpy as np
    ga = BioArchitectGA("La")
    
    # Test 1: Large distance (12.7 A) should get near-zero fitness
    coords_bad = np.zeros((120, 3))
    coords_bad[92] = [12.7, 0, 0]
    coords_bad[100] = [0, 0, 0]
    f_bad, d_bad, b_bad = ga.calculate_fitness(coords_bad, 92, 100)
    print(f"  r=12.7A: fitness={f_bad:.8f} (should be ~0)")
    
    # Test 2: Good distance (~2.5 A) should score high
    coords_good = np.zeros((120, 3))
    coords_good[92] = [2.5, 0, 0]
    coords_good[100] = [0, 0, 0]
    f_good, d_good, b_good = ga.calculate_fitness(coords_good, 92, 100)
    print(f"  r=2.5A:  fitness={f_good:.4f} (should be >0.3)")
    
    # Test 3: Borderline (4.0 A) should be penalized but not zero
    coords_mid = np.zeros((120, 3))
    coords_mid[92] = [4.0, 0, 0]
    coords_mid[100] = [0, 0, 0]
    f_mid, d_mid, b_mid = ga.calculate_fitness(coords_mid, 92, 100)
    print(f"  r=4.0A:  fitness={f_mid:.6f} (should be small)")
    
    # Verify ordering: good > mid > bad
    assert f_good > f_mid > f_bad, "Fitness ordering broken!"
    assert f_bad < 0.05, f"12.7A penalty too weak: {f_bad}"
    print("  ALL PENALTY TESTS PASSED")
    
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    all_ok = False

print()
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
