"""Quick verification for step2_esmfold_evaluator.py"""
import py_compile
import sys

print("=" * 50)
print("  ESMFOLD MODULE VERIFICATION")
print("=" * 50)

# 1. Syntax check
try:
    py_compile.compile("step2_esmfold_evaluator.py", doraise=True)
    print("  OK  Syntax check passed")
except py_compile.PyCompileError as e:
    print(f"  FAIL Syntax error: {e}")
    sys.exit(1)

# 2. Import check
try:
    from step2_esmfold_evaluator import ESMFoldEvaluator
    print("  OK  Import successful")
except Exception as e:
    print(f"  FAIL Import error: {e}")
    sys.exit(1)

# 3. API contract check
api_methods = [
    "predict_structure",
    "predict_full",
    "get_confidence_scores",
    "write_pdb",
    "get_residue_distance",
    "_apply_pdbfixer",
    "_cleanup_memory",
]

for method in api_methods:
    if hasattr(ESMFoldEvaluator, method):
        print(f"  OK  Method: {method}")
    else:
        print(f"  FAIL Missing method: {method}")
        sys.exit(1)

# 4. Check init signature
import inspect
sig = inspect.signature(ESMFoldEvaluator.__init__)
params = list(sig.parameters.keys())
expected = ["self", "device", "ef_hands", "fix_pdb", "fix_pH"]
if params == expected:
    print(f"  OK  __init__ signature: {params}")
else:
    print(f"  WARN __init__ params: {params} (expected {expected})")

# 5. No AF2-specific args
af2_args = ["num_recycles", "use_msa", "msa_mode", "max_seq", "model_type", "chunk_size"]
for arg in af2_args:
    if arg in params:
        print(f"  FAIL AF2 arg still present: {arg}")
        sys.exit(1)
print(f"  OK  No AF2-specific args in __init__")

print()
print("ALL ESMFOLD CHECKS PASSED")
