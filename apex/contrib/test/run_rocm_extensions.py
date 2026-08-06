import unittest
import sys
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
test_dirs = ["groupbn", "fused_dense", "layer_norm", "multihead_attn", "transducer", "focal_loss", "index_mul_2d", "."] # "." for test_label_smoothing.py
ROCM_BLACKLIST = [
    "layer_norm"
]

runner = unittest.TextTestRunner(verbosity=2)

errcode = 0

for test_dir in test_dirs:
    if test_dir in ROCM_BLACKLIST:
        continue
    suite = unittest.TestLoader().discover(str(TEST_ROOT / test_dir))

    print("\nExecuting tests from " + test_dir)

    result = runner.run(suite)

    if not result.wasSuccessful():
        errcode = 1

sys.exit(errcode)
