# fly_brain/simulations/run_all.py
"""
Run all fly_brain simulations and tests.
Produces a summary report of all results.

Usage:
    cd C:\\Users\\student
    python -m fly_brain.simulations.run_all
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
REPORT_FILE  = os.path.join(RESULTS_DIR, 'validation_report.txt')

TESTS = [
    ('Test 1: Generalisation',          'test_01_generalisation'),
    ('Test 2: Catastrophic Forgetting',  'test_02_catastrophic_forgetting'),
    ('Test 3: Sensor Dropout',           'test_03_sensor_dropout'),
    ('Test 4: Learning Curve',           'test_04_learning_curve'),
    ('Test 5: Wave Visualisation',       'test_05_wave_visualisation'),
    ('Test 6: Thermodynamic Profiles',   'test_06_thermodynamic_profiles'),
    ('Test 7: Baseline Comparison',      'test_07_baseline_comparison'),
    ('Test 8: Memory Persistence',       'test_08_memory_persistence'),
    ('Test 9: Scaling',                  'test_09_scaling'),
]


def _import_test(module_name: str):
    """Import a simulation test module. Returns module or None on failure."""
    from fly_brain.simulations import (
        test_01_generalisation,
        test_02_catastrophic_forgetting,
        test_03_sensor_dropout,
        test_04_learning_curve,
        test_05_wave_visualisation,
        test_06_thermodynamic_profiles,
        test_07_baseline_comparison,
        test_08_memory_persistence,
        test_09_scaling,
    )
    _map = {
        'test_01_generalisation':          test_01_generalisation,
        'test_02_catastrophic_forgetting': test_02_catastrophic_forgetting,
        'test_03_sensor_dropout':          test_03_sensor_dropout,
        'test_04_learning_curve':          test_04_learning_curve,
        'test_05_wave_visualisation':      test_05_wave_visualisation,
        'test_06_thermodynamic_profiles':  test_06_thermodynamic_profiles,
        'test_07_baseline_comparison':     test_07_baseline_comparison,
        'test_08_memory_persistence':      test_08_memory_persistence,
        'test_09_scaling':                 test_09_scaling,
    }
    return _map.get(module_name)


def main():
    """Run all validation tests and print/save summary report."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "═" * 65)
    print("  fly_brain Extended Validation Suite")
    print("  Wave-Based Drone Navigation Controller")
    print("═" * 65)
    print(f"  Running {len(TESTS)} tests...\n")

    results        = []
    total_start    = time.perf_counter()
    n_pass_fail    = 0   # tests with a definite pass/fail
    n_passed       = 0

    for label, module_name in TESTS:
        print(f"\n{'▶':>3} {label}")
        t0  = time.perf_counter()
        mod = _import_test(module_name)

        if mod is None:
            result = {
                'label':    label,
                'passed':   False,
                'elapsed':  0.0,
                'metric':   'import error',
                'error':    'Module not found',
                'info_only': False,
            }
        else:
            try:
                ret = mod.main()
                elapsed = time.perf_counter() - t0
                is_info = ret.get('is_info_only', False)
                result = {
                    'label':    label,
                    'passed':   ret.get('passed', True),
                    'elapsed':  elapsed,
                    'metric':   ret.get('metric_label', ''),
                    'error':    None,
                    'info_only': is_info,
                }
                if not is_info:
                    n_pass_fail += 1
                    if ret.get('passed', False):
                        n_passed += 1
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"\n  *** EXCEPTION in {label} ***")
                traceback.print_exc()
                result = {
                    'label':    label,
                    'passed':   False,
                    'elapsed':  elapsed,
                    'metric':   'EXCEPTION',
                    'error':    str(e),
                    'info_only': False,
                }
                n_pass_fail += 1

        results.append(result)

    total_elapsed = time.perf_counter() - total_start

    # ---- Summary table ----
    report_lines = []
    sep_line = "═" * 65

    def _add(line: str):
        report_lines.append(line)
        print(line)

    _add("\n" + sep_line)
    _add("  fly_brain Extended Validation Results")
    _add(sep_line)

    for r in results:
        if r['error']:
            status = "ERROR"
        elif r['info_only']:
            status = "INFO "
        else:
            status = "PASS " if r['passed'] else "FAIL "

        metric_str = r['metric'][:25] if r['metric'] else ""
        _add(f"  {r['label']:<35} {status}  {metric_str:<25}  ({r['elapsed']:.1f}s)")

    _add(sep_line)
    _add(f"  Pass/fail tests: {n_passed}/{n_pass_fail} passed")
    _add(f"  Total time:      {total_elapsed:.1f}s")
    _add(sep_line)

    # ---- Save report ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Full report saved to: {REPORT_FILE}")

    return results


if __name__ == '__main__':
    main()
