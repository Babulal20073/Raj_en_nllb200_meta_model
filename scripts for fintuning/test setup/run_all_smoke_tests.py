"""Runner to execute all or selected smoke tests across models, approaches, and languages."""
import subprocess
import sys
from pathlib import Path

test_dir = Path(__file__).resolve().parent

def main():
    smoke_files = sorted(test_dir.glob("smokeTest_*.py"))
    if not smoke_files:
        print("No smoke test scripts found.")
        return

    print("=" * 65)
    print(f" Found {len(smoke_files)} Smoke Tests in '{test_dir.name}'")
    print("=" * 65)

    passed = []
    skipped_or_failed = []

    for f in smoke_files:
        print(f"\n>>> Running {f.name} ...")
        res = subprocess.run([sys.executable, str(f)], capture_output=False)
        if res.returncode == 0:
            passed.append(f.name)
        else:
            skipped_or_failed.append(f.name)

    print("\n" + "=" * 65)
    print(" SMOKE TEST SUMMARY")
    print(f" Passed / Handled: {len(passed)} / {len(smoke_files)}")
    if skipped_or_failed:
        print(f" Failed: {skipped_or_failed}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

