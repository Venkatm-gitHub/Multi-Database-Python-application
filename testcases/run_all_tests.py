#run_all_tests
import unittest
import os

def discover_and_run_tests(test_directory='.', pattern='test_*.py'):
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_directory, pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s).")
        exit(1)

if __name__ == "__main__":
    print("🔍 Discovering and running tests...")
    discover_and_run_tests()
