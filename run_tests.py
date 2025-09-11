#!/usr/bin/env python3
"""
Comprehensive Test Runner for Smart Traffic Management System
Runs all test suites with detailed reporting and coverage analysis
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

class TestRunner:
    """Comprehensive test runner for the Smart Traffic Management System"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_suites": {},
            "coverage": {},
            "performance": {},
            "summary": {}
        }
        self.verbose = False
        
    def run_all_tests(self, verbose: bool = False, coverage: bool = True, 
                     performance: bool = True, parallel: bool = False):
        """Run all test suites"""
        self.verbose = verbose
        
        print("🚀 Starting Smart Traffic Management System Test Suite")
        print("=" * 60)
        
        # Run unit tests
        self._run_unit_tests(coverage, parallel)
        
        # Run integration tests
        self._run_integration_tests(coverage, parallel)
        
        # Run end-to-end tests
        self._run_e2e_tests(coverage, parallel)
        
        # Run performance tests
        if performance:
            self._run_performance_tests(parallel)
        
        # Run security tests
        self._run_security_tests()
        
        # Generate coverage report
        if coverage:
            self._generate_coverage_report()
        
        # Generate test report
        self._generate_test_report()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _run_unit_tests(self, coverage: bool = True, parallel: bool = False):
        """Run unit tests"""
        print("\n📋 Running Unit Tests...")
        print("-" * 40)
        
        start_time = time.time()
        
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v"]
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=json"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if self.verbose:
            cmd.append("--tb=long")
        else:
            cmd.append("--tb=short")
        
        result = self._run_command(cmd, "unit_tests")
        
        self.results["test_suites"]["unit_tests"] = {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": time.time() - start_time,
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", {})
        }
        
        if result["returncode"] == 0:
            print("✅ Unit tests passed")
        else:
            print("❌ Unit tests failed")
            if self.verbose:
                print(result["stderr"])
    
    def _run_integration_tests(self, coverage: bool = True, parallel: bool = False):
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        print("-" * 40)
        
        start_time = time.time()
        
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "-m", "integration"]
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=json"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if self.verbose:
            cmd.append("--tb=long")
        else:
            cmd.append("--tb=short")
        
        result = self._run_command(cmd, "integration_tests")
        
        self.results["test_suites"]["integration_tests"] = {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": time.time() - start_time,
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", {})
        }
        
        if result["returncode"] == 0:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            if self.verbose:
                print(result["stderr"])
    
    def _run_e2e_tests(self, coverage: bool = True, parallel: bool = False):
        """Run end-to-end tests"""
        print("\n🌐 Running End-to-End Tests...")
        print("-" * 40)
        
        start_time = time.time()
        
        cmd = ["python", "-m", "pytest", "tests/e2e/", "-v", "-m", "e2e"]
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=json"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if self.verbose:
            cmd.append("--tb=long")
        else:
            cmd.append("--tb=short")
        
        result = self._run_command(cmd, "e2e_tests")
        
        self.results["test_suites"]["e2e_tests"] = {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": time.time() - start_time,
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", {})
        }
        
        if result["returncode"] == 0:
            print("✅ End-to-end tests passed")
        else:
            print("❌ End-to-end tests failed")
            if self.verbose:
                print(result["stderr"])
    
    def _run_performance_tests(self, parallel: bool = False):
        """Run performance tests"""
        print("\n⚡ Running Performance Tests...")
        print("-" * 40)
        
        start_time = time.time()
        
        cmd = ["python", "-m", "pytest", "tests/performance/", "-v", "-m", "performance"]
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if self.verbose:
            cmd.append("--tb=long")
        else:
            cmd.append("--tb=short")
        
        result = self._run_command(cmd, "performance_tests")
        
        self.results["test_suites"]["performance_tests"] = {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": time.time() - start_time,
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "performance_metrics": result.get("performance_metrics", {})
        }
        
        if result["returncode"] == 0:
            print("✅ Performance tests passed")
        else:
            print("❌ Performance tests failed")
            if self.verbose:
                print(result["stderr"])
    
    def _run_security_tests(self):
        """Run security tests"""
        print("\n🔒 Running Security Tests...")
        print("-" * 40)
        
        start_time = time.time()
        
        # Run security scanning tools
        security_tools = [
            "bandit",  # Python security linter
            "safety",  # Dependency vulnerability scanner
            "semgrep",  # Static analysis tool
        ]
        
        security_results = {}
        
        for tool in security_tools:
            try:
                if tool == "bandit":
                    cmd = ["bandit", "-r", "src/", "-f", "json"]
                elif tool == "safety":
                    cmd = ["safety", "check", "--json"]
                elif tool == "semgrep":
                    cmd = ["semgrep", "--config=auto", "src/", "--json"]
                
                result = self._run_command(cmd, f"security_{tool}")
                security_results[tool] = {
                    "status": "passed" if result["returncode"] == 0 else "failed",
                    "output": result["stdout"],
                    "errors": result["stderr"]
                }
                
            except FileNotFoundError:
                print(f"⚠️  {tool} not found, skipping...")
                security_results[tool] = {
                    "status": "skipped",
                    "output": "",
                    "errors": f"{tool} not installed"
                }
        
        self.results["test_suites"]["security_tests"] = {
            "status": "completed",
            "duration": time.time() - start_time,
            "tools": security_results
        }
        
        print("✅ Security tests completed")
    
    def _run_command(self, cmd: List[str], test_type: str) -> Dict[str, Any]:
        """Run a command and return results"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse pytest output for test counts
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            
            if "pytest" in cmd[1]:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Parse line like "5 passed, 2 failed in 1.23s"
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                if "passed" in line:
                                    tests_passed = int(part)
                                elif "failed" in line:
                                    tests_failed = int(part)
                        tests_run = tests_passed + tests_failed
                        break
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed
            }
            
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
    
    def _generate_coverage_report(self):
        """Generate coverage report"""
        print("\n📊 Generating Coverage Report...")
        print("-" * 40)
        
        try:
            # Generate HTML coverage report
            cmd = ["python", "-m", "coverage", "html", "-d", "htmlcov"]
            result = self._run_command(cmd, "coverage_html")
            
            # Generate JSON coverage report
            cmd = ["python", "-m", "coverage", "json", "-o", "coverage.json"]
            result = self._run_command(cmd, "coverage_json")
            
            # Read coverage data
            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                
                self.results["coverage"] = {
                    "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                    "lines_total": coverage_data.get("totals", {}).get("num_statements", 0),
                    "files": len(coverage_data.get("files", {}))
                }
            
            print("✅ Coverage report generated")
            
        except Exception as e:
            print(f"❌ Error generating coverage report: {e}")
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📝 Generating Test Report...")
        print("-" * 40)
        
        # Calculate summary statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_duration = 0
        
        for suite_name, suite_data in self.results["test_suites"].items():
            if "tests_run" in suite_data:
                total_tests += suite_data["tests_run"]
                total_passed += suite_data["tests_passed"]
                total_failed += suite_data["tests_failed"]
                total_duration += suite_data["duration"]
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_duration": total_duration,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "end_time": datetime.now().isoformat()
        }
        
        # Save report to file
        report_file = "test_report.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Test report saved to {report_file}")
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        print("\nTest Suites:")
        for suite_name, suite_data in self.results["test_suites"].items():
            status_icon = "✅" if suite_data.get("status") == "passed" else "❌"
            duration = suite_data.get("duration", 0)
            tests_run = suite_data.get("tests_run", 0)
            print(f"  {status_icon} {suite_name}: {tests_run} tests in {duration:.2f}s")
        
        if "coverage" in self.results:
            coverage = self.results["coverage"]
            print(f"\nCoverage: {coverage['total_coverage']:.1f}%")
            print(f"Lines Covered: {coverage['lines_covered']}/{coverage['lines_total']}")
            print(f"Files: {coverage['files']}")
        
        print("\n" + "=" * 60)
        
        if summary["total_failed"] == 0:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {summary['total_failed']} tests failed")
            sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Traffic Management System Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage analysis")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e-only", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.unit_only:
        runner._run_unit_tests(not args.no_coverage, args.parallel)
    elif args.integration_only:
        runner._run_integration_tests(not args.no_coverage, args.parallel)
    elif args.e2e_only:
        runner._run_e2e_tests(not args.no_coverage, args.parallel)
    elif args.performance_only:
        runner._run_performance_tests(args.parallel)
    else:
        runner.run_all_tests(
            verbose=args.verbose,
            coverage=not args.no_coverage,
            performance=not args.no_performance,
            parallel=args.parallel
        )

if __name__ == "__main__":
    main()
