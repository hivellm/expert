"""
Progress Testing Module

Executes tests at each checkpoint save and generates JSON+Markdown reports.
Compares results with previous checkpoints and versions.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import PeftModel


class ProgressTestRunner:
    """Runs progress tests on a checkpoint model"""
    
    def __init__(self, expert_dir: Path, test_cases_path: Optional[Path] = None, output_dir: Path = None):
        self.expert_dir = Path(expert_dir)
        self.test_cases_path = test_cases_path or (self.expert_dir / "tests" / "test_cases.json")
        self.output_dir = Path(output_dir) if output_dir else (self.expert_dir / "weights" / "training_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_test_cases(self) -> Optional[List[Dict[str, Any]]]:
        """Load test cases from JSON file"""
        if not self.test_cases_path.exists():
            return None
        
        try:
            with open(self.test_cases_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("test_cases", [])
        except Exception as e:
            print(f"   [WARN] Failed to load test cases: {e}")
            return None
    
    def load_model_from_checkpoint(self, checkpoint_path: Path, base_model_path: str, device: str = "cuda"):
        """Load model and tokenizer from checkpoint"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Load adapter if checkpoint has adapter
            adapter_path = checkpoint_path / "adapter_model.safetensors"
            if adapter_path.exists() or (checkpoint_path / "adapter_config.json").exists():
                model = PeftModel.from_pretrained(model, str(checkpoint_path))
            
            model.eval()
            return model, tokenizer
        except Exception as e:
            print(f"   [ERROR] Failed to load model from checkpoint: {e}")
            return None, None
    
    def generate_output(self, model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
        """Generate output from model"""
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"   [ERROR] Generation failed: {e}")
            return ""
    
    def evaluate_test_case(self, model, tokenizer, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case"""
        prompt = test_case.get("prompt", test_case.get("input", ""))
        expected_keywords = test_case.get("expected_keywords", [])
        
        output = self.generate_output(model, tokenizer, prompt)
        
        # Check for expected keywords
        keywords_found = []
        for keyword in expected_keywords:
            if keyword.lower() in output.lower():
                keywords_found.append(keyword)
        
        # Simple scoring: 10.0 if all keywords found, otherwise proportional
        score = 10.0 if len(keywords_found) == len(expected_keywords) else (len(keywords_found) / len(expected_keywords) * 10.0) if expected_keywords else 5.0
        status = "passed" if score >= 8.0 else "failed"
        
        return {
            "id": test_case.get("id", "unknown"),
            "name": test_case.get("name", ""),
            "status": status,
            "score": score,
            "output": output[:500],  # Truncate for report
            "expected_keywords": expected_keywords,
            "keywords_found": keywords_found
        }
    
    def run_tests(self, checkpoint_path: Path, base_model_path: str, expert_name: str, expert_version: str, step: int) -> Optional[Dict[str, Any]]:
        """Run all tests on a checkpoint"""
        test_cases = self.load_test_cases()
        
        if not test_cases:
            print(f"   [INFO] No test cases found at {self.test_cases_path}, skipping progress tests")
            return None
        
        print(f"\n[PROGRESS TEST] Running tests on checkpoint-{step}...")
        print(f"   Test cases: {len(test_cases)}")
        
        # Load model
        model, tokenizer = self.load_model_from_checkpoint(checkpoint_path, base_model_path)
        if model is None or tokenizer is None:
            print(f"   [ERROR] Failed to load model, skipping tests")
            return None
        
        # Run tests
        results = []
        for test_case in test_cases:
            result = self.evaluate_test_case(model, tokenizer, test_case)
            results.append(result)
        
        # Calculate summary
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "passed")
        failed = total - passed
        success_rate = passed / total if total > 0 else 0.0
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "expert_name": expert_name,
            "expert_version": expert_version,
            "test_results": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": success_rate
            },
            "test_cases": results
        }
    
    def find_previous_checkpoints(self, current_step: int) -> List[Dict[str, Any]]:
        """Find previous checkpoint reports for comparison"""
        previous_reports = []
        
        # Look for reports in same directory
        for report_dir in self.output_dir.iterdir():
            if not report_dir.is_dir() or not report_dir.name.startswith("checkpoint-"):
                continue
            
            try:
                step_num = int(report_dir.name.split("-")[1])
                if step_num < current_step:
                    report_file = report_dir / "report.json"
                    if report_file.exists():
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                            previous_reports.append((step_num, report))
            except (ValueError, json.JSONDecodeError):
                continue
        
        # Sort by step number
        previous_reports.sort(key=lambda x: x[0])
        return [report for _, report in previous_reports]
    
    def compare_with_previous(self, current_report: Dict[str, Any], previous_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare current report with previous checkpoints"""
        if not previous_reports:
            return {
                "previous_checkpoint": None,
                "improvements": 0,
                "regressions": 0,
                "new_failures": [],
                "fixed_failures": []
            }
        
        # Compare with most recent previous checkpoint
        previous_report = previous_reports[-1]
        previous_checkpoint = previous_report["checkpoint"]
        
        current_results = {r["id"]: r for r in current_report["test_cases"]}
        previous_results = {r["id"]: r for r in previous_report["test_cases"]}
        
        improvements = []
        regressions = []
        new_failures = []
        fixed_failures = []
        
        for test_id, current_result in current_results.items():
            if test_id in previous_results:
                previous_result = previous_results[test_id]
                
                # Check if status changed
                if current_result["status"] == "passed" and previous_result["status"] == "failed":
                    fixed_failures.append(test_id)
                elif current_result["status"] == "failed" and previous_result["status"] == "passed":
                    regressions.append(test_id)
                
                # Check score improvement/regression
                score_diff = current_result["score"] - previous_result["score"]
                if score_diff > 1.0:
                    improvements.append(test_id)
                elif score_diff < -1.0:
                    regressions.append(test_id)
            else:
                # New test case
                if current_result["status"] == "failed":
                    new_failures.append(test_id)
        
        return {
            "previous_checkpoint": previous_checkpoint,
            "improvements": len(improvements),
            "regressions": len(regressions),
            "new_failures": new_failures,
            "fixed_failures": fixed_failures,
            "improved_tests": improvements[:10],  # Limit to 10 for report
            "regressed_tests": regressions[:10]
        }
    
    def generate_json_report(self, report: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON report"""
        report["comparison"] = comparison
        return report
    
    def generate_markdown_report(self, report: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        md = f"""# Progress Test Report: {report['checkpoint']}

**Expert**: {report['expert_name']} v{report['expert_version']}  
**Checkpoint**: {report['checkpoint']} (Step {report['step']})  
**Date**: {report['timestamp']}

## Summary

- **Total Tests**: {report['test_results']['total']}
- **Passed**: {report['test_results']['passed']}
- **Failed**: {report['test_results']['failed']}
- **Success Rate**: {report['test_results']['success_rate']*100:.1f}%

## Comparison with Previous Checkpoint

"""
        
        if comparison["previous_checkpoint"]:
            md += f"""**Previous Checkpoint**: {comparison['previous_checkpoint']}

- **Improvements**: {comparison['improvements']} tests
- **Regressions**: {comparison['regressions']} tests
- **Fixed Failures**: {len(comparison['fixed_failures'])} tests
- **New Failures**: {len(comparison['new_failures'])} tests

"""
            if comparison['fixed_failures']:
                md += f"### Fixed Failures\n\n"
                for test_id in comparison['fixed_failures'][:5]:
                    md += f"- {test_id}\n"
                md += "\n"
            
            if comparison['new_failures']:
                md += f"### New Failures\n\n"
                for test_id in comparison['new_failures'][:5]:
                    md += f"- {test_id}\n"
                md += "\n"
        else:
            md += "No previous checkpoint found for comparison.\n\n"
        
        md += """## Test Results

| ID | Name | Status | Score | Keywords Found |
|----|------|--------|-------|----------------|
"""
        
        for test_case in report['test_cases']:
            status_icon = "✅" if test_case['status'] == "passed" else "❌"
            keywords = ", ".join(test_case['keywords_found'][:3])
            if len(test_case['keywords_found']) > 3:
                keywords += "..."
            
            md += f"| {test_case['id']} | {test_case['name'][:30]} | {status_icon} {test_case['status']} | {test_case['score']:.1f}/10 | {keywords} |\n"
        
        return md
    
    def save_reports(self, checkpoint_name: str, json_report: Dict[str, Any], markdown_report: str):
        """Save JSON and Markdown reports"""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = checkpoint_dir / "report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # Save Markdown
        md_path = checkpoint_dir / "report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"   [OK] Reports saved to {checkpoint_dir}")


class ProgressTestCallback(TrainerCallback):
    """Callback that runs progress tests when checkpoints are saved"""
    
    def __init__(self, expert_dir: Path, base_model_path: str, expert_name: str, expert_version: str, test_cases_path: Optional[Path] = None):
        self.expert_dir = Path(expert_dir)
        self.base_model_path = base_model_path
        self.expert_name = expert_name
        self.expert_version = expert_version
        self.test_cases_path = test_cases_path
        self.runner = ProgressTestRunner(expert_dir, test_cases_path)
    
    def on_save(self, args, state, control, **kwargs):
        """Execute tests when checkpoint is saved"""
        checkpoint_name = f"checkpoint-{state.global_step}"
        checkpoint_path = Path(args.output_dir) / checkpoint_name
        
        if not checkpoint_path.exists():
            return
        
        # Run tests
        report = self.runner.run_tests(
            checkpoint_path,
            self.base_model_path,
            self.expert_name,
            self.expert_version,
            state.global_step
        )
        
        if report is None:
            return
        
        # Find previous checkpoints for comparison
        previous_reports = self.runner.find_previous_checkpoints(state.global_step)
        comparison = self.runner.compare_with_previous(report, previous_reports)
        
        # Generate reports
        json_report = self.runner.generate_json_report(report, comparison)
        markdown_report = self.runner.generate_markdown_report(report, comparison)
        
        # Save reports
        self.runner.save_reports(checkpoint_name, json_report, markdown_report)
        
        # Print summary
        print(f"\n[PROGRESS TEST] Summary:")
        print(f"   Success Rate: {report['test_results']['success_rate']*100:.1f}% ({report['test_results']['passed']}/{report['test_results']['total']})")
        if comparison["previous_checkpoint"]:
            print(f"   vs {comparison['previous_checkpoint']}: +{comparison['improvements']} improvements, -{comparison['regressions']} regressions")

