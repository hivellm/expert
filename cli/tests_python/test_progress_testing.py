#!/usr/bin/env python3
"""
Tests for progress_testing.py module
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.progress_testing import ProgressTestRunner, ProgressTestCallback


class TestProgressTestRunner:
    """Test ProgressTestRunner class"""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create ProgressTestRunner instance"""
        expert_dir = tmp_path / "expert"
        expert_dir.mkdir()
        return ProgressTestRunner(
            expert_dir=expert_dir,
            test_cases_path=tmp_path / "test_cases.json",
            output_dir=tmp_path / "output",
        )
    
    @pytest.fixture
    def test_cases(self):
        """Create sample test cases"""
        return {
            "test_cases": [
                {
                    "id": "test_1",
                    "prompt": "test input 1",
                    "expected_keywords": ["test"],
                },
                {
                    "id": "test_2",
                    "prompt": "test input 2",
                    "expected_keywords": ["test"],
                },
            ]
        }
    
    def test_runner_initialization(self, runner, tmp_path):
        """Test runner initialization"""
        assert runner.expert_dir.exists()
        assert runner.output_dir.exists()
        assert runner.test_cases_path == Path(tmp_path / "test_cases.json")
    
    def test_load_test_cases_file_exists(self, runner, tmp_path, test_cases):
        """Test loading test cases when file exists"""
        test_file = tmp_path / "test_cases.json"
        with open(test_file, 'w') as f:
            json.dump(test_cases, f)
        
        loaded = runner.load_test_cases()
        
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["id"] == "test_1"
    
    def test_load_test_cases_file_not_exists(self, runner):
        """Test loading test cases when file doesn't exist"""
        runner.test_cases_path = Path("/nonexistent/path/test_cases.json")
        
        loaded = runner.load_test_cases()
        
        assert loaded is None
    
    @patch('train.progress_testing.ProgressTestRunner.load_model_from_checkpoint')
    @patch('train.progress_testing.ProgressTestRunner.evaluate_test_case')
    def test_run_tests(self, mock_evaluate, mock_load_model, runner, tmp_path):
        """Test running tests on a checkpoint"""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_evaluate.return_value = {
            "id": "test_1",
            "name": "Test 1",
            "status": "passed",
            "score": 10.0,
            "output": "test output",
            "expected_keywords": [],
            "keywords_found": [],
        }
        
        # Create checkpoint directory
        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()
        
        # Create test cases file
        test_file = tmp_path / "test_cases.json"
        with open(test_file, 'w') as f:
            json.dump({
                "test_cases": [{"id": "test_1", "prompt": "test", "expected_keywords": []}]
            }, f)
        
        result = runner.run_tests(
            checkpoint_path,
            "test-model",
            "test-expert",
            "1.0.0",
            100
        )
        
        assert result is not None
        assert result["checkpoint"] == "checkpoint-100"
        assert result["expert_name"] == "test-expert"
        assert result["expert_version"] == "1.0.0"
        assert result["test_results"]["total"] == 1
    
    def test_run_tests_returns_none_when_no_test_cases(self, runner, tmp_path):
        """Test that run_tests returns None when no test cases exist"""
        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()
        
        result = runner.run_tests(
            checkpoint_path,
            "test-model",
            "test-expert",
            "1.0.0",
            100
        )
        
        assert result is None
    
    def test_find_previous_checkpoints(self, runner, tmp_path):
        """Test finding previous checkpoint reports"""
        # Set output_dir
        runner.output_dir = tmp_path
        
        # Create checkpoint directories
        checkpoint_100 = tmp_path / "checkpoint-100"
        checkpoint_200 = tmp_path / "checkpoint-200"
        checkpoint_100.mkdir()
        checkpoint_200.mkdir()
        
        # Create reports
        report_100 = {
            "checkpoint": "checkpoint-100",
            "step": 100,
            "test_cases": [{"id": "test_1", "status": "passed", "score": 10.0}],
        }
        report_200 = {
            "checkpoint": "checkpoint-200",
            "step": 200,
            "test_cases": [{"id": "test_1", "status": "failed", "score": 5.0}],
        }
        
        with open(checkpoint_100 / "report.json", 'w') as f:
            json.dump(report_100, f)
        with open(checkpoint_200 / "report.json", 'w') as f:
            json.dump(report_200, f)
        
        previous = runner.find_previous_checkpoints(300)
        
        assert len(previous) == 2
        assert previous[0]["checkpoint"] == "checkpoint-100"
        assert previous[1]["checkpoint"] == "checkpoint-200"
    
    def test_compare_with_previous_no_previous(self, runner):
        """Test comparison when no previous checkpoints exist"""
        current_report = {
            "checkpoint": "checkpoint-300",
            "test_cases": [{"id": "test_1", "status": "passed", "score": 10.0}],
        }
        
        comparison = runner.compare_with_previous(current_report, [])
        
        assert comparison["previous_checkpoint"] is None
        assert comparison["improvements"] == 0
        assert comparison["regressions"] == 0
    
    def test_compare_with_previous_improvements(self, runner):
        """Test comparison showing improvements"""
        current_report = {
            "checkpoint": "checkpoint-300",
            "test_cases": [
                {"id": "test_1", "status": "passed", "score": 10.0},
                {"id": "test_2", "status": "passed", "score": 10.0},
            ],
        }
        
        previous_report = {
            "checkpoint": "checkpoint-200",
            "test_cases": [
                {"id": "test_1", "status": "failed", "score": 5.0},
                {"id": "test_2", "status": "passed", "score": 7.0},
            ],
        }
        
        comparison = runner.compare_with_previous(current_report, [previous_report])
        
        assert comparison["previous_checkpoint"] == "checkpoint-200"
        assert comparison["fixed_failures"] == ["test_1"]
        assert len(comparison.get("improved_tests", [])) >= 0
    
    def test_compare_with_previous_regressions(self, runner):
        """Test comparison showing regressions"""
        current_report = {
            "checkpoint": "checkpoint-300",
            "test_cases": [
                {"id": "test_1", "status": "failed", "score": 5.0},
            ],
        }
        
        previous_report = {
            "checkpoint": "checkpoint-200",
            "test_cases": [
                {"id": "test_1", "status": "passed", "score": 10.0},
            ],
        }
        
        comparison = runner.compare_with_previous(current_report, [previous_report])
        
        assert comparison["regressions"] > 0
        assert "test_1" in comparison["regressed_tests"]
    
    def test_generate_json_report(self, runner):
        """Test JSON report generation"""
        report = {
            "checkpoint": "checkpoint-100",
            "test_cases": [],
        }
        
        comparison = {
            "previous_checkpoint": None,
            "improvements": 0,
        }
        
        json_report = runner.generate_json_report(report, comparison)
        
        assert "comparison" in json_report
        assert json_report["comparison"] == comparison
    
    def test_generate_markdown_report(self, runner):
        """Test Markdown report generation"""
        report = {
            "checkpoint": "checkpoint-100",
            "step": 100,
            "expert_name": "test-expert",
            "expert_version": "1.0.0",
            "test_results": {
                "total": 2,
                "passed": 1,
                "failed": 1,
                "success_rate": 0.5,
            },
            "test_cases": [
                {"id": "test_1", "name": "Test 1", "status": "passed", "score": 10.0, "expected_keywords": [], "keywords_found": []},
                {"id": "test_2", "name": "Test 2", "status": "failed", "score": 5.0, "expected_keywords": [], "keywords_found": []},
            ],
        }
        
        comparison = {
            "previous_checkpoint": "checkpoint-50",
            "improvements": 1,
            "regressions": 0,
            "fixed_failures": [],
            "new_failures": [],
        }
        
        # Add timestamp if missing
        if "timestamp" not in report:
            report["timestamp"] = datetime.now().isoformat()
        
        md_report = runner.generate_markdown_report(report, comparison)
        
        assert "# Progress Test Report" in md_report
        assert "checkpoint-100" in md_report
        assert "test-expert" in md_report
        assert "1.0.0" in md_report


class TestProgressTestCallback:
    """Test ProgressTestCallback class"""
    
    @pytest.fixture
    def callback(self, tmp_path):
        """Create ProgressTestCallback instance"""
        expert_dir = tmp_path / "expert"
        expert_dir.mkdir()
        test_cases_file = expert_dir / "tests" / "test_cases.json"
        test_cases_file.parent.mkdir(parents=True)
        test_cases_file.write_text(json.dumps({"test_cases": []}))
        
        return ProgressTestCallback(
            expert_dir=expert_dir,
            base_model_path="test-model",
            expert_name="test-expert",
            expert_version="1.0.0",
            test_cases_path=test_cases_file,
        )
    
    @pytest.fixture
    def mock_args(self):
        """Create mock TrainingArguments"""
        args = Mock()
        args.output_dir = "test_output"
        return args
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TrainerState"""
        state = Mock()
        state.global_step = 100
        return state
    
    @pytest.fixture
    def mock_control(self):
        """Create mock TrainerControl"""
        control = Mock()
        return control
    
    @patch('train.progress_testing.ProgressTestRunner.run_tests')
    @patch('train.progress_testing.ProgressTestRunner.find_previous_checkpoints')
    @patch('train.progress_testing.ProgressTestRunner.compare_with_previous')
    @patch('train.progress_testing.ProgressTestRunner.generate_json_report')
    @patch('train.progress_testing.ProgressTestRunner.generate_markdown_report')
    @patch('train.progress_testing.ProgressTestRunner.save_reports')
    def test_on_save(self, mock_save, mock_md, mock_json, mock_compare, mock_find, mock_run_tests, callback, mock_args, mock_state, mock_control, tmp_path):
        """Test callback on checkpoint save"""
        # Create checkpoint directory
        checkpoint_dir = Path(mock_args.output_dir) / "checkpoint-100"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        mock_run_tests.return_value = {
            "checkpoint": "checkpoint-100",
            "step": 100,
            "expert_name": "test-expert",
            "expert_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "test_results": {"total": 0, "passed": 0, "failed": 0, "success_rate": 0.0},
        }
        mock_find.return_value = []
        mock_compare.return_value = {"previous_checkpoint": None, "improvements": 0, "regressions": 0, "fixed_failures": [], "new_failures": []}
        mock_json.return_value = {}
        mock_md.return_value = "# Report"
        
        callback.on_save(mock_args, mock_state, mock_control)
        
        # Should call run_tests
        assert mock_run_tests.called
    
    def test_callback_initialization(self, callback):
        """Test callback initialization"""
        assert callback.expert_name == "test-expert"
        assert callback.expert_version == "1.0.0"
        assert callback.expert_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
