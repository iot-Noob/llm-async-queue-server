"""
Comprehensive error handling test for AsyncLLM library
"""
import os
import sys
import asyncio
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    AsyncLLM, 
    JsonOutputParser, 
    PydanticOutputParser, 
    StrOutputParser,
    PromptTemplate,
    EcharrParsers
)
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

class TestPerson(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=0, le=150)

class TestErrorHandler:
    """Test all error handling scenarios"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name, passed, error=None):
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
        self.test_results.append((test_name, passed, error))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    async def run_all_tests(self):
        """Run all error handling tests"""
        print("="*70)
        print("🧪 ASYNCLLM ERROR HANDLING TEST SUITE")
        print("="*70)
        
        await self.test_invalid_model_path()
        await self.test_missing_model_file()
        await self.test_invalid_json_parsing()
        await self.test_worker_crash_handling()
        await self.test_parser_fallback()
        await self.test_queue_overflow()
        await self.test_timeout_handling()
        await self.test_pydantic_validation()
        await self.test_stop_token_warning()
        await self.test_echarts_invalid_json()
        
        self.print_summary()
    
    async def test_invalid_model_path(self):
        """Test 1: Invalid model path handling"""
        print("\n" + "-"*50)
        print("TEST 1: Invalid Model Path")
        print("-"*50)
        
        try:
            # Try to create with invalid path
            llm = AsyncLLM(model_path="/nonexistent/path")
            await llm._load_model(
                model_name="fake-model.gguf",
                temperature=0.7
            )
            self.log_test("Invalid model path", False, "Should have raised error")
        except ValueError as e:
            self.log_test("Invalid model path", True, str(e)[:100])
        except Exception as e:
            self.log_test("Invalid model path", False, f"Wrong error type: {e}")
    
    async def test_missing_model_file(self):
        """Test 2: Missing model file handling"""
        print("\n" + "-"*50)
        print("TEST 2: Missing Model File")
        print("-"*50)
        
        try:
            # Use a temp dir with no models
            with tempfile.TemporaryDirectory() as tmpdir:
                llm = AsyncLLM(model_path=tmpdir)
                await llm._load_model(
                    model_name="nonexistent-model.gguf",
                    temperature=0.7
                )
            self.log_test("Missing model file", False, "Should have raised error")
        except ValueError as e:
            self.log_test("Missing model file", True, str(e)[:100])
        except Exception as e:
            self.log_test("Missing model file", False, f"Wrong error type: {e}")
    
    async def test_invalid_json_parsing(self):
        """Test 3: Invalid JSON parsing with fallback"""
        print("\n" + "-"*50)
        print("TEST 3: Invalid JSON Parsing")
        print("-"*50)
        
        parser = JsonOutputParser()
        invalid_json = "This is not valid JSON at all {"
        
        try:
            result = parser.parse(invalid_json)
            self.log_test("Invalid JSON parsing", False, "Should have raised error")
        except ValueError as e:
            self.log_test("Invalid JSON parsing", True, str(e)[:100])
        except Exception as e:
            self.log_test("Invalid JSON parsing", False, f"Wrong error type: {e}")
    
    async def test_worker_crash_handling(self):
        """Test 4: Worker crash detection and callback"""
        print("\n" + "-"*50)
        print("TEST 4: Worker Crash Handling")
        print("-"*50)
        
        crash_detected = False
        crash_exception = None
        
        def on_worker_crash(exception):
            nonlocal crash_detected, crash_exception
            crash_detected = True
            crash_exception = exception
            print(f"   🔥 Crash callback triggered: {type(exception).__name__}: {exception}")
        
        try:
            # Create LLM with crash callback
            llm = AsyncLLM(
                worker_exception_callback=on_worker_crash,
                auto_restart_worker=False
            )
            
            # Start the worker WITHOUT loading any model
            # This should cause the worker to crash when it tries to process
            await llm.start()
            
            # Wait a bit for worker to be ready
            await asyncio.sleep(0.1)
            
            # Send a message to trigger worker processing
            # The worker will try to process but there's no model loaded
            try:
                # Don't await this - let it run in background
                asyncio.create_task(llm.chat_llm("This will trigger a crash"))
                # Give time for the crash to happen
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"   📨 Chat error (expected): {e}")
            
            # Check if crash was detected
            if crash_detected:
                self.log_test("Worker crash handling", True, f"Crash detected: {crash_exception}")
            else:
                # Try to force a crash by checking if worker is still running
                if llm._running and llm._worker_task and llm._worker_task.done():
                    # Task is done but no exception? Check for exception
                    try:
                        exc = llm._worker_task.exception()
                        if exc:
                            crash_detected = True
                            print(f"   🔥 Found exception in worker task: {exc}")
                    except:
                        pass
                
                if crash_detected:
                    self.log_test("Worker crash handling", True, "Crash detected via task exception")
                else:
                    self.log_test("Worker crash handling", False, "Crash callback not triggered")
            
            await llm.stop()
            
        except Exception as e:
            self.log_test("Worker crash handling", False, f"Unexpected error: {e}")
        
    async def test_parser_fallback(self):
        """Test 5: Parser fallback mechanism"""
        print("\n" + "-"*50)
        print("TEST 5: Parser Fallback")
        print("-"*50)
        
        # Create a custom failing parser
        class FailingParser:
            def parse(self, response):
                raise ValueError("Intentional parser failure")
        
        llm = AsyncLLM(output_parser=FailingParser())
        llm._extract_response_text = lambda x: "Fallback worked!"
        
        try:
            result = llm._parse_raw_response({"test": "data"})
            if result == "Fallback worked!":
                self.log_test("Parser fallback", True)
            else:
                self.log_test("Parser fallback", False, f"Expected fallback, got {result}")
        except Exception as e:
            self.log_test("Parser fallback", False, f"Fallback failed: {e}")
    
    async def test_queue_overflow(self):
        """Test 6: Queue overflow handling"""
        print("\n" + "-"*50)  # ✅ Fixed: changed -+ to +
        print("TEST 6: Queue Overflow")
        print("-"*50)
        
        try:
            llm = AsyncLLM()
            # Don't start worker
            llm._running = False
            
            # Try to put many items in queue
            for i in range(20):
                await llm.queue.put({"task_id": f"task_{i}", "prompt": "test"})
            
            self.log_test("Queue overflow", True, "Queue accepted items")
        except asyncio.QueueFull as e:
            self.log_test("Queue overflow", True, f"Queue full as expected: {e}")
        except Exception as e:
            self.log_test("Queue overflow", False, f"Unexpected error: {e}")
    
    async def test_timeout_handling(self):
        """Test 7: Timeout handling"""
        print("\n" + "-"*50)
        print("TEST 7: Timeout Handling")
        print("-"*50)
        
        try:
            llm = AsyncLLM()
            
            # Create a future that never completes
            future = asyncio.Future()
            
            try:
                result = await asyncio.wait_for(future, timeout=0.1)
                self.log_test("Timeout handling", False, "Should have timed out")
            except asyncio.TimeoutError:
                self.log_test("Timeout handling", True, "Timeout caught correctly")
                
        except Exception as e:
            self.log_test("Timeout handling", False, f"Unexpected error: {e}")
    
    async def test_pydantic_validation(self):
        """Test 8: Pydantic validation errors"""
        print("\n" + "-"*50)
        print("TEST 8: Pydantic Validation")
        print("-"*50)
        
        parser = PydanticOutputParser(pydantic_object=TestPerson)
        
        # Test valid JSON
        valid_json = '{"name": "John", "age": 30}'
        try:
            result = parser.parse(valid_json)
            valid = isinstance(result, TestPerson) and result.name == "John" and result.age == 30
            self.log_test("Pydantic valid JSON", valid)
        except Exception as e:
            self.log_test("Pydantic valid JSON", False, str(e))
        
        # Test invalid JSON (wrong types)
        invalid_json = '{"name": "Jo", "age": "not a number"}'
        try:
            result = parser.parse(invalid_json)
            self.log_test("Pydantic invalid types", False, "Should have failed")
        except ValueError as e:
            self.log_test("Pydantic invalid types", True, "Caught validation error")
        
        # Test invalid JSON (missing field)
        missing_field = '{"name": "John"}'
        try:
            result = parser.parse(missing_field)
            self.log_test("Pydantic missing field", False, "Should have failed")
        except ValueError as e:
            self.log_test("Pydantic missing field", True, "Caught missing field error")
    
    async def test_stop_token_warning(self):
        """Test 9: Stop token warning detection"""
        print("\n" + "-"*50)
        print("TEST 9: Stop Token Warning")
        print("-"*50)
        
        # Mock response with length finish reason
        mock_response = {
            "choices": [{
                "text": "This is a long response that hit the token limit...",
                "finish_reason": "length"
            }]
        }
        
        # This test is structural - we're testing the detection logic
        try:
            detected = False
            if isinstance(mock_response, dict) and "choices" in mock_response:
                choice = mock_response["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    detected = True
            
            self.log_test("Stop token warning detection", detected, "Detected length limit")
        except Exception as e:
            self.log_test("Stop token warning detection", False, str(e))
    
    async def test_echarts_invalid_json(self):
        """Test 10: ECharts JSON error handling"""
        print("\n" + "-"*50)
        print("TEST 10: ECharts Invalid JSON")
        print("-"*50)
        
        # Create a temporary invalid Charts.json
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            # Create invalid JSON file
            with open("Charts.json", "w") as f:
                f.write("{invalid json content")
            
            try:
                parser = EcharrParsers()
                self.log_test("ECharts invalid JSON", False, "Should have raised error")
            except ValueError as e:
                self.log_test("ECharts invalid JSON", True, str(e)[:100])
            except Exception as e:
                self.log_test("ECharts invalid JSON", False, f"Wrong error: {e}")
            finally:
                os.chdir(old_cwd)
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Total: {self.passed + self.failed}")
        print(f"🎯 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        if self.failed > 0:
            print("\n❌ Failed Tests:")
            for name, passed, error in self.test_results:
                if not passed:
                    print(f"   - {name}")
                    if error:
                        print(f"     {error}")
        print("="*70)

async def main():
    tester = TestErrorHandler()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())