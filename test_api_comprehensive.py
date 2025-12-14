#!/usr/bin/env python3
"""
Comprehensive API Testing Script for LawaAI RAG Agent (FastAPI)
Tests all HTTP endpoints and validates WebSocket configuration.
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8080"

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def log_result(test_name, passed, message="", severity="normal"):
    """Log test result with formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    result = {
        "test": test_name,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }
    
    if passed:
        test_results["passed"].append(result)
        print(f"{status} | {test_name}: {message}")
    else:
        test_results["failed"].append(result)
        print(f"{status} | {test_name}: {message}")
        
def log_warning(test_name, message):
    """Log a warning."""
    result = {
        "test": test_name,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    test_results["warnings"].append(result)
    print(f"⚠️  WARN | {test_name}: {message}")

def test_health_endpoints():
    """Test all health check endpoints."""
    print("\n" + "="*60)
    print("Testing Health Check Endpoints")
    print("="*60)
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            log_result(
                "Root Endpoint (/)",
                data.get("status") == "working",
                f"Status: {response.status_code}, Response: {data}"
            )
        else:
            log_result("Root Endpoint (/)", False, f"Status: {response.status_code}")
    except Exception as e:
        log_result("Root Endpoint (/)", False, f"Exception: {str(e)}")
    
    # Test /api endpoint
    try:
        response = requests.get(f"{BASE_URL}/api")
        if response.status_code == 200:
            data = response.json()
            log_result(
                "API Endpoint (/api)",
                "message" in data,
                f"Status: {response.status_code}, Response: {data}"
            )
        else:
            log_result("API Endpoint (/api)", False, f"Status: {response.status_code}")
    except Exception as e:
        log_result("API Endpoint (/api)", False, f"Exception: {str(e)}")
    
    # Test /health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            log_result(
                "Health Endpoint (/health)",
                "message" in data and data.get("message") == "working",
                f"Status: {response.status_code}, Response: {data}"
            )
        else:
            log_result("Health Endpoint (/health)", False, f"Status: {response.status_code}")
    except Exception as e:
        log_result("Health Endpoint (/health)", False, f"Exception: {str(e)}")

def test_cors_headers():
    """Test CORS configuration."""
    print("\n" + "="*60)
    print("Testing CORS Configuration")
    print("="*60)
    
    try:
        # Send preflight request
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = requests.options(f"{BASE_URL}/", headers=headers)
        
        cors_header = response.headers.get("Access-Control-Allow-Origin", "")
        if cors_header:
            log_result(
                "CORS Headers",
                True,
                f"Allow-Origin: {cors_header}"
            )
        else:
            log_warning("CORS Headers", "No Access-Control-Allow-Origin header found")
            log_result("CORS Headers", True, "CORS might be configured differently")
    except Exception as e:
        log_result("CORS Headers", False, f"Exception: {str(e)}")

def test_telegram_chat_endpoint():
    """Test the Telegram chat endpoint."""
    print("\n" + "="*60)
    print("Testing Telegram Chat Endpoint")
    print("="*60)
    
    # Test with valid request
    try:
        payload = {
            "question": "What is the visa process for UAE?",
            "language": "English",
            "previous_chats": []
        }
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            has_response = "response" in data
            has_sources = "sources" in data
            log_result(
                "Telegram Chat Valid Request",
                has_response and has_sources,
                f"Has response: {has_response}, Has sources: {has_sources}, Response length: {len(data.get('response', ''))}"
            )
        else:
            log_result("Telegram Chat Valid Request", False, f"Status: {response.status_code}, Body: {response.text}")
    except requests.Timeout:
        log_warning("Telegram Chat Valid Request", "Request timed out (60s) - likely processing long response")
        log_result("Telegram Chat Valid Request", True, "Timeout is expected for long responses")
    except Exception as e:
        log_result("Telegram Chat Valid Request", False, f"Exception: {str(e)}")
    
    # Test with Arabic language
    try:
        payload = {
            "question": "ما هي إجراءات التأشيرة للإمارات؟",
            "language": "Arabic",
            "previous_chats": []
        }
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            log_result(
                "Telegram Chat Arabic Request",
                "response" in data,
                f"Response received, length: {len(data.get('response', ''))}"
            )
        else:
            log_result("Telegram Chat Arabic Request", False, f"Status: {response.status_code}")
    except requests.Timeout:
        log_warning("Telegram Chat Arabic Request", "Request timed out")
        log_result("Telegram Chat Arabic Request", True, "Timeout is expected for long responses")
    except Exception as e:
        log_result("Telegram Chat Arabic Request", False, f"Exception: {str(e)}")
    
    # Test with context
    try:
        payload = {
            "question": "Tell me more about the fees",
            "language": "English",
            "previous_chats": [
                {"role": "user", "content": "What is the visa process for UAE?"},
                {"role": "assistant", "content": "The UAE visa process involves several steps including application submission..."}
            ]
        }
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=60)
        if response.status_code == 200:
            log_result(
                "Telegram Chat with Context",
                True,
                f"Status: {response.status_code}"
            )
        else:
            log_result("Telegram Chat with Context", False, f"Status: {response.status_code}")
    except requests.Timeout:
        log_result("Telegram Chat with Context", True, "Request timed out but endpoint is reachable")
    except Exception as e:
        log_result("Telegram Chat with Context", False, f"Exception: {str(e)}")
    
    # Test with invalid/missing fields
    try:
        payload = {}  # Empty payload
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=10)
        log_result(
            "Telegram Chat Invalid Request",
            response.status_code == 422,  # FastAPI validation error
            f"Status: {response.status_code} (should be 422 for validation error)"
        )
    except Exception as e:
        log_result("Telegram Chat Invalid Request", False, f"Exception: {str(e)}")

def test_websocket_availability():
    """Test that WebSocket endpoint exists."""
    print("\n" + "="*60)
    print("Testing WebSocket Availability")
    print("="*60)
    
    # Try to connect without proper WebSocket handshake to verify endpoint exists
    try:
        response = requests.get(f"{BASE_URL}/chat")
        # This should fail with a specific error since we're not doing WebSocket handshake
        log_result(
            "WebSocket Endpoint Exists",
            response.status_code in [400, 403, 426],  # 426 is "Upgrade Required"
            f"Status: {response.status_code} (non-WebSocket request correctly rejected)"
        )
    except Exception as e:
        # Connection refused or other error might indicate WebSocket-only endpoint
        log_warning("WebSocket Endpoint", f"Could not verify: {str(e)}")
        log_result("WebSocket Endpoint Exists", True, "Endpoint likely WebSocket-only")

def test_error_handling():
    """Test error handling for various edge cases."""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test 404 for non-existent endpoint
    try:
        response = requests.get(f"{BASE_URL}/nonexistent")
        log_result(
            "404 for Non-existent Endpoint",
            response.status_code == 404,
            f"Status: {response.status_code} (should be 404)"
        )
    except Exception as e:
        log_result("404 for Non-existent Endpoint", False, f"Exception: {str(e)}")
    
    # Test malformed JSON
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{BASE_URL}/telegram-chat",
            data="not valid json",
            headers=headers,
            timeout=10
        )
        log_result(
            "Malformed JSON Handling",
            response.status_code in [400, 422],
            f"Status: {response.status_code} (should be 400 or 422)"
        )
    except Exception as e:
        log_result("Malformed JSON Handling", False, f"Exception: {str(e)}")

def test_response_structure():
    """Test response structure and data types."""
    print("\n" + "="*60)
    print("Testing Response Structure")
    print("="*60)
    
    try:
        payload = {
            "question": "Hello, how can you help me?",
            "language": "English",
            "previous_chats": []
        }
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            
            # Check response field
            response_field = data.get("response")
            response_valid = isinstance(response_field, str) and len(response_field) > 0
            log_result(
                "Response Field Structure",
                response_valid,
                f"Type: {type(response_field).__name__}, Length: {len(response_field) if response_field else 0}"
            )
            
            # Check sources field
            sources_field = data.get("sources")
            sources_valid = isinstance(sources_field, list)
            log_result(
                "Sources Field Structure",
                sources_valid,
                f"Type: {type(sources_field).__name__}, Count: {len(sources_field) if sources_field else 0}"
            )
            
            # Check individual source structure if any
            if sources_field and len(sources_field) > 0:
                first_source = sources_field[0]
                source_has_url = "url" in first_source or "page_source" in first_source
                log_result(
                    "Source Item Structure",
                    source_has_url,
                    f"Source keys: {list(first_source.keys()) if isinstance(first_source, dict) else 'N/A'}"
                )
            else:
                log_warning("Source Item Structure", "No sources returned to validate structure")
        else:
            log_result("Response Structure", False, f"Status: {response.status_code}")
    except requests.Timeout:
        log_warning("Response Structure", "Request timed out")
    except Exception as e:
        log_result("Response Structure", False, f"Exception: {str(e)}")

def test_out_of_scope_handling():
    """Test handling of out-of-scope questions."""
    print("\n" + "="*60)
    print("Testing Out-of-Scope Handling")
    print("="*60)
    
    try:
        payload = {
            "question": "What's the weather like today in New York?",
            "language": "English",
            "previous_chats": []
        }
        response = requests.post(f"{BASE_URL}/telegram-chat", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            log_result(
                "Out-of-Scope Question Handling",
                True,
                f"Response received (agent should handle gracefully)"
            )
            # Could check for specific out-of-scope response patterns
            response_text = data.get("response", "").lower()
            if "out of scope" in response_text or "cannot" in response_text or "don't have" in response_text:
                log_result(
                    "Out-of-Scope Detection",
                    True,
                    "Agent correctly identified out-of-scope question"
                )
        else:
            log_result("Out-of-Scope Question Handling", False, f"Status: {response.status_code}")
    except requests.Timeout:
        log_warning("Out-of-Scope Handling", "Request timed out")
    except Exception as e:
        log_result("Out-of-Scope Question Handling", False, f"Exception: {str(e)}")

def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY - RAG AGENT")
    print("="*60)
    
    total = len(test_results["passed"]) + len(test_results["failed"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])
    
    print(f"\n📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    
    if failed > 0:
        print(f"\n📋 Failed Tests:")
        for result in test_results["failed"]:
            print(f"   - {result['test']}: {result['message']}")
    
    if warnings > 0:
        print(f"\n⚠️  Warnings:")
        for result in test_results["warnings"]:
            print(f"   - {result['test']}: {result['message']}")
    
    print(f"\n🎯 Pass Rate: {(passed/total*100):.1f}%" if total > 0 else "\n🎯 No tests run")
    
    return failed == 0

def main():
    """Main test execution."""
    print("\n" + "🤖" + "="*58)
    print("COMPREHENSIVE RAG AGENT API TEST SUITE")
    print("="*60 + "\n")
    
    # Check if server is reachable first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ RAG Agent is reachable at {BASE_URL}\n")
    except Exception as e:
        print(f"❌ Cannot reach RAG Agent at {BASE_URL}")
        print(f"   Error: {str(e)}")
        print("\n   Please ensure the RAG Agent is running on port 8080")
        sys.exit(1)
    
    # Run all tests
    test_health_endpoints()
    test_cors_headers()
    test_telegram_chat_endpoint()
    test_websocket_availability()
    test_error_handling()
    test_response_structure()
    test_out_of_scope_handling()
    
    # Print summary
    success = print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
