#!/usr/bin/env python3
"""
Kịch bản test đơn giản cho Emotion Detection API
Không cần complex dependencies - chỉ cần requests
"""

import requests
import json
import time

# ── Cấu hình API ────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 120

# ── Test Cases ───────────────────────────────────────────
TEST_SCENARIOS = [
    {
        "name": "Happy Customer",
        "expected_emotion": "happy",
        "messages": [
            "Cảm ơn shop nhiều nhé! Hàng đẹp quá 😊",
            "Ok để mai tôi qua lấy nhé"
        ]
    },
    {
        "name": "Confused Customer",
        "expected_emotion": "confused",
        "messages": [
            "Cho tôi hỏi về sản phẩm này",
            "Nghĩa là sao? Tôi không hiểu"
        ]
    },
    {
        "name": "Anxious Customer",
        "expected_emotion": "anxious",
        "messages": [
            "Tôi đã đặt hàng 2 ngày rồi",
            "Sao chưa thấy ship confirm vậy? Tôi cần gấp lắm"
        ]
    },
    {
        "name": "Frustrated Customer",
        "expected_emotion": "frustrated",
        "messages": [
            "Sao chưa thấy trả lời vậy?",
            "Thôi frustrating quá, support chậm quá"
        ]
    },
    {
        "name": "Disappointed Customer (Nam Dialect)",
        "expected_emotion": "disappointed",
        "messages": [
            "Hương ơi, đơn mình bị hỏng rồi",
            "Thôi kệ đi, bể hết trơn. Mình sẽ không mua lại nữa đâu"
        ]
    },
    {
        "name": "Angry Customer",
        "expected_emotion": "angry",
        "messages": [
            "Sao hàng đến mà sai vậy?",
            "Vcl cái gì thế này? Lần cuối cùng đấy"
        ]
    },
    {
        "name": "Neutral Customer",
        "expected_emotion": "neutral",
        "messages": [
            "Cho hỏi giá sản phẩm này",
            "Ok cảm ơn, tôi sẽ xem lại"
        ]
    }
]

# ── Helper Functions ─────────────────────────────────────

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def call_chat_endpoint(message, ticket_id=None):
    """Call /chat endpoint"""
    url = f"{API_BASE_URL}/chat"
    payload = {"message": message}
    if ticket_id:
        payload["ticket_id"] = ticket_id

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_basic_connection():
    """Test 1: Basic API Connection"""
    print_header("TEST 1: Basic API Connection")

    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print(f"  ✅ API is running: {response.json()}")
            return True
        else:
            print(f"  ❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cannot connect to API: {e}")
        return False

def test_chat_basic():
    """Test 2: Basic Chat Functionality"""
    print_header("TEST 2: Basic Chat Functionality")

    result = call_chat_endpoint("Xin chào, bạn là ai?")

    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
        return False

    print(f"  ✅ Message: Xin chào, bạn là ai?")
    print(f"  ✅ Reply: {result.get('reply', '')[:100]}...")
    print(f"  ✅ Emotion: {result.get('emotion')} (confidence: {result.get('confidence', 0):.2f})")
    return True

def test_emotion_scenarios():
    """Test 3: Emotion Detection Scenarios"""
    print_header("TEST 3: Emotion Detection (7 Scenarios)")

    results = []

    for scenario in TEST_SCENARIOS:
        print(f"\n  📝 Scenario: {scenario['name']}")
        print(f"     Expected Emotion: {scenario['expected_emotion']}")

        ticket_id = None
        emotions_detected = []

        # Send all messages
        for msg in scenario['messages']:
            result = call_chat_endpoint(msg, ticket_id)
            if "error" not in result:
                ticket_id = result.get('ticket_id')
                emotion = result.get('emotion', 'unknown')
                confidence = result.get('confidence', 0)
                emotions_detected.append(emotion)

                print(f"     Message: {msg[:50]}...")
                print(f"     → Detected: {emotion} (confidence: {confidence:.2f})")

        # Check final emotion
        final_emotion = emotions_detected[-1] if emotions_detected else "unknown"
        expected = scenario['expected_emotion']

        match = final_emotion == expected
        icon = "✅" if match else "⚠️"
        print(f"     {icon} Final: {final_emotion} (expected: {expected})")

        results.append({
            "scenario": scenario['name'],
            "expected": expected,
            "detected": final_emotion,
            "match": match
        })

    return results

def test_conversation_context():
    """Test 4: Conversation Context Preservation"""
    print_header("TEST 4: Conversation Context")

    print("\n  Testing multi-turn conversation...")

    ticket_id = None
    conversation = [
        ("Agent", "Xin chào, tôi có thể giúp gì?"),
        ("Customer", "Tôi muốn hỏi về đơn hàng"),
        ("Agent", "Dạ cho tôi xin mã đơn hàng"),
        ("Customer", "Đơn hàng #12345, sao chưa ship?")
    ]

    for role, message in conversation:
        if role == "Customer":
            result = call_chat_endpoint(message, ticket_id)
            if "error" not in result:
                ticket_id = result.get('ticket_id')
                emotion = result.get('emotion')
                print(f"  📤 {role}: {message}")
                print(f"     → Emotion: {emotion}")
        else:
            print(f"  📥 {role}: {message}")

    return True

def test_streaming():
    """Test 5: Streaming Endpoint"""
    print_header("TEST 5: Streaming Endpoint")

    try:
        url = f"{API_BASE_URL}/chat/stream"
        payload = {"message": "Test streaming"}

        response = requests.post(url, json=payload, stream=True, timeout=API_TIMEOUT)

        if response.status_code == 200:
            print(f"  ✅ Streaming endpoint is working")
            print(f"  📊 Headers: {dict(response.headers)}")
            return True
        else:
            print(f"  ⚠️  Status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"  ⚠️  Streaming test skipped: {e}")
        return False

def generate_report(emotion_results):
    """Generate final report"""
    print_header("FINAL REPORT")

    total = len(emotion_results)
    matches = sum(1 for r in emotion_results if r['match'])
    accuracy = (matches / total * 100) if total > 0 else 0

    print(f"\n  📊 Test Results: {matches}/{total} correct ({accuracy:.1f}% accuracy)")

    print("\n  Detailed Results:")
    for result in emotion_results:
        icon = "✅" if result['match'] else "❌"
        print(f"     {icon} {result['scenario']}")
        print(f"        Expected: {result['expected']} | Detected: {result['detected']}")

    print("\n" + "=" * 70)

    if accuracy >= 80:
        print("  🎉 EXCELLENT - High accuracy!")
    elif accuracy >= 60:
        print("  ⚠️  GOOD - Some improvements needed")
    else:
        print("  🔧 NEEDS WORK - Low accuracy detected")

    print("=" * 70 + "\n")

def main():
    print_header("🤖 EMOTION DETECTION API - SIMPLE TEST SUITE")
    print(f"\n  API URL: {API_BASE_URL}")
    print(f"  Timeout: {API_TIMEOUT}s")
    print(f"  Test Scenarios: {len(TEST_SCENARIOS)}")

    start_time = time.time()

    # Run tests
    test_results = {}

    # Test 1: Connection
    test_results['Connection'] = test_basic_connection()
    if not test_results['Connection']:
        print("\n  ❌ Cannot connect to API. Please check if the server is running:")
        print("     python main.py")
        return

    # Test 2: Basic Chat
    test_results['Basic Chat'] = test_chat_basic()

    # Test 3: Emotion Detection
    emotion_results = test_emotion_scenarios()

    # Test 4: Context
    test_results['Conversation Context'] = test_conversation_context()

    # Test 5: Streaming
    test_results['Streaming'] = test_streaming()

    # Generate report
    generate_report(emotion_results)

    elapsed = time.time() - start_time
    print(f"  ⏱️  Total time: {elapsed:.2f}s\n")

if __name__ == "__main__":
    main()