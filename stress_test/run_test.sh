#!/bin/bash
# Quick stress test runner

echo "🚀 Lambda Architecture Stress Test"
echo "=================================="
echo ""
echo "Make sure API server is running on http://localhost:8000"
echo ""

# Check if locust is installed
if ! command -v locust &> /dev/null; then
    echo "❌ Locust not found. Installing..."
    pip install locust
fi

# Check if server is running
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "❌ API server not responding on http://localhost:8000"
    echo "Please start server first:"
    echo "  cd /home/hieupt/Bản\ tải\ về/emotion_detection-software_architecture"
    echo "  python main.py"
    exit 1
fi

echo "✅ API server is running"
echo ""
echo "Choose test mode:"
echo "  1) GUI Mode (recommended) — Opens browser"
echo "  2) Headless Mode — 50 users, 5 minutes"
echo "  3) Stress Test — 200 users, 10 minutes"
echo "  4) Speed Layer Only — Test realtime tracking"
echo ""
read -p "Enter choice (1-4): " choice

cd "/home/hieupt/Bản tải về/emotion_detection-software_architecture"

case $choice in
    1)
        echo "🌐 Starting Locust in GUI mode..."
        echo "Open http://localhost:8089 in your browser"
        locust -f stress_test/locustfile.py --host=http://localhost:8000
        ;;
    2)
        echo "🤖 Running headless test (50 users, 5 min)..."
        locust -f stress_test/locustfile.py --host=http://localhost:8000 \
          --headless -u 50 -r 5 -t 5m \
          --html stress_test_reports/report_$(date +%Y%m%d_%H%M%S).html
        echo "✅ Test complete! Check stress_test_reports/ for HTML report"
        ;;
    3)
        echo "🔥 Running stress test (200 users, 10 min)..."
        locust -f stress_test/locustfile.py --host=http://localhost:8000 \
          --headless -u 200 -r 20 -t 10m \
          --html stress_test_reports/report_$(date +%Y%m%d_%H%M%S).html
        echo "✅ Test complete! Check stress_test_reports/ for HTML report"
        ;;
    4)
        echo "⚡ Testing Speed Layer only..."
        locust -f stress_test/locustfile.py --host=http://localhost:8000 \
          --headless -u 100 -r 10 -t 3m \
          SpeedLayerUser \
          --html stress_test_reports/speed_layer_$(date +%Y%m%d_%H%M%S).html
        echo "✅ Test complete! Check stress_test_reports/ for HTML report"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac