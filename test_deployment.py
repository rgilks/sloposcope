#!/usr/bin/env python3
"""
Test script to verify the deployed sloposcope application
"""

import requests
import json
import time


def test_backend():
    """Test the Cloudflare Worker backend"""
    print("🧪 Testing Backend API...")

    # Test health endpoint
    try:
        response = requests.get("https://sloposcope-prod.rob-gilks.workers.dev/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test analyze endpoint
    try:
        test_data = {
            "text": "This is a test text to analyze for AI slop patterns. It contains some repetitive phrases and might show signs of AI generation.",
            "domain": "general",
            "explain": True,
            "spans": True,
        }

        response = requests.post(
            "https://sloposcope-prod.rob-gilks.workers.dev/analyze",
            headers={"Content-Type": "application/json"},
            json=test_data,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis successful:")
            print(f"   - Slop Score: {data['slop_score']:.3f} ({data['level']})")
            print(f"   - Confidence: {data['confidence']:.3f}")
            print(f"   - Metrics: {len(data['metrics'])} analyzed")
            print(f"   - Processing time: {data['timings_ms']['total']}ms")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def test_frontend():
    """Test the Cloudflare Pages frontend"""
    print("\n🌐 Testing Frontend...")

    # Test the main page
    try:
        response = requests.get("https://sloposcope-web.pages.dev/", timeout=10)
        if response.status_code == 200:
            if "Sloposcope" in response.text:
                print("✅ Frontend page loads successfully")
                return True
            else:
                print("❌ Frontend page content unexpected")
                return False
        else:
            print(f"❌ Frontend page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Sloposcope Deployment")
    print("=" * 50)

    backend_ok = test_backend()
    frontend_ok = test_frontend()

    print("\n📊 Test Results:")
    print("=" * 50)
    print(f"Backend API: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Frontend: {'✅ PASS' if frontend_ok else '❌ FAIL'}")

    if backend_ok and frontend_ok:
        print(
            "\n🎉 All tests passed! Your sloposcope application is deployed and working."
        )
        print("\n🔗 URLs:")
        print("   Backend API: https://sloposcope-prod.rob-gilks.workers.dev")
        print("   Frontend: https://sloposcope-web.pages.dev")
        print("\n📝 Next steps:")
        print("   1. Set up a custom domain")
        print("   2. Integrate the real Python sloposcope code")
        print("   3. Add authentication if needed")
        print("   4. Set up monitoring and analytics")
    else:
        print("\n❌ Some tests failed. Please check the deployment.")


if __name__ == "__main__":
    main()
