import sys
import time
import requests

BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"
TEST_IMAGE = "TestImage.jpg"


def test_health():
    print("Testing health endpoint...")

    try:
        response = requests.get(HEALTH_URL, timeout=10)

        if response.status_code != 200:
            print(
                f"❌ Health check failed: "
                f"HTTP {response.status_code}"
            )
            return False

        print("✅ Health check passed")
        print(f"Response: {response.text}")
        return True

    except requests.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_prediction():
    print("\nTesting prediction endpoint...")

    try:
        with open(TEST_IMAGE, "rb") as image:
            files = {
                "file": (
                    TEST_IMAGE,
                    image,
                    "image/jpeg"
                )
            }

            response = requests.post(
                PREDICT_URL,
                files=files,
                timeout=30
            )

        if response.status_code != 200:
            print(
                f"❌ Prediction test failed: "
                f"HTTP {response.status_code}"
            )
            print(response.text)
            return False

        print("✅ Prediction test passed")
        print(f"Response: {response.text}")
        return True

    except FileNotFoundError:
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False

    except requests.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        return False


def main():
    print("================================")
    print("Post-deployment Smoke Test")
    print("================================")

    # Give the container a little time to start
    print("\nWaiting for API...")
    time.sleep(3)

    health_ok = test_health()

    if not health_ok:
        print("\n❌ Smoke test FAILED")
        sys.exit(1)

    prediction_ok = test_prediction()

    if not prediction_ok:
        print("\n❌ Smoke test FAILED")
        sys.exit(1)

    print("\n================================")
    print("✅ All smoke tests PASSED")
    print("================================")

    sys.exit(0)


if __name__ == "__main__":
    main()