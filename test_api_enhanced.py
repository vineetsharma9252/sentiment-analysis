import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    result = response.json()
    print("Health Check:", result)
    return result['model_loaded']

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model_info")
    result = response.json()
    print("Model Info:", result)
    return response.status_code == 200

def test_predict():
    """Test single prediction endpoint with samples from the dataset"""
    test_samples = [
        "What a great day!!! Looks like dream.",
        "Don't angry me",
        "Its night 2 am, feeling neutral",
        "I'm depressed, I'm thinking about suicide, what I need to do now?",
        "happy mother`s day to all moms out there. Mothers are our real hero"
    ]
    
    for text in test_samples:
        test_data = {"text": text}
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        result = response.json()
        print(f"Text: {text[:50]}... → {result['sentiment']} (conf: {result['confidence']:.3f})")
    
    return response.status_code == 200

def test_batch_predict():
    """Test batch prediction endpoint"""
    test_data = {
        "texts": [
            "What a great day!!! Looks like dream.",
            "Don't angry me",
            "Its night 2 am, feeling neutral",
            "I'm really happy with this amazing product!",
            "This is terrible and awful service"
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=test_data)
    results = response.json()
    print("Batch Predictions:")
    for result in results:
        print(f"  - {result['text'][:30]}... → {result['sentiment']} (conf: {result['confidence']:.3f})")
    return response.status_code == 200

if __name__ == "__main__":
    print("🧪 Testing Sentiment Analysis API with Custom Dataset...\n")
    
    # Check health first
    if not test_health():
        print("❌ Model not loaded. Please run training script first!")
        exit(1)
    
    tests = [test_model_info, test_predict, test_batch_predict]
    
    for test in tests:
        try:
            success = test()
            print(f"✓ {test.__name__}: {'PASSED' if success else 'FAILED'}\n")
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {str(e)}\n")