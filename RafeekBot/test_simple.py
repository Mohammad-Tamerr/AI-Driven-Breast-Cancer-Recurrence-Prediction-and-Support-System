import requests
import json

BASE_URL = "http://localhost:8080"

print("🚀 Starting Simple Tests...\n")

# Test 1: سؤال عام
print("=" * 50)
print("Test 1: سؤال عام")
print("=" * 50)

response = requests.post(
    f"{BASE_URL}/chat",
    json={"question": "ما هو سرطان الثدي؟"}
)

print(f"Status: {response.status_code}")
print(f"Answer: {response.json()['answer'][:200]}...\n")

# Test 2: مع بيانات المريضة P001
print("=" * 50)
print("Test 2: المريضة P001")
print("=" * 50)

response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "patient_id": "P001",
        "question": "هل العلاج الكيماوي مناسب لحالتي؟"
    }
)

print(f"Status: {response.status_code}")
answer = response.json()['answer']
print(f"Answer: {answer[:300]}...")

# تحقق إن البوت استخدم اسم المريضة
if "مريضة ١" in answer or "P001" in answer:
    print("✅ البوت استخدم بيانات المريضة!")
else:
    print("⚠️ البوت ما استخدمش بيانات المريضة")

print("\n" + "=" * 50)

# Test 3: المريضة P003 (Triple-negative)
print("Test 3: المريضة P003")
print("=" * 50)

response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "patient_id": "P003",
        "question": "ما هي خيارات العلاج المتاحة؟"
    }
)

print(f"Status: {response.status_code}")
print(f"Answer: {response.json()['answer'][:250]}...\n")

print("✅ Tests completed!")