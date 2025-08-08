import requests
import json
import base64
import os

def create_test_document():
    """Create a simple test document"""
    test_content = """
    This is a test document for the RAG system.
    
    The company policy states that:
    1. Expense claims up to $500 are automatically approved
    2. Claims between $500-$2000 require manager approval
    3. Claims above $2000 require executive approval
    
    Travel expenses are reimbursed according to the following rules:
    - Domestic flights: Economy class only
    - International flights: Business class for flights over 8 hours
    - Hotel accommodation: Up to $200 per night
    - Meals: Up to $75 per day
    
    The approval process typically takes 3-5 business days.
    All receipts must be submitted within 30 days of the expense.
    """
    
    return test_content.encode('utf-8')

def test_rag_endpoint():
    """Test the RAG system endpoint"""
    
    # Create test document
    test_doc = create_test_document()
    test_doc_b64 = base64.b64encode(test_doc).decode('utf-8')
    
    # Prepare test payload
    payload = {
        "query": "What is the approval limit for expense claims?",
        "documents": [
            {
                "filename": "company_policy.txt",
                "content": test_doc_b64
            }
        ]
    }
    
    # Test endpoint
    url = "http://localhost:5000/api/v1/hackrx/run"
    
    try:
        print("Testing RAG system...")
        print(f"Query: {payload['query']}")
        print(f"Document: {payload['documents'][0]['filename']}")
        print("-" * 50)
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response:")
            print(json.dumps(result, indent=2))
        else:
            print("ERROR! Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server. Make sure the Flask app is running.")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. The server might be processing.")
    except Exception as e:
        print(f"ERROR: {str(e)}")

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {str(e)}")

if __name__ == "__main__":
    print("Testing RAG System")
    print("=" * 50)
    
    # Test health first
    test_health_endpoint()
    print()
    
    # Test main functionality
    test_rag_endpoint()