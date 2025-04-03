#!/usr/bin/env python3
import json
import requests
import sys

# Sample user data
user_data = {
    "status": "OK",
    "user": {
        "_id": "6057cfd8cc42f31212222111",
        "firstName": "Jane",
        "lastName": "Smith",
        "email": "jane.smith@example.com",
        "categories": ["Quantitative Analyst (Quant)"],
        "specialities": ["Statistical Analysis"],
        "level": "Mid-level",
        "skills": [
            "Python",
            "R",
            "SQL",
            "Statistical Analysis",
            "Financial Modeling",
            "Risk Management",
            "ETL",
            "Data Visualization",
            "Machine Learning",
            "Business Intelligence"
        ],
        "workExperience": [
            {
                "company": "Goldman Sachs",
                "title": "Associate Quantitative Analyst",
                "description": "Developed financial models and conducted risk assessments for trading strategies.",
                "startDate": "2022-01-01",
                "endDate": "2023-05-01",
                "location": "New York, NY"
            }
        ],
        # Required fields based on validation errors
        "verified": True,
        "active": True,
        "categoriesAndSpecialitiesAdded": True,
        "educationHistory": [],
        "skillsAdded": True,
        "workExperienceAdded": True,
        "educationHistoryAdded": True, 
        "bio": "Experienced quantitative analyst with a background in financial modeling and risk assessment.",
        "bioAdded": True,
        "onboarded": True,
        "profileCompletion": 100,
        "userType": "fellow"
    }
}

# Sample problem data
problem_data = {
    "payRange": {
        "min": 5000,
        "max": 10000
    },
    "_id": "67e3a4d9e2e076e069d7dc8a",
    "fellowField": "Quantitative Analysis",
    "type": [
        "Financial Modeling",
        "Risk Assessment"
    ],
    "skills": [
        "Structured Thinking",
        "SQL & Databases",
        "Business Acumen",
        "ETL"
    ],
    "description": "We need a solution to address inaccurate financial risk assessments and difficulties in modeling complex financial instruments.",
    "candidatesQualification": "Experience in financial modeling and risk assessment. Strong SQL and database skills.",
    "niceToHaves": "Experience with quantitative analysis tools and business intelligence software."
}

# Create the full request payload
data = {
    "user_data": user_data,
    "problem": problem_data
}

# Send request to the API
url = "http://localhost:8000/api/v1/ai-match"
headers = {"Content-Type": "application/json"}

try:
    print("Calling AI matching API...")
    response = requests.post(url, json=data, headers=headers, timeout=30)
    
    # Print response
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\n=== AI MATCH RESULT ===")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error occurred: {str(e)}") 