import requests
import json

# URL of the API
API_URL = "http://127.0.0.1:8000/api/v1/ai-match/detailed"

# Test payload with missing _id fields
payload = {
  "user_data": {
    "status": "active",
    "user": {
      # No _id field - will be automatically generated
      "firstName": "Godwin",
      "lastName": "Ego",
      "email": "godwinehile@gmail.com",
      "verified": True,
      "active": True,
      "skills": ["JavaScript", "TypeScript", "Python", "Web Development"],
      "categories": ["Software Engineering"],
      "specialities": ["Web Development"],
      "categoriesAndSpecialitiesAdded": True,
      "workExperience": [
        {
          # No _id field - will be automatically generated
          "title": "Software Engineer",
          "company": "Example Company",
          "location": "Lagos, Nigeria",
          "startDate": "2018-01-01T00:00:00.000Z",
          "endDate": "2023-01-01T00:00:00.000Z",
          "description": "Developed scalable web applications"
        }
      ],
      "educationHistory": [
        {
          # No _id field - will be automatically generated
          "school": "Obafemi Awolowo University",
          "degree": "Bachelor's",
          "fieldOfStudy": "Computer Science",
          "startDate": "2010-09-01T00:00:00.000Z",
          "endDate": "2014-07-30T00:00:00.000Z",
          "description": "Studied computer science and software engineering"
        }
      ],
      "skillsAdded": True,
      "workExperienceAdded": True,
      "educationHistoryAdded": True,
      "bio": "I am a passionate software engineer...",
      "bioAdded": True,
      "onboarded": True,
      "profileCompletion": 100,
      "userType": "user",
      "address": "Obafemi Awolowo University",
      "city": "Lagos",
      "country": "Nigeria",
      "dateOfBirth": "1990-01-01T00:00:00.000Z",
      "phone": "+2348012345678",
      "state": "Osun",
      "photoUrl": "https://example.com/photo.png"
    }
  },
  "problem": {
    "payRange": {"min": 300, "max": 2000},
    "fellowField": "Data Analyst",
    "type": ["Data Analyst", "Marketing Analyst"],
    "skills": ["Data Analysis", "SQL", "Excel", "Data Visualization"],
    "description": "general data",
    "candidatesQualification": "general data",
    "niceToHaves": "general data"
  }
}

# Use the detailed endpoint to get more data back
response = requests.post(API_URL, json=payload)

# Extract the actual input data from the logs on the server
print(f"Request sent to server. Check server logs for the actual user object with IDs")

if response.status_code == 200:
    print("Request succeeded! Auto-generated IDs are working.")
    
    # Ask user to check the server logs to confirm IDs are generated
    print("\nThe server logs should contain something like:")
    print("- User ID (possibly auto-generated): user-xxxxxxxx")
    print("- Work experience should have an ID like exp-xxxxxxxx")
    print("- Education entries should have an ID like edu-xxxxxxxx")
    
    print("\nIf you see these prefixes in the logs, auto ID generation is working correctly!") 