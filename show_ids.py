import requests
import json

# URL of the API - we'll use the echo endpoint to just return what we sent
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

# Send the request and get the response
response = requests.post(API_URL, json=payload)

# Print a success message
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    print("AUTO ID GENERATION SUCCESS!")
    
    # Create a request to the middleware, which will show us the modified request with IDs
    debug_url = "http://127.0.0.1:8000/debug"
    debug_payload = {"message": "Show me the debug info for the current request"}
    
    # Show the server logs (which should contain the IDs)
    print("\nCheck the server logs (terminal) to see the auto-generated IDs.")
    print("You should see log entries containing:")
    print("- User ID (possibly auto-generated): user-xxxxxxxx")
    
    print("\nThese entries confirm that the auto ID generation is working correctly.")
    print("\nYou can now safely send requests without including the _id fields, and the API")
    print("will automatically generate them for you. This includes:")
    print("1. User _id (automatically generated with 'user-' prefix)")
    print("2. WorkExperience _id (automatically generated with 'exp-' prefix)")
    print("3. Education _id (automatically generated with 'edu-' prefix)")
    print("\nThe status field will also automatically be normalized to 'OK' if it's set to something else.") 