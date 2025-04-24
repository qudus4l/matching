from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, root_validator
from datetime import datetime
import uuid

class WorkExperience(BaseModel):
    title: str
    company: str
    location: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: Optional[str] = None
    _id: Optional[str] = None  # Changed to Optional with default set to None

    @root_validator(pre=True)
    def set_id_if_not_provided(cls, values):
        if "_id" not in values or values["_id"] is None:
            values["_id"] = f"exp-{str(uuid.uuid4())[:8]}"
        return values

class Education(BaseModel):
    school: str
    degree: str
    fieldOfStudy: str
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None
    _id: Optional[str] = None  # Changed to Optional with default set to None

    @root_validator(pre=True)
    def set_id_if_not_provided(cls, values):
        if "_id" not in values or values["_id"] is None:
            values["_id"] = f"edu-{str(uuid.uuid4())[:8]}"
        return values

class User(BaseModel):
    _id: Optional[str] = None  # Changed to Optional with default set to None
    firstName: str
    lastName: str
    email: str
    verified: bool
    active: bool
    skills: List[str]
    categories: List[str]
    specialities: List[str]
    categoriesAndSpecialitiesAdded: bool
    workExperience: List[WorkExperience]
    educationHistory: List[Education]
    skillsAdded: bool
    workExperienceAdded: bool
    educationHistoryAdded: bool
    bio: str
    bioAdded: bool
    onboarded: bool
    profileCompletion: int
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    dateOfBirth: Optional[datetime] = None
    phone: Optional[str] = None
    state: Optional[str] = None
    photoUrl: Optional[str] = None
    userType: str

    @root_validator(pre=True)
    def set_id_if_not_provided(cls, values):
        if "_id" not in values or values["_id"] is None:
            values["_id"] = f"user-{str(uuid.uuid4())[:8]}"
        return values

class UserResponse(BaseModel):
    status: str
    user: User

    @root_validator(pre=True)
    def normalize_status(cls, values):
        # Ensure status is set to "OK" if it's something else
        if "status" in values and values["status"] != "OK":
            values["status"] = "OK"
        return values

class BusinessData(BaseModel):
    companyName: str
    fullName: str
    email: str
    skills: List[str]
    categories: List[str]
    problems: List[str]
    companyType: str
    companySize: str
    noOfEmployees: int
    businessData: List[str]
    challengesFaced: List[str]
    dataAnalysisTools: List[str]
    expectedOutcome: List[str]
    
class EmployerResponse(BaseModel):
    status: str
    user: BusinessData

class PayRange(BaseModel):
    min: int
    max: int

class JobDescription(BaseModel):
    payRange: PayRange
    fellowField: str
    type: List[str]
    skills: List[str]
    description: str
    candidatesQualification: str
    niceToHaves: str
    _id: Optional[str] = None  # Added to make compatible with Problem

class JobDescriptionResponse(BaseModel):
    problems: List[JobDescription]
    
# For compatibility with AI Matching API
Problem = JobDescription

class MatchInput(BaseModel):
    user_data: UserResponse
    problem: Problem

class MatchReason(BaseModel):
    feature: str
    score: float
    explanation: str
    details: Optional[Dict[str, Any]] = None

class AIMatchResult(BaseModel):
    percentage_match: int = Field(..., description="Overall match percentage")
    what_matched: List[str] = Field(..., description="Categories that matched")
    went_against: List[str] = Field(..., description="Categories that didn't match")
    detailed_reasoning: List[MatchReason] = Field(..., description="Detailed reasoning for each match component")
    suggested_improvements: List[str] = Field(..., description="Suggestions for improving match")
    semantic_similarity: float = Field(..., description="Semantic similarity score between user and job")
    compatibility_by_area: Dict[str, float] = Field(..., description="Compatibility scores by area")

class SimpleMatchResult(BaseModel):
    percentage_match: int = Field(..., description="Overall match percentage")
    what_matched: List[str] = Field(..., description="Categories that matched")
    went_against: List[str] = Field(..., description="Categories that didn't match") 