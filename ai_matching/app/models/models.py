from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

class WorkExperience(BaseModel):
    title: str
    company: str
    location: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: Optional[str] = None
    _id: str

class Education(BaseModel):
    school: str
    degree: str
    fieldOfStudy: str
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None
    _id: str

class BookmarkedJob(BaseModel):
    _id: str
    title: str
    company: str
    location: str
    salary: str
    applicationDeadline: datetime
    description: str
    postedDate: datetime
    type: str
    workArrangement: str
    status: str
    noOfApplicants: int

class User(BaseModel):
    _id: str
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
    bookmarkedJobs: Optional[List[BookmarkedJob]] = None
    userType: str

class UserResponse(BaseModel):
    status: str
    user: User

class PayRange(BaseModel):
    min: int
    max: int

class Problem(BaseModel):
    payRange: PayRange
    _id: str
    fellowField: str
    type: List[str]
    skills: List[str]
    description: str
    candidatesQualification: str
    niceToHaves: str

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