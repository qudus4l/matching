from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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

class JobDescriptionResponse(BaseModel):
    problem: JobDescription 