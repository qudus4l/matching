import os
import logging
import json
import numpy as np
import openai
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from models.models import (
    User, BusinessData, JobDescription, Problem, AIMatchResult, 
    MatchReason, PayRange
)
from services.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedService:
    """Combined service for JD generation and AI matching."""
    
    def __init__(self, use_openai_completion: bool = True):
        """
        Initialize the unified service.
        
        Args:
            use_openai_completion: Whether to use OpenAI for completions
        """
        self.use_openai_completion = use_openai_completion
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(use_openai=True)
        
        # Initialize TFIDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(max_features=100)
        
        # Default salary range if OpenAI doesn't provide one
        self.default_salary = {"min": 5000, "max": 10000}
        
        # Check if OpenAI API key is set
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. Some features may be limited.")
            self.use_openai_completion = False
        else:
            openai.api_key = self.api_key
    
    # ===== JD GENERATION METHODS =====
    
    def generate_job_description(self, employer: BusinessData) -> JobDescription:
        """
        Generate a job description based on employer information using OpenAI.
        
        Args:
            employer: The employer business data
            
        Returns:
            JobDescription with all fields populated
        """
        logger.info(f"Generating job description for {employer.companyName} using OpenAI")
        
        # Create a prompt for OpenAI
        prompt = self._create_jd_prompt(employer)
        
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Can be upgraded to GPT-4 for better results
                messages=[
                    {"role": "system", "content": "You are a skilled HR professional who creates detailed job descriptions based on employer information. Create a structured job description in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the job description from OpenAI's response
            jd_text = response.choices[0].message.content.strip()
            
            # Parse JSON from OpenAI's response
            try:
                jd_data = json.loads(jd_text)
                
                # Create and return the job description
                job_description = self._convert_to_job_description(jd_data)
                logger.info(f"Successfully generated JD with OpenAI for {employer.companyName}")
                return job_description
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from OpenAI response: {jd_text}")
                return self._create_fallback_job_description(employer)
                
        except Exception as e:
            logger.error(f"Error using OpenAI to generate job description: {str(e)}")
            return self._create_fallback_job_description(employer)
    
    def _create_jd_prompt(self, employer: BusinessData) -> str:
        """Create a prompt for OpenAI based on employer data."""
        
        prompt = f"""
        Please create a detailed job description based on the following employer information:
        
        Company: {employer.companyName}
        Company Size: {employer.companySize}
        Number of Employees: {employer.noOfEmployees}
        Company Type: {employer.companyType}
        
        Job Category/Field: {', '.join(employer.categories) if employer.categories else 'Not specified'}
        
        Required Skills: {', '.join(employer.skills) if employer.skills else 'To be determined based on job needs'}
        
        Business Problems to Solve:
        {', '.join(employer.problems) if employer.problems else 'Not specified'}
        
        Business Data Types:
        {', '.join(employer.businessData) if employer.businessData else 'Not specified'}
        
        Challenges Faced:
        {', '.join(employer.challengesFaced) if employer.challengesFaced else 'Not specified'}
        
        Data Analysis Tools Used:
        {', '.join(employer.dataAnalysisTools) if employer.dataAnalysisTools else 'Not specified'}
        
        Expected Outcomes:
        {', '.join(employer.expectedOutcome) if employer.expectedOutcome else 'Not specified'}
        
        Please generate a complete job description in the following JSON format:
        {{
            "payRange": {{
                "min": [minimum salary as integer],
                "max": [maximum salary as integer]
            }},
            "fellowField": [main field name as string],
            "type": [array of 2-3 job types as strings],
            "skills": [array of 4-5 required skills as strings],
            "description": [job description text as string],
            "candidatesQualification": [qualifications text as string],
            "niceToHaves": [nice-to-have skills/experience text as string]
        }}
        
        Make sure all fields are present and properly formatted.
        """
        
        return prompt
    
    def _convert_to_job_description(self, jd_data: Dict[str, Any]) -> JobDescription:
        """Convert the OpenAI JSON response to a JobDescription object."""
        
        # Extract data with fallbacks
        pay_range_data = jd_data.get("payRange", self.default_salary)
        pay_range = PayRange(
            min=pay_range_data.get("min", 5000),
            max=pay_range_data.get("max", 10000)
        )
        
        fellow_field = jd_data.get("fellowField", "Data Analysis")
        
        # Ensure we have at least 1 job type
        job_types = jd_data.get("type", ["Data Analysis"])
        if not job_types or not isinstance(job_types, list):
            job_types = ["Data Analysis"]
            
        # Ensure we have at least 3 skills
        skills = jd_data.get("skills", ["Analytical Thinking", "Problem Solving", "Communication"])
        if not skills or not isinstance(skills, list) or len(skills) < 3:
            skills.extend(["Analytical Thinking", "Problem Solving", "Communication"])
            skills = list(set(skills))[:5]  # Ensure no duplicates and limit to 5
            
        description = jd_data.get("description", "Job description not provided.")
        qualifications = jd_data.get("candidatesQualification", "Qualifications not provided.")
        nice_to_haves = jd_data.get("niceToHaves", "Experience with relevant tools and technologies.")
        
        return JobDescription(
            payRange=pay_range,
            fellowField=fellow_field,
            type=job_types,
            skills=skills,
            description=description,
            candidatesQualification=qualifications,
            niceToHaves=nice_to_haves
        )
    
    def _create_fallback_job_description(self, employer: BusinessData) -> JobDescription:
        """Create a fallback job description if OpenAI fails."""
        logger.info(f"Using fallback JD generation for {employer.companyName}")
        
        # Use whatever information we have from the employer
        field = employer.categories[0] if employer.categories else "Data Analysis"
        skills = employer.skills if employer.skills and len(employer.skills) >= 3 else ["Analytical Thinking", "Problem Solving", "Communication", "Data Analysis"]
        
        description = f"We are looking for a skilled professional to help with data analysis and business challenges at {employer.companyName}."
        if employer.problems:
            description += f" Key problems to solve include {' and '.join(employer.problems).lower()}."
            
        qualifications = f"Experience in {field} or a related field. Strong technical and analytical skills."
        
        nice_to_haves = "Experience with data analysis tools and business intelligence platforms."
        if employer.dataAnalysisTools:
            nice_to_haves = f"Experience with {', '.join(employer.dataAnalysisTools)}."
        
        return JobDescription(
            payRange=PayRange(min=5000, max=10000),
            fellowField=field,
            type=["Data Analysis", "Business Intelligence"],
            skills=skills[:5],
            description=description,
            candidatesQualification=qualifications,
            niceToHaves=nice_to_haves
        )
    
    # ===== AI MATCHING METHODS =====
    
    def calculate_match(self, user: User, problem: Problem) -> AIMatchResult:
        """
        Calculate advanced AI-based match between user and job problem.
        
        Args:
            user: User profile
            problem: Job problem
            
        Returns:
            AIMatchResult with detailed match information
        """
        logger.info(f"Starting AI match calculation for user: {user.firstName} {user.lastName}")
        
        # Calculate different match components
        skills_match = self._calculate_skills_match(user.skills, problem.skills)
        logger.info(f"Skills match: {skills_match}")
        
        field_match = self._calculate_field_match(user.categories, user.specialities, problem.fellowField)
        logger.info(f"Field match: {field_match}")
        
        experience_match = self._calculate_experience_match(user.workExperience, problem.candidatesQualification)
        logger.info(f"Experience match: {experience_match}")
        
        # Calculate semantic similarity between user profile and job description
        semantic_similarity = self._calculate_semantic_similarity(user, problem)
        logger.info(f"Semantic similarity: {semantic_similarity:.4f}")
        
        # Extract key concepts match
        concept_match = self._calculate_concept_match(user, problem)
        logger.info(f"Concept match: {concept_match}")
        
        # Calculate compatibility by area
        compatibility_by_area = {
            "Skills": skills_match["score"],
            "Field": field_match["score"],
            "Experience": experience_match["score"],
            "Semantic": semantic_similarity,
            "Concepts": concept_match["score"]
        }
        
        # Determine overall match score with weighted components
        weights = {
            "Skills": 0.25,
            "Field": 0.15,
            "Experience": 0.20,
            "Semantic": 0.25,
            "Concepts": 0.15
        }
        
        weighted_scores = [score * weights[area] for area, score in compatibility_by_area.items()]
        overall_match = sum(weighted_scores) * 100
        
        # Cap at 100 and convert to integer
        percentage_match = min(int(round(overall_match)), 100)
        
        # Determine what matched and what went against
        match_threshold = 0.6  # Threshold to consider a category as matched
        match_results = {
            category: score >= match_threshold 
            for category, score in compatibility_by_area.items()
        }
        
        what_matched = [category for category, matched in match_results.items() if matched]
        went_against = [category for category, matched in match_results.items() if not matched]
        
        # Create detailed reasoning
        detailed_reasoning = [
            MatchReason(
                feature="Skills",
                score=skills_match["score"],
                explanation=skills_match["details"],
                details={"matched_skills": skills_match.get("matched_items", [])}
            ),
            MatchReason(
                feature="Field",
                score=field_match["score"],
                explanation=field_match["details"],
                details={"matched_fields": field_match.get("matched_items", [])}
            ),
            MatchReason(
                feature="Experience",
                score=experience_match["score"],
                explanation=experience_match["details"]
            ),
            MatchReason(
                feature="Semantic Similarity",
                score=semantic_similarity,
                explanation=f"Semantic similarity between profile and job: {semantic_similarity:.2f}"
            ),
            MatchReason(
                feature="Key Concepts",
                score=concept_match["score"],
                explanation=concept_match["details"],
                details={"matched_concepts": concept_match.get("matched_items", [])}
            )
        ]
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(user, problem, what_matched, went_against)
        
        return AIMatchResult(
            percentage_match=percentage_match,
            what_matched=what_matched,
            went_against=went_against,
            detailed_reasoning=detailed_reasoning,
            suggested_improvements=suggestions,
            semantic_similarity=semantic_similarity,
            compatibility_by_area=compatibility_by_area
        )
    
    def _calculate_skills_match(self, user_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
        """Calculate sophisticated skills match using embeddings."""
        if not required_skills:
            return {
                "matched": True,
                "score": 1.0,
                "details": "No specific skills required",
                "matched_items": []
            }
        
        # Direct match (exact string matching)
        exact_matches = [skill for skill in required_skills if skill in user_skills]
        
        # Semantic match (for skills that aren't exact matches)
        missing_skills = [skill for skill in required_skills if skill not in exact_matches]
        semantic_matches = []
        
        for missing_skill in missing_skills:
            # Find the most semantically similar user skill for each missing skill
            if user_skills:
                result = self.embedding_service.find_most_similar(missing_skill, user_skills)
                if result["score"] > 0.8:  # High similarity threshold
                    semantic_matches.append({
                        "required": missing_skill,
                        "user_skill": result["text"],
                        "similarity": result["score"]
                    })
        
        # Calculate overall score
        exact_match_score = len(exact_matches) / len(required_skills) if required_skills else 1.0
        semantic_match_contribution = sum(match["similarity"] for match in semantic_matches) / len(required_skills) if required_skills else 0
        
        # Combine scores (weighting exact matches higher)
        overall_score = exact_match_score * 0.7 + semantic_match_contribution * 0.3
        
        # Create details
        all_matched = exact_matches + [match["user_skill"] for match in semantic_matches]
        details = f"Matched {len(all_matched)}/{len(required_skills)} required skills"
        
        return {
            "matched": overall_score >= 0.5,
            "score": overall_score,
            "details": details,
            "matched_items": all_matched,
            "exact_matches": exact_matches,
            "semantic_matches": semantic_matches
        }
    
    def _calculate_field_match(self, categories: List[str], specialities: List[str], required_field: str) -> Dict[str, Any]:
        """Calculate field match using embeddings for semantic similarity."""
        all_user_fields = categories + specialities
        
        if not required_field:
            return {
                "matched": True,
                "score": 1.0,
                "details": "No specific field required",
                "matched_items": []
            }
        
        if not all_user_fields:
            return {
                "matched": False,
                "score": 0.0,
                "details": "User has no fields or specialities",
                "matched_items": []
            }
        
        # Check for exact match
        if required_field in all_user_fields:
            return {
                "matched": True,
                "score": 1.0,
                "details": f"User has the exact required field: {required_field}",
                "matched_items": [required_field]
            }
        
        # Find the most similar field
        result = self.embedding_service.find_most_similar(required_field, all_user_fields)
        best_match = result["text"]
        similarity = result["score"]
        
        if similarity >= 0.85:
            score = similarity
            details = f"User has a very similar field: {best_match} (similarity: {similarity:.2f})"
            matched = True
        elif similarity >= 0.7:
            score = similarity * 0.8  # Reduce score for moderate matches
            details = f"User has a related field: {best_match} (similarity: {similarity:.2f})"
            matched = True
        else:
            score = similarity * 0.4  # Further reduce score for weak matches
            details = f"User's best field match is weak: {best_match} (similarity: {similarity:.2f})"
            matched = False
        
        return {
            "matched": matched,
            "score": score,
            "details": details,
            "matched_items": [best_match]
        }
    
    def _calculate_experience_match(self, work_experience: List[Any], qualifications: str) -> Dict[str, Any]:
        """Calculate experience match based on job qualifications and user's work history."""
        if not work_experience:
            return {
                "matched": False,
                "score": 0.0,
                "details": "User has no work experience"
            }
        
        # Extract total years of experience
        total_years = 0
        for exp in work_experience:
            # Calculate duration for each job
            start_date = self._ensure_datetime(exp.startDate)
            
            if exp.endDate:
                end_date = self._ensure_datetime(exp.endDate)
            else:
                # If no end date, assume current job (use present date)
                import datetime
                end_date = datetime.datetime.now(datetime.timezone.utc)
            
            # Add to total years
            duration = end_date - start_date
            total_years += duration.days / 365.25  # Account for leap years
        
        # Determine experience level
        if total_years >= 8:
            experience_level = "senior"
            score = 1.0
        elif total_years >= 4:
            experience_level = "mid-level"
            score = 0.8
        elif total_years >= 2:
            experience_level = "junior"
            score = 0.6
        elif total_years > 0:
            experience_level = "entry-level"
            score = 0.4
        else:
            experience_level = "no experience"
            score = 0.0
        
        # Determine if this meets job requirements
        # Look for years mentioned in qualifications text
        import re
        years_pattern = r'(\d+)[\+\-]?\s*(?:years|yrs)'
        years_match = re.search(years_pattern, qualifications, re.IGNORECASE)
        
        if years_match:
            required_years = int(years_match.group(1))
            
            if total_years >= required_years:
                matched = True
                details = f"User has {total_years:.1f} years experience, meeting the required {required_years}+ years"
                score = min(1.0, total_years / required_years)  # Cap at 1.0
            else:
                matched = False
                details = f"User has {total_years:.1f} years experience, less than the required {required_years}+ years"
                score = max(0.0, total_years / required_years)  # Partial credit
        else:
            # No specific year requirement found, use general assessment
            if "senior" in qualifications.lower() and experience_level != "senior":
                matched = False
                details = f"Job requires senior experience but user has {total_years:.1f} years ({experience_level})"
            elif "mid" in qualifications.lower() and total_years < 4:
                matched = False
                details = f"Job requires mid-level experience but user has {total_years:.1f} years ({experience_level})"
            else:
                matched = True
                details = f"User's experience level ({experience_level}, {total_years:.1f} years) appears sufficient"
        
        return {
            "matched": matched,
            "score": score,
            "details": details
        }
    
    def _calculate_semantic_similarity(self, user: User, problem: Problem) -> float:
        """Calculate semantic similarity between user profile and job description."""
        user_text = self._create_user_text(user)
        job_text = self._create_job_text(problem)
        
        return self.embedding_service.calculate_text_similarity(user_text, job_text)
    
    def _calculate_concept_match(self, user: User, problem: Problem) -> Dict[str, Any]:
        """Calculate match based on key concepts in both user profile and job."""
        # Create combined texts
        user_text = self._create_user_text(user)
        job_text = self._create_job_text(problem)
        
        # Extract key concepts using TFIDF
        try:
            # Fit TFIDF on combined corpus
            corpus = [user_text, job_text]
            self.tfidf.fit(corpus)
            
            # Get top terms for job
            job_vector = self.tfidf.transform([job_text]).toarray()[0]
            job_indices = job_vector.argsort()[-10:][::-1]  # Top 10 terms
            feature_names = self.tfidf.get_feature_names_out()
            job_terms = [feature_names[i] for i in job_indices if job_vector[i] > 0]
            
            # Get top terms for user
            user_vector = self.tfidf.transform([user_text]).toarray()[0]
            user_indices = user_vector.argsort()[-20:][::-1]  # Top 20 terms
            user_terms = [feature_names[i] for i in user_indices if user_vector[i] > 0]
            
            # Find common concepts
            common_terms = [term for term in job_terms if term in user_terms]
            
            # Calculate score
            score = len(common_terms) / len(job_terms) if job_terms else 0.0
            score = min(1.0, score)  # Cap at 1.0
            
            details = f"Found {len(common_terms)} common key concepts out of {len(job_terms)} important job concepts"
            
            return {
                "matched": score >= 0.3,  # Consider it a match if at least 30% of concepts match
                "score": score,
                "details": details,
                "matched_items": common_terms
            }
        except Exception as e:
            logger.error(f"Error calculating concept match: {str(e)}")
            return {
                "matched": False,
                "score": 0.0,
                "details": "Could not calculate concept match",
                "matched_items": []
            }
    
    def _create_user_text(self, user: User) -> str:
        """Create a comprehensive text representation of a user profile."""
        parts = []
        
        # Personal info
        parts.append(f"Name: {user.firstName} {user.lastName}")
        if user.bio:
            parts.append(f"Bio: {user.bio}")
        
        # Skills, categories and specialties
        if user.skills:
            parts.append(f"Skills: {', '.join(user.skills)}")
        if user.categories:
            parts.append(f"Fields: {', '.join(user.categories)}")
        if user.specialities:
            parts.append(f"Specialties: {', '.join(user.specialities)}")
        
        # Work experience
        if user.workExperience:
            parts.append("Work Experience:")
            for exp in user.workExperience:
                exp_text = f"- {exp.title} at {exp.company}"
                if exp.description:
                    exp_text += f": {exp.description}"
                parts.append(exp_text)
        
        # Education
        if user.educationHistory:
            parts.append("Education:")
            for edu in user.educationHistory:
                parts.append(f"- {edu.degree} in {edu.fieldOfStudy} from {edu.school}")
        
        return "\n".join(parts)
    
    def _create_job_text(self, problem: Problem) -> str:
        """Create a comprehensive text representation of a job description."""
        parts = [
            f"Field: {problem.fellowField}",
            f"Job Types: {', '.join(problem.type)}",
            f"Required Skills: {', '.join(problem.skills)}",
            f"Description: {problem.description}",
            f"Qualifications: {problem.candidatesQualification}",
            f"Nice-to-Have: {problem.niceToHaves}"
        ]
        return "\n".join(parts)
    
    def _ensure_datetime(self, dt):
        """Ensure a value is a datetime."""
        from datetime import datetime, timezone
        
        if isinstance(dt, str):
            try:
                # Convert ISO format string to timezone-aware datetime
                return datetime.fromisoformat(dt.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
            except:
                try:
                    # Try parsing with different format and set UTC timezone
                    return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                except:
                    # Return a default datetime with UTC timezone
                    return datetime(2000, 1, 1, tzinfo=timezone.utc)
        
        # If it's already a datetime, ensure it has timezone info
        if isinstance(dt, datetime) and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
            
        return dt
    
    def _generate_improvement_suggestions(self, user: User, problem: Problem, 
                                     what_matched: List[str], went_against: List[str]) -> List[str]:
        """Generate suggestions for improving match."""
        suggestions = []
        
        # Skills suggestions
        if "Skills" in went_against:
            missing_skills = [skill for skill in problem.skills if skill not in user.skills]
            if missing_skills:
                suggestions.append(f"Consider adding these skills to your profile: {', '.join(missing_skills[:3])}")
        
        # Field suggestions
        if "Field" in went_against:
            if problem.fellowField and problem.fellowField not in user.categories + user.specialities:
                suggestions.append(f"Consider adding '{problem.fellowField}' to your fields or specialties")
        
        # Experience suggestions
        if "Experience" in went_against:
            suggestions.append("Add more details to your work experience that relate to the job requirements")
        
        # If OpenAI completion is enabled, get more detailed suggestions
        if self.use_openai_completion and self.api_key:
            try:
                openai_suggestions = self._get_openai_suggestions(user, problem)
                suggestions.extend(openai_suggestions)
            except Exception as e:
                logger.error(f"Error getting OpenAI suggestions: {str(e)}")
        
        return suggestions
    
    def _get_openai_suggestions(self, user: User, problem: Problem) -> List[str]:
        """Get improvement suggestions using OpenAI."""
        try:
            user_text = self._create_user_text(user)
            job_text = self._create_job_text(problem)
            
            prompt = f"""
            I need suggestions for how this job candidate could improve their profile to better match a job.
            
            CANDIDATE PROFILE:
            {user_text}
            
            JOB DESCRIPTION:
            {job_text}
            
            Provide 2-3 specific, actionable suggestions for how this candidate could improve their profile 
            to better match this specific job. Each suggestion should be concise (1-2 sentences).
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a career advisor helping job seekers improve their profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # Extract and process suggestions
            suggestion_text = response.choices[0].message.content.strip()
            raw_suggestions = suggestion_text.split('\n')
            
            # Clean up suggestions (remove numbering, etc.)
            import re
            cleaned_suggestions = []
            for suggestion in raw_suggestions:
                # Remove numbering like "1.", "•", "-", etc.
                cleaned = re.sub(r'^[\d\.\-•\*\s]+', '', suggestion).strip()
                if cleaned and len(cleaned) > 10:  # Ensure it's not an empty or very short line
                    cleaned_suggestions.append(cleaned)
            
            return cleaned_suggestions[:3]  # Return at most 3 suggestions
            
        except Exception as e:
            logger.error(f"Error in OpenAI suggestion generation: {str(e)}")
            return ["Add relevant skills and experience to your profile to improve match."] 