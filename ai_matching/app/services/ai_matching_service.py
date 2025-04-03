import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.models import User, Problem, AIMatchResult, MatchReason
from app.services.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMatchingService:
    """Service for AI-based matching between user profiles and job problems."""
    
    def __init__(self, use_openai_completion: bool = False):
        """
        Initialize the AI matching service.
        
        Args:
            use_openai_completion: Whether to use OpenAI for completions or just embeddings
        """
        self.use_openai_completion = use_openai_completion
        
        # Initialize embedding service (could use local model if OpenAI not available)
        self.embedding_service = EmbeddingService(use_openai=True)
        
        # Initialize TFIDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(max_features=100)
        
        if use_openai_completion:
            # Check if OpenAI API key is set
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found. Disabling completion features.")
                self.use_openai_completion = False
            else:
                openai.api_key = self.api_key
    
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
            "matched_items": [best_match] if matched else [],
            "similarity": similarity
        }
    
    def _calculate_experience_match(self, work_experience: List[Any], qualifications: str) -> Dict[str, Any]:
        """Evaluate if the user's experience matches the job qualifications."""
        if not work_experience:
            return {
                "matched": False,
                "score": 0.0,
                "details": "User has no work experience"
            }
        
        # Count years of experience
        total_years = 0
        experience_texts = []
        
        for exp in work_experience:
            try:
                # Convert to datetime objects if they're strings
                start = self._ensure_datetime(exp.startDate)
                end = self._ensure_datetime(exp.endDate) if exp.endDate else datetime.now()
                
                # Calculate duration
                duration = end - start
                years = duration.days / 365.25
                total_years += years
                
                # Build experience text for semantic matching
                experience_text = f"{exp.title} at {exp.company}"
                if exp.description:
                    experience_text += f": {exp.description}"
                experience_texts.append(experience_text)
            except Exception as e:
                logger.error(f"Error processing experience: {e}")
        
        # Qualifications text to compare against
        if not qualifications:
            qualifications = "Previous work experience in the field"
        
        # Calculate semantic match with qualifications
        semantic_scores = []
        for exp_text in experience_texts:
            similarity = self.embedding_service.calculate_text_similarity(exp_text, qualifications)
            semantic_scores.append(similarity)
        
        # Get the best semantic match
        best_semantic_score = max(semantic_scores) if semantic_scores else 0
        
        # Calculate scores based on different factors
        years_score = min(total_years / 5, 1.0)  # Cap at 5 years
        
        # Combine scores
        final_score = years_score * 0.6 + best_semantic_score * 0.4
        
        if final_score >= 0.7:
            details = f"User has strong relevant experience ({total_years:.1f} years)"
            matched = True
        elif final_score >= 0.5:
            details = f"User has moderate relevant experience ({total_years:.1f} years)"
            matched = True
        else:
            details = f"User's experience is insufficient or not relevant enough ({total_years:.1f} years)"
            matched = False
        
        return {
            "matched": matched,
            "score": final_score,
            "details": details,
            "years": total_years,
            "relevance": best_semantic_score
        }
    
    def _calculate_semantic_similarity(self, user: User, problem: Problem) -> float:
        """Calculate semantic similarity between user profile and job description."""
        # Create comprehensive text representations
        user_text = self._create_user_text(user)
        job_text = self._create_job_text(problem)
        
        # Calculate similarity using embeddings
        similarity = self.embedding_service.calculate_text_similarity(user_text, job_text)
        return similarity
    
    def _calculate_concept_match(self, user: User, problem: Problem) -> Dict[str, Any]:
        """Extract and compare key concepts from user profile and job description."""
        # Extract user concepts
        user_text = self._create_user_text(user)
        job_text = self._create_job_text(problem)
        
        # Create a combined corpus for TF-IDF
        corpus = [user_text, job_text]
        
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get top terms for job
            job_tfidf = tfidf_matrix[1].toarray()[0]
            job_indices = job_tfidf.argsort()[-20:][::-1]  # Top 20 terms
            top_job_terms = [feature_names[i] for i in job_indices if job_tfidf[i] > 0]
            
            # Get top terms for user
            user_tfidf = tfidf_matrix[0].toarray()[0]
            user_indices = user_tfidf.argsort()[-30:][::-1]  # Top 30 terms
            top_user_terms = [feature_names[i] for i in user_indices if user_tfidf[i] > 0]
            
            # Find common terms
            common_terms = [term for term in top_job_terms if term in top_user_terms]
            
            # Calculate score based on proportion of job terms matched
            if not top_job_terms:
                score = 0.0
            else:
                score = len(common_terms) / len(top_job_terms)
            
            details = f"Matched {len(common_terms)}/{len(top_job_terms)} key concepts"
            
            return {
                "matched": score >= 0.3,
                "score": score,
                "details": details,
                "matched_items": common_terms,
                "job_terms": top_job_terms,
                "user_terms": top_user_terms
            }
        except Exception as e:
            logger.error(f"Error in concept matching: {e}")
            return {
                "matched": False,
                "score": 0.0,
                "details": "Could not extract concepts",
                "matched_items": []
            }
    
    def _create_user_text(self, user: User) -> str:
        """Create a comprehensive text representation of the user profile."""
        lines = [
            f"Name: {user.firstName} {user.lastName}",
            f"Bio: {user.bio}",
            f"Skills: {', '.join(user.skills)}",
            f"Categories: {', '.join(user.categories)}",
            f"Specialities: {', '.join(user.specialities)}"
        ]
        
        # Add work experience
        for exp in user.workExperience:
            exp_text = f"Work: {exp.title} at {exp.company}"
            if exp.description:
                exp_text += f", {exp.description}"
            lines.append(exp_text)
        
        # Add education
        for edu in user.educationHistory:
            edu_text = f"Education: {edu.degree} in {edu.fieldOfStudy} from {edu.school}"
            if edu.description:
                edu_text += f", {edu.description}"
            lines.append(edu_text)
        
        return " ".join(lines)
    
    def _create_job_text(self, problem: Problem) -> str:
        """Create a comprehensive text representation of the job problem."""
        lines = [
            f"Field: {problem.fellowField}",
            f"Types: {', '.join(problem.type)}",
            f"Skills: {', '.join(problem.skills)}",
            f"Description: {problem.description}",
            f"Qualifications: {problem.candidatesQualification}",
            f"Nice to Have: {problem.niceToHaves}"
        ]
        return " ".join(lines)
    
    def _ensure_datetime(self, dt):
        """Convert string to datetime if needed and ensure it's timezone-naive for comparison."""
        try:
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            
            # Make timezone-naive if it's timezone-aware
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            
            return dt
        except Exception as e:
            logger.error(f"Error processing datetime: {e}")
            # Return current time as fallback
            return datetime.now()
    
    def _generate_improvement_suggestions(self, user: User, problem: Problem, 
                                          what_matched: List[str], went_against: List[str]) -> List[str]:
        """Generate suggestions for improving the match."""
        suggestions = []
        
        # Generate suggestions based on missing skills
        if "Skills" in went_against:
            missing_skills = [skill for skill in problem.skills if skill not in user.skills]
            if missing_skills:
                suggestions.append(f"Acquire skills in: {', '.join(missing_skills)}")
        
        # Generate suggestions based on field mismatch
        if "Field" in went_against:
            suggestions.append(f"Gain experience or certifications in {problem.fellowField}")
        
        # Generate suggestions based on experience
        if "Experience" in went_against:
            suggestions.append("Gain more relevant work experience in the field")
        
        # Generate suggestions based on key concepts
        if "Concepts" in went_against:
            suggestions.append("Focus on developing expertise in key areas mentioned in the job description")
        
        # If using OpenAI, get AI-generated suggestions
        if self.use_openai_completion and suggestions:
            try:
                ai_suggestions = self._get_openai_suggestions(user, problem)
                if ai_suggestions:
                    suggestions.extend(ai_suggestions)
            except Exception as e:
                logger.error(f"Error getting OpenAI suggestions: {e}")
        
        return suggestions
    
    def _get_openai_suggestions(self, user: User, problem: Problem) -> List[str]:
        """Get improvement suggestions using OpenAI."""
        try:
            prompt = f"""
            I need specific, actionable suggestions for how this candidate can improve their profile for this job.
            
            CANDIDATE PROFILE:
            Name: {user.firstName} {user.lastName}
            Skills: {', '.join(user.skills)}
            Categories: {', '.join(user.categories)}
            Specialities: {', '.join(user.specialities)}
            Work Experience: {[f"{exp.title} at {exp.company}" for exp in user.workExperience]}
            Education: {[f"{edu.degree} in {edu.fieldOfStudy} from {edu.school}" for edu in user.educationHistory]}
            Bio: {user.bio}
            
            JOB REQUIREMENTS:
            Field: {problem.fellowField}
            Type: {', '.join(problem.type)}
            Required Skills: {', '.join(problem.skills)}
            Description: {problem.description}
            Qualifications: {problem.candidatesQualification}
            Nice-to-Haves: {problem.niceToHaves}
            
            Provide 3-5 specific, actionable suggestions to help this candidate better match this job.
            Format each suggestion as a bullet point.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a career coach helping candidates improve their job match."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            suggestions_text = response.choices[0].message.content
            
            # Extract bullet points
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    suggestions.append(line[1:].strip())
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Error generating OpenAI suggestions: {e}")
            return [] 