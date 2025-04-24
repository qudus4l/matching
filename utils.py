def preprocess_match_data(user, problem):
    """
    Preprocess user and problem data to ensure it meets the API requirements.
    Transforms data to the correct format to prevent 422 errors.
    ID fields are now completely optional.
    """
    import re
    html_tag_pattern = re.compile(r'<\/?[^>]+(>|$)')
    
    # Process user work experience
    formatted_work_experience = []
    if user.get('workExperience'):
        for exp in user['workExperience']:
            work_exp = {
                'title': exp.get('title', 'Untitled Position'),
                'company': exp.get('company', 'Unknown Company'),
                'location': exp.get('location', 'Remote'),
                'startDate': exp.get('startDate') or '2020-01-01T00:00:00Z',
                'description': exp.get('description', '')
            }
            # Only add _id if it already exists
            if exp.get('_id'):
                work_exp['_id'] = exp['_id']
            if exp.get('endDate'):
                work_exp['endDate'] = exp['endDate']
            formatted_work_experience.append(work_exp)
    
    # Process user education
    formatted_education = []
    if user.get('educationHistory'):
        for edu in user['educationHistory']:
            education = {
                'school': edu.get('school', 'Unknown School'),
                'degree': edu.get('degree', 'Unknown Degree'),
                'fieldOfStudy': edu.get('fieldOfStudy', 'Unknown Field'),
                'startDate': edu.get('startDate') or '2018-01-01T00:00:00Z',
                'endDate': edu.get('endDate') or '2022-01-01T00:00:00Z',
                'description': edu.get('description', '')
            }
            # Only add _id if it already exists
            if edu.get('_id'):
                education['_id'] = edu['_id']
            formatted_education.append(education)
    
    # Process skills, categories, and specialities
    formatted_skills = []
    if user.get('skills'):
        for skill in user['skills']:
            if isinstance(skill, dict) and skill.get('name'):
                formatted_skills.append(skill['name'])
            elif isinstance(skill, str):
                formatted_skills.append(skill)
            else:
                formatted_skills.append(str(skill))
    
    formatted_categories = []
    if user.get('categories'):
        for category in user['categories']:
            if isinstance(category, dict) and category.get('name'):
                formatted_categories.append(category['name'])
            elif isinstance(category, str):
                formatted_categories.append(category)
            else:
                formatted_categories.append(str(category))
    
    formatted_specialities = []
    if user.get('specialities'):
        for speciality in user['specialities']:
            if isinstance(speciality, dict) and speciality.get('name'):
                formatted_specialities.append(speciality['name'])
            elif isinstance(speciality, str):
                formatted_specialities.append(speciality)
            else:
                formatted_specialities.append(str(speciality))
    
    # Format date of birth if present
    date_of_birth = None
    if user.get('dateOfBirth'):
        from datetime import datetime
        try:
            if isinstance(user['dateOfBirth'], str):
                date_of_birth = user['dateOfBirth']
            else:
                date_of_birth = user['dateOfBirth'].isoformat() if hasattr(user['dateOfBirth'], 'isoformat') else str(user['dateOfBirth'])
        except:
            date_of_birth = None

    # Create user object without _id
    user_obj = {
        'firstName': user.get('firstName', 'Anonymous'),
        'lastName': user.get('lastName', 'User'),
        'email': user.get('email', 'user@example.com'),
        'verified': user.get('verified', True),
        'active': user.get('active', True),
        'skills': formatted_skills,
        'categories': formatted_categories,
        'specialities': formatted_specialities,
        'categoriesAndSpecialitiesAdded': user.get('categoriesAndSpecialitiesAdded', True),
        'workExperience': formatted_work_experience,
        'educationHistory': formatted_education,
        'skillsAdded': user.get('skillsAdded', True),
        'workExperienceAdded': user.get('workExperienceAdded', True),
        'educationHistoryAdded': user.get('educationHistoryAdded', True),
        'bio': user.get('bio', ''),
        'bioAdded': user.get('bioAdded', True),
        'onboarded': user.get('onboarded', True),
        'profileCompletion': user.get('profileCompletion', 50),
        'address': user.get('address', ''),
        'city': user.get('city', ''),
        'country': user.get('country', ''),
        'dateOfBirth': date_of_birth,
        'phone': user.get('phone', ''),
        'state': user.get('state', ''),
        'photoUrl': user.get('photoUrl', ''),
        'userType': user.get('userType', 'user')
    }
    
    # Only add _id if it exists
    if user.get('_id'):
        user_obj['_id'] = user['_id']

    # Create problem object
    problem_obj = {
        'payRange': {
            'min': int(problem.get('payRange', {}).get('min', 0) or 0),
            'max': int(problem.get('payRange', {}).get('max', 0) or 0)
        },
        'fellowField': problem.get('fellowField', ''),
        'type': problem.get('type', []) if isinstance(problem.get('type', []), list) else [problem.get('type', '')],
        'skills': problem.get('skills', []) if isinstance(problem.get('skills', []), list) else [problem.get('skills', '')],
        'description': html_tag_pattern.sub('', problem.get('description', '') or ''),
        'candidatesQualification': html_tag_pattern.sub('', problem.get('candidatesQualification', '') or ''),
        'niceToHaves': html_tag_pattern.sub('', problem.get('niceToHaves', '') or '')
    }
    
    # Only add _id if it exists
    if problem.get('_id'):
        problem_obj['_id'] = problem['_id']
    
    # Create properly formatted payload
    formatted_payload = {
        'user_data': {
            'status': 'active',
            'user': user_obj
        },
        'problem': problem_obj
    }
    
    return formatted_payload 