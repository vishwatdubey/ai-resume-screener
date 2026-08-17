import io
import os
import json
import logging
import re
import tempfile
from typing import List, Optional, Dict, Any, Set, Tuple
import uuid
from datetime import datetime

import pdfplumber
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import PyPDF2
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.api.dependencies import get_db
from app.services.ollama_service import OllamaClient
from app.db import models
from app.schemas import candidate as schemas

logger = logging.getLogger(__name__)

router = APIRouter()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    try:
        # Debug: Log file size and first 8 bytes
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
            logger.info(f"[PDF DEBUG] File: {file_path}, Size: {len(file_bytes)} bytes, First 8 bytes: {file_bytes[:8]}")
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                # Try different extraction methods
                page_text = page.extract_text()
                if not page_text:
                    # Fallback to raw text extraction
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if not page_text:
                    # Try with layout preservation
                    page_text = page.extract_text(layout=True)
                
                if page_text:
                    text += page_text + "\n\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing PDF: {str(e)}"
        )

def clean_text(text: str) -> str:
    """Clean and format the extracted text."""
    if not text:
        return ""
        
    # Split into lines and clean each line
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Fix common formatting issues
        line = line.replace('|', ' | ')
        line = line.replace('•', '• ')
        line = line.replace(':', ': ')
        
        # Smart word separation
        # 1. Handle camelCase and PascalCase
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        
        # 2. Handle consecutive capital letters (acronyms)
        line = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', line)
        
        # 3. Handle numbers
        line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', line)
        line = re.sub(r'([A-Za-z])(\d)', r'\1 \2', line)
        
        # 4. Handle special characters
        line = re.sub(r'([A-Za-z])([+\-–—])', r'\1 \2', line)
        line = re.sub(r'([+\-–—])([A-Za-z])', r'\1 \2', line)
        
        # 5. Handle common punctuation
        line = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', line)
        line = re.sub(r'([,;])([A-Za-z])', r'\1 \2', line)
        
        # 6. Fix multiple spaces and clean up
        line = ' '.join(line.split())
        
        # 7. Fix specific cases
        line = line.replace(' .', '.')
        line = line.replace(' ,', ',')
        line = line.replace('  ', ' ')
        
        # 8. Preserve certain patterns
        # Keep email addresses intact
        line = re.sub(r'(\S+)@(\S+)', r'\1@\2', line)
        # Keep URLs intact
        line = re.sub(r'(https?://\S+)', r'\1', line)
        # Keep phone numbers intact
        line = re.sub(r'(\+\d{1,3}[- ]?\d{1,4}[- ]?\d{1,4}[- ]?\d{1,9})', r'\1', line)
        
        if line.strip():
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

class FileNamesRequest(BaseModel):
    file_names: List[str]

class JobMatchRequest(BaseModel):
    job_description: str
    budget: float
    candidate_emails: List[str]

class CandidateCreate(BaseModel):
    name: str
    email: str
    current_ctc: float
    expected_ctc: float
    resume_text: Optional[str] = None

@router.get("/test")
def test_candidates():
    return {"message": "Candidates route is working"}

@router.get("/candidates")
def get_candidates(db: Session = Depends(get_db)):
    """Fetch all candidates from the database."""
    candidates = db.query(models.Candidate).all()
    return [
        {
            "email": c.email,
            "name": c.name,
            "current_ctc": c.current_ctc,
            "expected_ctc": c.expected_ctc,
            "resume_path": c.resume_path,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in candidates
    ]

@router.post("/rank")
async def rank_resume(job_description: str, resume1: UploadFile = File(...), resume2: UploadFile = File(...)):
    try:
        # Read PDF files into bytes
        resume1_bytes = await resume1.read()
        resume2_bytes = await resume2.read()
        logger.info("PDF files read successfully.")

        # Extract text and layout information from PDFs
        with pdfplumber.open(io.BytesIO(resume1_bytes)) as pdf:
            resume1_text = ""
            for page in pdf.pages:
                resume1_text += page.extract_text() + "\n"
        logger.info("Text extracted from resume1.")

        with pdfplumber.open(io.BytesIO(resume2_bytes)) as pdf:
            resume2_text = ""
            for page in pdf.pages:
                resume2_text += page.extract_text() + "\n"
        logger.info("Text extracted from resume2.")

        # Initialize Ollama client
        ollama_client = OllamaClient()
        logger.info("Ollama client initialized.")
        
        # Rank resumes using Ollama
        analysis = ollama_client.rank_resumes(job_description, [resume1_text, resume2_text])
        logger.info("Resume ranking completed with Ollama.")
        
        # If Ollama returned error or empty response, return appropriate message
        if not analysis or "error" in analysis:
            return {
                "status": "error",
                "message": analysis.get("error") if "error" in analysis else "Failed to analyze resumes"
            }

        return {
            "status": "success",
            "analysis": analysis
        }
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@router.post("/process-files")
async def process_files(request: FileNamesRequest):
    try:
        downloads_dir = os.path.expanduser("~/Downloads")
        results = []
        
        for file_name in request.file_names:
            file_path = os.path.join(downloads_dir, file_name)
            
            if not os.path.exists(file_path):
                results.append({
                    "file_name": file_name,
                    "status": "error",
                    "message": "File not found"
                })
                continue
                
            try:
                with open(file_path, 'rb') as file:
                    file_bytes = file.read()
                    
                # Enhanced text extraction from PDF
                text = ""
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page in pdf.pages:
                            # Try different extraction methods
                            page_text = page.extract_text()
                            if not page_text:
                                # Fallback to raw text extraction
                                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                            if not page_text:
                                # Try with layout preservation
                                page_text = page.extract_text(layout=True)
                            
                            # Clean up the text
                            if page_text:
                                # Split into lines and clean each line
                                lines = page_text.split('\n')
                                cleaned_lines = []
                                for line in lines:
                                    # Fix common formatting issues first
                                    line = line.replace('|', ' | ')
                                    line = line.replace('•', '• ')
                                    line = line.replace(':', ': ')
                                    
                                    # Add spaces between words that are stuck together
                                    import re
                                    
                                    # Smart word separation
                                    # 1. Handle camelCase and PascalCase
                                    line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
                                    
                                    # 2. Handle consecutive capital letters (acronyms)
                                    # Only add space if the next letter is lowercase
                                    line = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', line)
                                    
                                    # 3. Handle numbers
                                    # Add space between numbers and letters
                                    line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', line)
                                    line = re.sub(r'([A-Za-z])(\d)', r'\1 \2', line)
                                    
                                    # 4. Handle special characters
                                    # Add space around plus signs, hyphens, and other special chars
                                    line = re.sub(r'([A-Za-z])([+\-–—])', r'\1 \2', line)
                                    line = re.sub(r'([+\-–—])([A-Za-z])', r'\1 \2', line)
                                    
                                    # 5. Handle common punctuation
                                    line = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', line)
                                    line = re.sub(r'([,;])([A-Za-z])', r'\1 \2', line)
                                    
                                    # 6. Fix multiple spaces and clean up
                                    line = ' '.join(line.split())
                                    
                                    # 7. Fix specific cases
                                    line = line.replace(' .', '.')
                                    line = line.replace(' ,', ',')
                                    line = line.replace('  ', ' ')
                                    
                                    # 8. Preserve certain patterns
                                    # Keep email addresses intact
                                    line = re.sub(r'(\S+)@(\S+)', r'\1@\2', line)
                                    # Keep URLs intact
                                    line = re.sub(r'(https?://\S+)', r'\1', line)
                                    # Keep phone numbers intact
                                    line = re.sub(r'(\+\d{1,3}[- ]?\d{1,4}[- ]?\d{1,4}[- ]?\d{1,9})', r'\1', line)
                                    
                                    if line.strip():
                                        cleaned_lines.append(line)
                                
                                # Join lines with proper spacing
                                text += '\n'.join(cleaned_lines) + '\n\n'
                except Exception as pdf_error:
                    logger.error(f"Error extracting text from PDF: {str(pdf_error)}")
                    text = f"Error extracting text: {str(pdf_error)}"
                
                # Save to uploads directory
                uploads_dir = os.path.join(os.getcwd(), "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                
                save_path = os.path.join(uploads_dir, file_name)
                with open(save_path, 'wb') as f:
                    f.write(file_bytes)
                
                # Return full text content
                results.append({
                    "file_name": file_name,
                    "status": "success",
                    "message": "File processed successfully",
                    "text_content": text.strip()  # Return complete text
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
                results.append({
                    "file_name": file_name,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"An error occurred in process_files: {str(e)}")
        raise

@router.post("/match-with-budget")
async def match_with_budget(request: JobMatchRequest, db: Session = Depends(get_db)):
    try:
        # Get candidates from database
        candidates = db.query(models.Candidate).filter(
            models.Candidate.email.in_(request.candidate_emails)
        ).all()
        
        if len(candidates) != len(request.candidate_emails):
            raise HTTPException(status_code=404, detail="One or more candidates not found")
        
        # Prepare candidate information for analysis
        candidates_info = []
        for candidate in candidates:
            candidates_info.append({
                "email": candidate.email,
                "name": candidate.name,
                "resume_text": candidate.resume_text,
                "current_ctc": candidate.current_ctc,
                "expected_ctc": candidate.expected_ctc
            })
        
        # Create job entry
        new_job = models.Job(
            title="Java Developer",
            description=request.job_description,
            min_budget=request.budget * 0.9,  # 10% buffer
            max_budget=request.budget,
            status="active"
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        # Job info for Ollama
        job_info = {
            "id": new_job.id,
            "title": "Java Developer",
            "jd_text": request.job_description,
            "min_budget": new_job.min_budget,
            "max_budget": request.budget
        }
        
        # Initialize Ollama client
        ollama_client = OllamaClient()
        
        # Get candidate scores compared to job
        jd_scores = {}
        for candidate in candidates:
            candidate_info = {
                "email": candidate.email,
                "name": candidate.name,
                "resume_text": candidate.resume_text,
                "current_ctc": candidate.current_ctc,
                "expected_ctc": candidate.expected_ctc
            }
            jd_scores[candidate.email] = ollama_client.compare_candidate_with_jd(candidate_info, job_info)
        
        # Get comparative scores
        comparative_scores = ollama_client.compare_candidates(candidates_info, job_info)
        
        # Create analysis structure
        analysis = {
            "job_id": new_job.id,
            "candidates": [],
            "comparative_analysis": {
                "best_match": None,
                "reasoning": "Based on overall skills and salary fit",
                "salary_considerations": f"Budget is ${request.budget}",
                "risk_assessment": "Standard evaluation"
            }
        }
        
        # Build detailed analysis for each candidate
        for candidate in candidates:
            # Calculate technical and experience scores (using the JD match score as a base)
            jd_score = jd_scores.get(candidate.email, 0.5)
            
            # Calculate salary match score based on expected CTC vs budget
            salary_ratio = candidate.expected_ctc / request.budget
            salary_match_score = max(0, min(1, 1 - abs(1 - salary_ratio)))
            
            # Calculate overall score (weighted average)
            overall_score = (jd_score * 0.7) + (salary_match_score * 0.3)
            
            # Determine budget fit status
            if candidate.expected_ctc <= request.budget:
                budget_fit = "Within budget"
                negotiation_recommendation = "No negotiation needed, within budget"
            elif candidate.expected_ctc <= request.budget * 1.1:
                budget_fit = "Slightly above budget"
                negotiation_recommendation = "Minor negotiation may be needed"
            else:
                budget_fit = "Above budget"
                negotiation_recommendation = "Significant negotiation required"
            
            # Calculate salary gap percentage
            salary_gap_percentage = round(((candidate.expected_ctc / request.budget) - 1) * 100, 2)
            
            # Add candidate to analysis
            candidate_analysis = {
                "email": candidate.email,
                "overall_score": round(overall_score, 2),
                "salary_match_score": round(salary_match_score, 2),
                "technical_match_score": round(jd_score, 2),
                "experience_match_score": round(jd_score, 2),  # Using same score for simplicity
                "strengths": ["Technical skills match", "Relevant experience"],
                "weaknesses": ["Budget constraints" if salary_gap_percentage > 0 else "None identified"],
                "salary_analysis": {
                    "current_ctc": candidate.current_ctc,
                    "expected_ctc": candidate.expected_ctc,
                    "budget_fit": budget_fit,
                    "salary_gap_percentage": salary_gap_percentage,
                    "negotiation_recommendation": negotiation_recommendation
                },
                "recommendation": f"Candidate has a {round(jd_score * 100)}% match with required skills"
            }
            
            analysis["candidates"].append(candidate_analysis)
        
        # Find best match based on overall score
        if analysis["candidates"]:
            best_match = max(analysis["candidates"], key=lambda x: x["overall_score"])
            analysis["comparative_analysis"]["best_match"] = best_match["email"]
        
        # Store match results in database
        for candidate_analysis in analysis["candidates"]:
            match = models.CandidateJobMatch(
                candidate_email=candidate_analysis["email"],
                job_id=new_job.id,
                overall_score=candidate_analysis["overall_score"],
                salary_match_score=candidate_analysis["salary_match_score"],
                technical_match_score=candidate_analysis["technical_match_score"],
                experience_match_score=candidate_analysis["experience_match_score"],
                strengths=json.dumps(candidate_analysis["strengths"]),
                weaknesses=json.dumps(candidate_analysis["weaknesses"]),
                salary_analysis=json.dumps(candidate_analysis["salary_analysis"]),
                recommendation=candidate_analysis["recommendation"],
                comparative_analysis=json.dumps(analysis["comparative_analysis"])
            )
            db.add(match)
        
        db.commit()
        
        return {
            "status": "success",
            "job_id": new_job.id,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"An error occurred in match_with_budget: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-candidates-from-pdfs")
async def create_candidates_from_pdfs(
    files: List[UploadFile] = File(...),
    current_ctcs: List[float] = None,
    expected_ctcs: List[float] = None,
    db: Session = Depends(get_db)
):
    try:
        if len(files) != len(current_ctcs) or len(files) != len(expected_ctcs):
            raise HTTPException(
                status_code=400,
                detail="Number of files must match number of CTC values"
            )
        
        created_candidates = []
        for i, file in enumerate(files):
            # Read PDF file
            file_bytes = await file.read()
            
            # Extract text from PDF
            text = ""
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        # Try different extraction methods
                        page_text = page.extract_text()
                        if not page_text:
                            # Fallback to raw text extraction
                            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if not page_text:
                            # Try with layout preservation
                            page_text = page.extract_text(layout=True)
                        
                        # Clean up the text
                        if page_text:
                            # Split into lines and clean each line
                            lines = page_text.split('\n')
                            cleaned_lines = []
                            for line in lines:
                                # Fix common formatting issues
                                line = line.replace('|', ' | ')
                                line = line.replace('•', '• ')
                                line = line.replace(':', ': ')
                                
                                # Add spaces between words that are stuck together
                                import re
                                
                                # Smart word separation
                                # 1. Handle camelCase and PascalCase
                                line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
                                
                                # 2. Handle consecutive capital letters (acronyms)
                                line = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', line)
                                
                                # 3. Handle numbers
                                line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', line)
                                line = re.sub(r'([A-Za-z])(\d)', r'\1 \2', line)
                                
                                # 4. Handle special characters
                                line = re.sub(r'([A-Za-z])([+\-–—])', r'\1 \2', line)
                                line = re.sub(r'([+\-–—])([A-Za-z])', r'\1 \2', line)
                                
                                # 5. Handle common punctuation
                                line = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', line)
                                line = re.sub(r'([,;])([A-Za-z])', r'\1 \2', line)
                                
                                # 6. Fix multiple spaces and clean up
                                line = ' '.join(line.split())
                                
                                # 7. Fix specific cases
                                line = line.replace(' .', '.')
                                line = line.replace(' ,', ',')
                                line = line.replace('  ', ' ')
                                
                                if line.strip():
                                    cleaned_lines.append(line)
                            
                            # Join lines with proper spacing
                            text += '\n'.join(cleaned_lines) + '\n\n'
            except Exception as pdf_error:
                logger.error(f"Error extracting text from PDF: {str(pdf_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing PDF {file.filename}: {str(pdf_error)}"
                )
            
            # Save PDF file
            uploads_dir = os.path.join(os.getcwd(), "uploads", "resumes")
            os.makedirs(uploads_dir, exist_ok=True)
            resume_path = os.path.join(uploads_dir, file.filename)
            with open(resume_path, 'wb') as f:
                f.write(file_bytes)
            
            # Extract email using Ollama LLM
            email = None
            try:
                # Initialize Ollama client
                ollama_client = OllamaClient()
                
                # Extract email from resume text
                email = ollama_client.extract_email_from_resume(text)
                logger.info(f"Email extracted by Ollama: {email}")
                
                # If Ollama couldn't extract an email, use a fallback method
                if not email:
                    # Try fallback extraction method
                    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                    matches = re.findall(email_pattern, text)
                    if matches:
                        email = matches[0]
                        logger.info(f"Email extracted by regex: {email}")
                    else:
                        # Generate a synthetic email as last resort
                        email = f"{file.filename.replace('.pdf', '').lower()}@example.com"
                        logger.warning(f"Using synthetic email: {email}")
                
                # Check if this email already exists
                existing_candidate = db.query(models.Candidate).filter(models.Candidate.email == email).first()
                if existing_candidate:
                    logger.info(f"Candidate with email {email} already exists, updating instead of creating new")
                    
                    # Update existing candidate
                    existing_candidate.current_ctc = current_ctcs[i]
                    existing_candidate.expected_ctc = expected_ctcs[i]
                    existing_candidate.resume_text = text
                    existing_candidate.resume_path = resume_path
                    
                    db.add(existing_candidate)
                    created_candidates.append(existing_candidate)
                    continue
                
            except Exception as e:
                logger.error(f"Error extracting email: {e}", exc_info=True)
                # Use filename-based email as fallback
                email = f"{file.filename.replace('.pdf', '').lower()}@example.com"
                logger.warning(f"Using fallback synthetic email due to error: {email}")
            
            # Get candidate name from resume or use filename
            name = file.filename.replace('.pdf', '')
            
            # Try to extract more details with Ollama
            try:
                details = ollama_client.extract_candidate_details(text)
                if details and 'fullName' in details and details['fullName']:
                    name = details['fullName']
                    logger.info(f"Name extracted by Ollama: {name}")
            except Exception as e:
                logger.error(f"Error extracting candidate details: {e}", exc_info=True)
            
            # Create candidate
            new_candidate = models.Candidate(
                name=name,
                email=email,  # Use the extracted email as primary key
                current_ctc=current_ctcs[i],
                expected_ctc=expected_ctcs[i],
                resume_text=text,
                resume_path=resume_path
            )
            db.add(new_candidate)
            created_candidates.append(new_candidate)
        
        db.commit()
        
        # Refresh all created candidates to get their IDs
        for candidate in created_candidates:
            db.refresh(candidate)
        
        return {
            "status": "success",
            "message": f"Successfully created {len(created_candidates)} candidates",
            "candidates": [
                {
                    "name": c.name,
                    "email": c.email,
                    "current_ctc": c.current_ctc,
                    "expected_ctc": c.expected_ctc
                }
                for c in created_candidates
            ]
        }
        
    except Exception as e:
        logger.error(f"An error occurred in create_candidates_from_pdfs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add utility function to extract email from resume text using OpenAI

def extract_email_from_text(resume_text: str, fallback_email: str = None) -> tuple[str | None, bool]:
    # Try regex first
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, resume_text)
    if matches:
        return matches[0], False  # Found, not synthetic
    # If all else fails, generate a synthetic email
    if fallback_email:
        logger.warning(f"Falling back to synthetic email: {fallback_email}")
        return fallback_email, True
    return None, True

@router.post("/process-and-match")
async def process_and_match(
    job_description: str = Form(...),
    budget: float = Form(...),
    files: List[UploadFile] = File(...),
    current_ctcs: List[str] = Form(...),
    expected_ctcs: List[str] = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Validate number of files matches CTC values
        if len(files) != len(current_ctcs) or len(files) != len(expected_ctcs):
            raise HTTPException(
                status_code=400,
                detail="Number of files must match number of CTC values"
            )

        # Validate files
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"File name is missing for one of the uploads"
                )
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
            content_type = file.content_type
            if not content_type or content_type != 'application/pdf':
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} has invalid content type: {content_type}. Expected application/pdf"
                )

        os.makedirs("uploads/resumes", exist_ok=True)

        candidates = []
        resume_texts = []  # Store resume texts for ranking
        synthetic_email_warnings = []
        for file, current_ctc, expected_ctc in zip(files, current_ctcs, expected_ctcs):
            try:
                content = await file.read()
                if not content:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} is empty"
                    )
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}.pdf"
                file_path = os.path.join("uploads/resumes", filename)
                with open(file_path, "wb") as f:
                    f.write(content)
                try:
                    text = extract_text_from_pdf(file_path)
                    if not text.strip():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not extract text from {file.filename}. The PDF might be empty or corrupted."
                        )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing {file.filename}: {str(e)}"
                    )
                text = clean_text(text)
                resume_texts.append(text)  # Store text for ranking

                # Extract email using Ollama LLM
                email = None
                synthetic = False
                try:
                    # Initialize Ollama client
                    ollama_client = OllamaClient()
                    
                    # Extract email from resume text
                    email = ollama_client.extract_email_from_resume(text)
                    logger.info(f"Email extracted by Ollama for {file.filename}: {email}")
                    
                    # If Ollama couldn't extract an email, use regex fallback
                    if not email:
                        # Fallback to regex extraction
                        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                        matches = re.findall(email_pattern, text)
                        if matches:
                            email = matches[0]
                            logger.info(f"Email extracted by regex for {file.filename}: {email}")
                        else:
                            # Generate a synthetic email as last resort
                            email = f"{os.path.splitext(file.filename)[0].lower()}@example.com"
                            logger.warning(f"Using synthetic email for {file.filename}: {email}")
                            synthetic = True
                            synthetic_email_warnings.append({"file": file.filename, "email": email})
                except Exception as e:
                    logger.error(f"Error extracting email with Ollama: {e}", exc_info=True)
                    # Fallback to the original method
                    fallback_email = f"{os.path.splitext(file.filename)[0].lower()}@example.com"
                    email, synthetic = extract_email_from_text(text, fallback_email=fallback_email)
                    if synthetic:
                        logger.warning(f"Used synthetic email for {file.filename}: {email}")
                        synthetic_email_warnings.append({"file": file.filename, "email": email})

                if not email:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not extract or generate email for {file.filename}."
                    )

                # Try to extract detailed candidate information using Ollama
                name = os.path.splitext(file.filename)[0]  # Default name from filename
                try:
                    # Extract candidate details
                    details = ollama_client.extract_candidate_details(text)
                    if details and isinstance(details, dict):
                        # Extract name if available
                        if 'fullName' in details and details['fullName']:
                            name = details['fullName']
                            logger.info(f"Name extracted by Ollama for {file.filename}: {name}")
                except Exception as e:
                    logger.error(f"Error extracting candidate details: {e}", exc_info=True)

                # Check if candidate exists
                candidate = db.query(models.Candidate).filter(models.Candidate.email == email).first()
                if candidate:
                    # Update candidate info
                    candidate.name = name
                    candidate.resume_path = file_path
                    candidate.resume_text = text
                    candidate.current_ctc = float(current_ctc)
                    candidate.expected_ctc = float(expected_ctc)
                else:
                    candidate = models.Candidate(
                        name=name,
                        email=email,
                        resume_path=file_path,
                        resume_text=text,
                        current_ctc=float(current_ctc),
                        expected_ctc=float(expected_ctc)
                    )
                    db.add(candidate)
                    db.flush()
                candidates.append(candidate)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid CTC value for {file.filename}: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing {file.filename}: {str(e)}"
                )

        # Create job
        job = models.Job(
            description=job_description,
            min_budget=budget * 0.9,
            max_budget=budget
        )
        db.add(job)
        db.flush()  # Flush to get the ID

        try:
            # Initialize Ollama client
            ollama_client = OllamaClient()
            
            # Get resume texts for ranking
            resume_texts = [c.resume_text for c in candidates]
            
            # Use Ollama to rank resumes
            ranking_analysis = ollama_client.rank_resumes(job_description, resume_texts)
            
            if not ranking_analysis or "error" in ranking_analysis:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error getting ranking analysis: {ranking_analysis.get('error', 'Unknown error')}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting ranking analysis: {str(e)}"
            )

        # Prepare job info for Ollama
        job_info = {
            "id": job.id,
            "title": "Position",  # Generic title
            "jd_text": job_description,
            "min_budget": budget * 0.9,  # Assuming 10% buffer
            "max_budget": budget
        }
        
        # Prepare candidate info for Ollama
        candidates_info = []
        for c in candidates:
            candidates_info.append({
                "email": c.email,
                "name": c.name,
                "resume_text": c.resume_text,
                "current_ctc": c.current_ctc,
                "expected_ctc": c.expected_ctc
            })
        
        try:
            # Get comparative scores using Ollama
            comparative_scores = ollama_client.compare_candidates(candidates_info, job_info)
            
            # Get individual JD match scores
            jd_scores = {}
            for candidate_info in candidates_info:
                jd_scores[candidate_info["email"]] = ollama_client.compare_candidate_with_jd(candidate_info, job_info)
            
            # Build analysis structure
            analysis = {
                "job_id": job.id,
                "candidates": [],
                "comparative_analysis": {
                    "best_match": None,
                    "reasoning": "Based on skills and budget fit",
                    "salary_considerations": f"Budget: ${budget}",
                    "risk_assessment": "Standard evaluation"
                }
            }
            
            # Create detailed analysis for each candidate
            for candidate in candidates:
                # Get scores
                jd_score = jd_scores.get(candidate.email, 0.5)
                comparative_score = comparative_scores.get(candidate.email, 0.5)
                
                # Calculate salary match score based on expected CTC vs budget
                salary_ratio = float(candidate.expected_ctc) / budget
                salary_match_score = max(0, min(1, 1 - abs(1 - salary_ratio)))
                
                # Calculate overall score (weighted average)
                overall_score = (jd_score * 0.4) + (comparative_score * 0.3) + (salary_match_score * 0.3)
                
                # Determine budget fit status
                if float(candidate.expected_ctc) <= budget:
                    budget_fit = "Within budget"
                    negotiation_recommendation = "No negotiation needed"
                elif float(candidate.expected_ctc) <= budget * 1.1:
                    budget_fit = "Slightly above budget"
                    negotiation_recommendation = "Minor negotiation may be needed"
                else:
                    budget_fit = "Above budget"
                    negotiation_recommendation = "Significant negotiation required"
                
                # Calculate salary gap percentage
                salary_gap_percentage = round(((float(candidate.expected_ctc) / budget) - 1) * 100, 2)
                
                # Add candidate to analysis
                candidate_analysis = {
                    "email": candidate.email,
                    "overall_score": round(overall_score, 2),
                    "salary_match_score": round(salary_match_score, 2),
                    "technical_match_score": round(jd_score, 2),
                    "experience_match_score": round(comparative_score, 2),
                    "strengths": ["Skills match job requirements", "Relevant experience"],
                    "weaknesses": ["Budget constraints" if salary_gap_percentage > 0 else "None identified"],
                    "salary_analysis": {
                        "current_ctc": float(candidate.current_ctc),
                        "expected_ctc": float(candidate.expected_ctc),
                        "budget_fit": budget_fit,
                        "salary_gap_percentage": salary_gap_percentage,
                        "negotiation_recommendation": negotiation_recommendation
                    },
                    "recommendation": f"Candidate has a {round(overall_score * 100)}% overall match"
                }
                
                analysis["candidates"].append(candidate_analysis)
            
            # Find best match based on overall score
            if analysis["candidates"]:
                best_match = max(analysis["candidates"], key=lambda x: x["overall_score"])
                analysis["comparative_analysis"]["best_match"] = best_match["email"]
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting analysis: {str(e)}"
            )

        # Store analysis results
        for candidate_analysis in analysis["candidates"]:
            match = models.CandidateJobMatch(
                candidate_email=candidate_analysis["email"],  # Use email as the primary key
                job_id=job.id,
                overall_score=candidate_analysis["overall_score"],
                salary_match_score=candidate_analysis["salary_match_score"],
                technical_match_score=candidate_analysis["technical_match_score"],
                experience_match_score=candidate_analysis["experience_match_score"],
                strengths=json.dumps(candidate_analysis["strengths"]),
                weaknesses=json.dumps(candidate_analysis["weaknesses"]),
                salary_analysis=json.dumps(candidate_analysis["salary_analysis"]),
                recommendation=candidate_analysis["recommendation"],
                comparative_analysis=json.dumps(analysis["comparative_analysis"])
            )
            db.add(match)

        db.commit()

        return {
            "status": "success",
            "job_id": job.id,
            "candidates": [
                {
                    "name": c.name,
                    "email": c.email,  # Email is the primary key
                    "current_ctc": c.current_ctc,
                    "expected_ctc": c.expected_ctc
                } for c in candidates
            ],
            "analysis": analysis,
            "ranking_analysis": ranking_analysis,
            **({"synthetic_email_warnings": synthetic_email_warnings} if synthetic_email_warnings else {})
        }

    except HTTPException as he:
        db.rollback()
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred in process_and_match: {str(e)}"
        )

@router.post("/extract-email")
async def extract_email_ollama(resume: UploadFile = File(...)):
    """
    Extract email from a resume using Ollama LLM.
    
    Args:
        resume: The resume file to extract email from
        
    Returns:
        JSON with extracted email
    """
    try:
        # Read the file
        file_content = await resume.read()
        
        # Create a temporary file path
        temp_file_path = f"uploads/temp_{resume.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file temporarily
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Extract text from the file
        resume_text = ""
        if temp_file_path.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_file_path)
        else:
            # For non-PDF files, try to read as text
            with open(temp_file_path, "r", errors="ignore") as f:
                resume_text = f.read()
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        if not resume_text:
            return {"error": "Could not extract text from the resume"}
        
        # Initialize Ollama client
        ollama_client = OllamaClient()
        
        # Extract email
        email = ollama_client.extract_email_from_resume(resume_text)
        
        # Extract full candidate details
        details = ollama_client.extract_candidate_details(resume_text)
        
        return {
            "email": email,
            "details": details,
            "text_sample": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        }
        
    except Exception as e:
        logger.error(f"Error extracting email: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting email: {str(e)}"
        )


# ==================== RANKING ENDPOINTS ====================

class StatusUpdate(BaseModel):
    status: str  # "active", "saved", "rejected"


@router.post("/rank-by-job/{job_id}")
async def rank_by_job(job_id: int, db: Session = Depends(get_db)):
    """
    Rank ALL candidates against a specific job using Ollama.
    Computes both JD match scores and comparative rankings.
    """
    try:
        # Get the job
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get all candidates
        candidates = db.query(models.Candidate).all()
        if not candidates:
            raise HTTPException(status_code=404, detail="No candidates found")

        # Prepare job info
        job_info = {
            "id": job.id,
            "title": job.title or "",
            "jd_text": job.description or job.jd_text or "",
            "min_budget": job.min_budget or 0,
            "max_budget": job.max_budget or 0,
        }

        # Initialize Ollama client
        ollama_client = OllamaClient()

        # --- Step 1: Get individual JD match scores ---
        jd_scores = {}
        for candidate in candidates:
            candidate_info = {
                "name": candidate.name,
                "resume_text": candidate.resume_text or "",
                "current_ctc": candidate.current_ctc or 0,
                "expected_ctc": candidate.expected_ctc or 0,
            }
            score = ollama_client.compare_candidate_with_jd(candidate_info, job_info)
            jd_scores[candidate.email] = score
            logger.info(f"JD match score for {candidate.email}: {score}")

        # --- Step 2: Get comparative scores ---
        candidates_info_for_compare = []
        for candidate in candidates:
            candidates_info_for_compare.append({
                "id": hash(candidate.email) % 10000,  # numeric id for prompt
                "name": candidate.name,
                "resume_text": candidate.resume_text or "",
                "current_ctc": candidate.current_ctc or 0,
                "expected_ctc": candidate.expected_ctc or 0,
            })

        # Map numeric ids back to emails
        id_to_email = {}
        for i, candidate in enumerate(candidates):
            id_to_email[hash(candidate.email) % 10000] = candidate.email

        comparative_scores_raw = ollama_client.compare_candidates(candidates_info_for_compare, job_info)
        comparative_scores = {}
        for k, v in comparative_scores_raw.items():
            email = id_to_email.get(int(k), None)
            if email:
                comparative_scores[email] = v

        # --- Step 3: Build results and store in DB ---
        # Delete old matches for this job first
        db.query(models.CandidateJobMatch).filter(
            models.CandidateJobMatch.job_id == job_id
        ).delete()

        results = []
        for candidate in candidates:
            jd_score = jd_scores.get(candidate.email, 0.5)
            comp_score = comparative_scores.get(candidate.email, 0.5)

            # Salary match
            budget = job.max_budget or 1
            salary_ratio = (candidate.expected_ctc or 0) / budget if budget > 0 else 0
            salary_match = max(0.0, min(1.0, 1.0 - abs(1.0 - salary_ratio)))

            # Overall = 40% JD match + 30% comparative + 30% salary
            overall = (jd_score * 0.4) + (comp_score * 0.3) + (salary_match * 0.3)

            # Budget fit
            expected = candidate.expected_ctc or 0
            if expected <= budget:
                budget_fit = "Within budget"
            elif expected <= budget * 1.1:
                budget_fit = "Slightly above"
            else:
                budget_fit = "Above budget"

            salary_gap = round(((expected / budget) - 1) * 100, 2) if budget > 0 else 0

            strengths = ["Technical skills match"] if jd_score >= 0.6 else ["Potential growth candidate"]
            if salary_match >= 0.8:
                strengths.append("Good salary fit")
            weaknesses = []
            if jd_score < 0.5:
                weaknesses.append("JD skills mismatch")
            if salary_match < 0.5:
                weaknesses.append("Budget constraints")
            if not weaknesses:
                weaknesses.append("None identified")

            match_record = models.CandidateJobMatch(
                candidate_email=candidate.email,
                job_id=job_id,
                jd_match_score=round(jd_score, 3),
                comparative_score=round(comp_score, 3),
                overall_score=round(overall, 3),
                salary_match_score=round(salary_match, 3),
                technical_match_score=round(jd_score, 3),
                experience_match_score=round(comp_score, 3),
                strengths=json.dumps(strengths),
                weaknesses=json.dumps(weaknesses),
                salary_analysis=json.dumps({
                    "current_ctc": candidate.current_ctc,
                    "expected_ctc": candidate.expected_ctc,
                    "budget_fit": budget_fit,
                    "salary_gap_percentage": salary_gap,
                }),
                recommendation=f"{round(jd_score * 100)}% JD match, {round(comp_score * 100)}% comparative",
                status="active",
            )
            db.add(match_record)

            results.append({
                "candidate_email": candidate.email,
                "candidate_name": candidate.name,
                "current_ctc": candidate.current_ctc,
                "expected_ctc": candidate.expected_ctc,
                "jd_match_score": round(jd_score, 3),
                "comparative_score": round(comp_score, 3),
                "overall_score": round(overall, 3),
                "salary_match_score": round(salary_match, 3),
                "strengths": strengths,
                "weaknesses": weaknesses,
                "budget_fit": budget_fit,
                "salary_gap_percentage": salary_gap,
                "recommendation": match_record.recommendation,
                "status": "active",
            })

        db.commit()

        # Sort by overall score descending
        results.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "job_id": job_id,
            "job_title": job.title,
            "total_candidates": len(results),
            "rankings": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rank_by_job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings/{job_id}")
async def get_rankings(job_id: int, db: Session = Depends(get_db)):
    """Get cached ranking results for a job (no re-ranking)."""
    try:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        matches = (
            db.query(models.CandidateJobMatch)
            .filter(models.CandidateJobMatch.job_id == job_id)
            .all()
        )

        if not matches:
            return {
                "job_id": job_id,
                "job_title": job.title,
                "total_candidates": 0,
                "rankings": [],
            }

        results = []
        for m in matches:
            candidate = db.query(models.Candidate).filter(
                models.Candidate.email == m.candidate_email
            ).first()

            results.append({
                "candidate_email": m.candidate_email,
                "candidate_name": candidate.name if candidate else m.candidate_email,
                "current_ctc": candidate.current_ctc if candidate else 0,
                "expected_ctc": candidate.expected_ctc if candidate else 0,
                "jd_match_score": m.jd_match_score or 0,
                "comparative_score": m.comparative_score or 0,
                "overall_score": m.overall_score or 0,
                "salary_match_score": m.salary_match_score or 0,
                "strengths": json.loads(m.strengths) if m.strengths else [],
                "weaknesses": json.loads(m.weaknesses) if m.weaknesses else [],
                "budget_fit": json.loads(m.salary_analysis).get("budget_fit", "Unknown") if m.salary_analysis else "Unknown",
                "salary_gap_percentage": json.loads(m.salary_analysis).get("salary_gap_percentage", 0) if m.salary_analysis else 0,
                "recommendation": m.recommendation or "",
                "status": m.status or "active",
            })

        results.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "job_id": job_id,
            "job_title": job.title,
            "total_candidates": len(results),
            "rankings": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_rankings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/candidate-match/{candidate_email}/{job_id}/status")
async def update_match_status(
    candidate_email: str,
    job_id: int,
    body: StatusUpdate,
    db: Session = Depends(get_db),
):
    """Update candidate-job match status to active/saved/rejected."""
    if body.status not in ("active", "saved", "rejected"):
        raise HTTPException(status_code=400, detail="Status must be active, saved, or rejected")

    match = (
        db.query(models.CandidateJobMatch)
        .filter(
            models.CandidateJobMatch.candidate_email == candidate_email,
            models.CandidateJobMatch.job_id == job_id,
        )
        .first()
    )

    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    match.status = body.status
    db.commit()

    return {"status": "success", "candidate_email": candidate_email, "job_id": job_id, "new_status": body.status}