// TypeScript interfaces for AI Resume Screener API

// User types
export interface User {
  id: number;
  email: string;
  full_name: string;
  is_active: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// Candidate types
export interface Candidate {
  email: string;
  name: string;
  current_ctc: number;
  expected_ctc: number;
  resume_path?: string;
  resume_text?: string;
  additional_info?: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
}

export interface CandidateWithScores extends Candidate {
  jd_match_score?: number;
  comparative_score?: number;
  overall_score?: number;
  salary_match_score?: number;
  technical_match_score?: number;
  experience_match_score?: number;
  strengths?: string;
  weaknesses?: string;
  salary_analysis?: string;
  recommendation?: string;
}

export interface CandidateCreate {
  name: string;
  email: string;
  current_ctc: number;
  expected_ctc: number;
  resume_text?: string;
}

// Job types
export interface Job {
  id: number;
  title: string;
  description?: string;
  min_budget?: number;
  max_budget?: number;
  status: 'active' | 'closed' | 'draft';
  recruiter_id?: number;
  created_at: string;
  updated_at?: string;
}

export interface JobCreate {
  title: string;
  description?: string;
  min_budget?: number;
  max_budget?: number;
  status?: string;
}

export interface JobWithCandidates extends Job {
  candidates?: CandidateWithScores[];
}

// Match request types
export interface JobMatchRequest {
  job_description: string;
  budget: number;
  candidate_emails: string[];
}

// API response types
export interface ApiError {
  detail: string;
}

export interface MatchResult {
  candidate: CandidateWithScores;
  within_budget: boolean;
  salary_analysis: string;
}

export interface ProcessAndMatchResponse {
  candidates_created: number;
  match_results: MatchResult[];
  job_id?: number;
}

// Ranking types
export interface RankingResult {
  candidate_email: string;
  candidate_name: string;
  current_ctc: number;
  expected_ctc: number;
  jd_match_score: number;
  comparative_score: number;
  overall_score: number;
  salary_match_score: number;
  strengths: string[];
  weaknesses: string[];
  budget_fit: string;
  salary_gap_percentage: number;
  recommendation: string;
  status: 'active' | 'saved' | 'rejected';
}

export interface RankingResponse {
  job_id: number;
  job_title: string;
  total_candidates: number;
  rankings: RankingResult[];
}
