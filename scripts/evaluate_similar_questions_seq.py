#!/usr/bin/env python3
import json
import csv
import logging
import time
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evals/similar_questions_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SimilarQuestionEvaluationResult:
    question_id: str
    original_question: str
    subject: str
    similar_question_text: str
    original_similarity_score: float
    summarized_solution_approach: str
    conceptual_similarity_score: Optional[int] = None
    structural_similarity_score: Optional[int] = None
    difficulty_alignment_score: Optional[int] = None
    solution_approach_transferability_score: Optional[int] = None
    total_score: Optional[int] = None
    evaluation_notes: Optional[str] = None
    processing_time: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = ""

class RateLimitedSimilarQuestionsEvaluationRunner:
    def __init__(self, 
                 evaluation_api_url: str = "http://localhost:8000",
                 output_file: str = "evals/similar_questions_evaluation_results.csv",
                 request_delay: float = 30.0,  # 30 seconds between requests
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.evaluation_api_url = evaluation_api_url.rstrip('/')
        self.output_file = output_file
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.processed_pairs = set()
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results to resume from where we left off"""
        if Path(self.output_file).exists():
            try:
                df = pd.read_csv(self.output_file)
                # Create unique identifier for each question-similar_question pair
                self.processed_pairs = set(
                    f"{row['question_id']}_{hash(row['similar_question_text'])}" 
                    for _, row in df.iterrows()
                    if pd.notna(row['question_id']) and pd.notna(row['similar_question_text'])
                )
                logger.info(f"Loaded {len(self.processed_pairs)} existing evaluations from {self.output_file}")
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")

    def _clean_text(self, text: str) -> str:
        """Clean and validate text for API requests"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit length to prevent oversized requests
        if len(text) > 4500:  # Leave buffer for 5000 max
            text = text[:4500] + "..."
            logger.warning("Text truncated due to length")
        
        # Remove any problematic characters that might cause JSON issues
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        return text

    def _save_result(self, result: SimilarQuestionEvaluationResult):
        """Save individual result immediately to prevent data loss"""
        result.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        file_exists = Path(self.output_file).exists()
        
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    # Write header
                    headers = [
                        'question_id', 'original_question', 'subject',
                        'similar_question_text', 'original_similarity_score', 'summarized_solution_approach',
                        'conceptual_similarity_score', 'structural_similarity_score', 'difficulty_alignment_score',
                        'solution_approach_transferability_score', 'total_score', 'evaluation_notes',
                        'processing_time', 'error', 'timestamp'
                    ]
                    writer.writerow(headers)
                
                # Clean text fields before saving
                row = [
                    result.question_id,
                    self._clean_text(result.original_question),
                    result.subject,
                    self._clean_text(result.similar_question_text),
                    result.original_similarity_score,
                    self._clean_text(result.summarized_solution_approach),
                    result.conceptual_similarity_score,
                    result.structural_similarity_score,
                    result.difficulty_alignment_score,
                    result.solution_approach_transferability_score,
                    result.total_score,
                    self._clean_text(result.evaluation_notes or ""),
                    result.processing_time,
                    self._clean_text(result.error or ""),
                    result.timestamp
                ]
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to save result for {result.question_id}: {e}")

    def _make_request_with_retry(self, method: str, url: str, json_data: Dict, timeout: int = 120) -> Dict[str, Any]:
        """Make HTTP request with retry logic and better error handling"""
        
        for attempt in range(self.max_retries):
            try:
                # Validate JSON data before sending
                if not json_data or not isinstance(json_data, dict):
                    raise ValueError("Invalid JSON data")
                
                # Clean all string values in the request
                cleaned_data = {}
                for key, value in json_data.items():
                    if isinstance(value, str):
                        cleaned_data[key] = self._clean_text(value)
                    else:
                        cleaned_data[key] = value
                
                response = self.session.request(
                    method=method,
                    url=url,
                    json=cleaned_data,
                    timeout=timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Check for successful response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 400:
                    # Log the actual error for 400s
                    try:
                        error_detail = response.json().get('detail', 'Unknown validation error')
                    except:
                        error_detail = response.text[:200]
                    raise ValueError(f"Validation error (400): {error_detail}")
                elif response.status_code == 429:
                    # Rate limit hit - wait longer and retry
                    if attempt < self.max_retries - 1:
                        wait_time = self.request_delay * 2  # Double the normal delay for rate limits
                        logger.warning(f"Rate limit hit (429), waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.exceptions.HTTPError(f"Rate limit exceeded after {self.max_retries} attempts")
                elif response.status_code in [500, 502, 503, 504]:
                    # Server errors - retry
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.exceptions.HTTPError(f"Server error {response.status_code} after {self.max_retries} attempts")
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Request timeout after {self.max_retries} attempts")
                    
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Connection error after {self.max_retries} attempts")
                    
            except ValueError as e:
                # Don't retry validation errors
                raise e
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed ({str(e)}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")
        
        raise Exception("Max retries exceeded")

    def _call_evaluation_api(self, question_text: str, similar_question: str, solution_approach: str) -> Dict[str, Any]:
        """Make API call to similar questions evaluation"""
        
        # Validate and clean inputs
        clean_question = self._clean_text(question_text)
        clean_similar = self._clean_text(similar_question)
        clean_approach = self._clean_text(solution_approach)
        
        if len(clean_question) < 10:
            raise ValueError("Original question too short after cleaning")
        if len(clean_similar) < 10:
            raise ValueError("Similar question too short after cleaning")
        if len(clean_approach) < 10:
            raise ValueError("Solution approach too short after cleaning")
        
        request_data = {
            "question_text": clean_question,
            "similar_question": clean_similar,
            "summarized_solution_approach": clean_approach
        }
        
        return self._make_request_with_retry(
            "POST", 
            f"{self.evaluation_api_url}/evaluate", 
            request_data
        )

    def _process_similar_question_pair(self, question_data: Dict[str, Any], similar_question_data: Dict[str, Any]) -> SimilarQuestionEvaluationResult:
        """Process a single original question with one of its similar questions"""
        question_id = question_data.get('question_id', 'unknown')
        original_question = question_data.get('question_text', '')
        subject = question_data.get('subject', 'Unknown')
        
        similar_question_text = similar_question_data.get('similar_question_text', '')
        original_similarity_score = similar_question_data.get('similarity_score', 0.0)
        solution_approach = similar_question_data.get('summarized_solution_approach', '')
        
        # Validate inputs
        if not all([question_id, original_question, similar_question_text, solution_approach]):
            raise ValueError("Invalid question pair data: missing required fields")
        
        result = SimilarQuestionEvaluationResult(
            question_id=question_id,
            original_question=original_question,
            subject=subject,
            similar_question_text=similar_question_text,
            original_similarity_score=original_similarity_score,
            summarized_solution_approach=solution_approach
        )
        
        try:
            # Call evaluation API
            response = self._call_evaluation_api(
                original_question, 
                similar_question_text, 
                solution_approach
            )
            
            # Extract results
            data = response['data']
            result.conceptual_similarity_score = data.get('conceptual_similarity_score')
            result.structural_similarity_score = data.get('structural_similarity_score')
            result.difficulty_alignment_score = data.get('difficulty_alignment_score')
            result.solution_approach_transferability_score = data.get('solution_approach_transferability_score')
            result.total_score = data.get('total_score')
            result.evaluation_notes = data.get('notes')
            result.processing_time = response.get('processing_time_ms')
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"Evaluation failed for {question_id}: {e}")

        return result

    def _generate_question_pairs(self, data: List[Dict[str, Any]]) -> List[tuple]:
        """Generate all question-similar_question pairs to evaluate"""
        pairs = []
        
        for question_data in data:
            question_id = question_data.get('question_id')
            similar_questions = question_data.get('similar_questions', [])
            
            if not question_id or not similar_questions:
                continue
                
            for similar_q in similar_questions:
                if isinstance(similar_q, dict) and similar_q.get('similar_question_text'):
                    pair_id = f"{question_id}_{hash(similar_q['similar_question_text'])}"
                    if pair_id not in self.processed_pairs:
                        pairs.append((question_data, similar_q))
        
        return pairs

    def run_evaluation(self, data_file: str = 'similar_question_data.json'):
        """Run evaluation on all question-similar_question pairs with rate limiting"""
        
        # Load data
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)[:100]
            logger.info(f"Loaded {len(data)} questions from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            return

        # Validate and filter data
        valid_data = []
        for item in data:
            if isinstance(item, dict) and item.get('question_id') and item.get('question_text'):
                valid_data.append(item)
            else:
                logger.warning(f"Skipping invalid data item: {item}")
        
        logger.info(f"Found {len(valid_data)} valid questions")

        # Generate all question-similar_question pairs
        question_pairs = self._generate_question_pairs(valid_data)
        total_pairs = len(question_pairs)
        
        logger.info(f"Generated {total_pairs} question-similar_question pairs to evaluate")
        logger.info(f"Skipping {len(self.processed_pairs)} already processed pairs")
        logger.info(f"Processing requests with {self.request_delay}s delay between each request")

        if not question_pairs:
            logger.info("All question pairs have already been processed!")
            return

        # Process pairs sequentially with delay
        successful_count = 0
        error_count = 0
        
        with tqdm(total=len(question_pairs), desc="Evaluating similar questions") as pbar:
            for i, (question_data, similar_q) in enumerate(question_pairs):
                pair_id = f"{question_data['question_id']}_{hash(similar_q['similar_question_text'])}"
                
                try:
                    # Process the pair
                    result = self._process_similar_question_pair(question_data, similar_q)
                    self._save_result(result)
                    self.processed_pairs.add(pair_id)
                    
                    if result.total_score is not None:
                        successful_count += 1
                        pbar.set_postfix({
                            'ID': result.question_id[:8],
                            'Total': result.total_score,
                            'Concept': result.conceptual_similarity_score or 0,
                            'Success': successful_count,
                            'Errors': error_count
                        })
                    else:
                        error_count += 1
                        pbar.set_postfix({
                            'ID': result.question_id[:8], 
                            'Status': 'Error',
                            'Success': successful_count,
                            'Errors': error_count
                        })
                        
                except Exception as e:
                    error_count += 1
                    question_id = question_data.get('question_id', 'unknown')
                    logger.error(f"Failed to process pair {pair_id}: {e}")
                    
                    # Save error result
                    error_result = SimilarQuestionEvaluationResult(
                        question_id=question_id,
                        original_question=question_data.get('question_text', 'Error loading question'),
                        subject=question_data.get('subject', 'Unknown'),
                        similar_question_text=similar_q.get('similar_question_text', 'Error loading similar question'),
                        original_similarity_score=similar_q.get('similarity_score', 0.0),
                        summarized_solution_approach=similar_q.get('summarized_solution_approach', 'Error loading approach'),
                        error=f"Processing failed: {str(e)}"
                    )
                    self._save_result(error_result)
                    
                    pbar.set_postfix({
                        'ID': question_id[:8], 
                        'Status': 'Error',
                        'Success': successful_count,
                        'Errors': error_count
                    })
                
                pbar.update(1)
                
                # Wait before next request (except for the last one)
                if i < len(question_pairs) - 1:
                    logger.info(f"Waiting {self.request_delay}s before next request...")
                    
                    # Show countdown in progress bar
                    for remaining in range(int(self.request_delay), 0, -1):
                        pbar.set_description(f"Waiting {remaining}s")
                        time.sleep(1)
                    pbar.set_description("Evaluating similar questions")

        logger.info(f"Similar questions evaluation completed! Results saved to {self.output_file}")
        logger.info(f"Final stats - Successful: {successful_count}, Errors: {error_count}")
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics"""
        try:
            df = pd.read_csv(self.output_file)
            total = len(df)
            successful_evaluations = len(df[df['total_score'].notna()])
            errors = len(df[df['error'].notna() & (df['error'] != '')])
            
            print(f"\n{'='*60}")
            print(f"SIMILAR QUESTIONS EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total pairs processed: {total}")
            print(f"Successful evaluations: {successful_evaluations}")
            print(f"Errors: {errors}")
            
            if successful_evaluations > 0:
                # Score statistics
                avg_total = df['total_score'].mean()
                avg_conceptual = df['conceptual_similarity_score'].mean()
                avg_structural = df['structural_similarity_score'].mean()
                avg_difficulty = df['difficulty_alignment_score'].mean()
                avg_transferability = df['solution_approach_transferability_score'].mean()
                
                # Score distribution
                high_total = len(df[df['total_score'] >= 80])
                medium_total = len(df[(df['total_score'] >= 60) & (df['total_score'] < 80)])
                low_total = len(df[df['total_score'] < 60])
                
                print(f"\nScore Averages:")
                print(f"  Total Score: {avg_total:.2f}")
                print(f"  Conceptual Similarity: {avg_conceptual:.2f}")
                print(f"  Structural Similarity: {avg_structural:.2f}")
                print(f"  Difficulty Alignment: {avg_difficulty:.2f}")
                print(f"  Solution Transferability: {avg_transferability:.2f}")
                
                print(f"\nScore Distribution:")
                print(f"  High scores (≥80): {high_total}")
                print(f"  Medium scores (60-79): {medium_total}")
                print(f"  Low scores (<60): {low_total}")
                
                # Subject breakdown if available
                if 'subject' in df.columns:
                    subject_stats = df.groupby('subject')['total_score'].agg(['count', 'mean']).round(2)
                    print(f"\nBy Subject:")
                    for subject, stats in subject_stats.iterrows():
                        print(f"  {subject}: {stats['count']} pairs, avg score {stats['mean']}")
                        
            print(f"{'='*60}")
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

def main():
    # Configuration
    EVALUATION_API_URL = "http://localhost:8000"  # Your similar questions evaluation API
    OUTPUT_FILE = "evals/similar_questions_evaluation_results_mas.csv"
    REQUEST_DELAY = 30.0  #
    # Test API connectivity
    # try:
    #     session = requests.Session()
    #     health_check = session.get(f"{EVALUATION_API_URL}/health", timeout=10)
        
    #     if health_check.status_code != 200:
    #         raise Exception(f"Evaluation API not healthy: {health_check.status_code}")
            
    #     logger.info("Evaluation API is healthy and ready")
        
    #     try:
    #         test_response = session.post(
    #             f"{EVALUATION_API_URL}/evaluate",
    #             json={
    #                 "question_text": "What is 2+2?",
    #                 "similar_question": "What is 3+3?", 
    #                 "summarized_solution_approach": "Simple addition"
    #             },
    #             timeout=30
    #         )
    #         if test_response.status_code == 200:
    #             logger.info("Evaluation endpoint test successful")
    #         elif test_response.status_code == 429:
    #             logger.warning("Rate limit detected during test - this is expected")
    #         else:
    #             logger.warning(f"Evaluation endpoint test returned {test_response.status_code}")
    #     except Exception as e:
    #         logger.warning(f"Evaluation endpoint test failed: {e}")
        
    # except Exception as e:
    #     logger.error(f"API connectivity check failed: {e}")
    #     return

    # Run evaluation
    runner = RateLimitedSimilarQuestionsEvaluationRunner(
        evaluation_api_url=EVALUATION_API_URL,
        output_file=OUTPUT_FILE,
        request_delay=REQUEST_DELAY,
        max_retries=1,
        retry_delay=2.0
    )
    
    try:
        logger.info(f"Starting rate-limited evaluation with {REQUEST_DELAY}s delays between requests")
        runner.run_evaluation()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
