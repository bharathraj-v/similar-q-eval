#!/usr/bin/env python3
import json
import csv
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    question_id: str
    question_text: str
    subject: str
    standard_explanation: Optional[str] = None
    standard_answer: Optional[str] = None
    standard_processing_time: Optional[int] = None
    standard_error: Optional[str] = None
    sim_explanation: Optional[str] = None
    sim_answer: Optional[str] = None
    sim_processing_time: Optional[int] = None
    sim_error: Optional[str] = None
    sim_score: Optional[int] = None
    standard_score: Optional[int] = None
    analysis_notes: Optional[str] = None
    analysis_error: Optional[str] = None
    timestamp: str = ""

class EvaluationRunner:
    def __init__(self, 
                 solution_api_url: str = "http://localhost:8000",
                 analysis_api_url: str = "http://localhost:8003",
                 output_file: str = "evaluation_results.csv",
                 max_workers: int = 4):
        self.solution_api_url = solution_api_url.rstrip('/')
        self.analysis_api_url = analysis_api_url.rstrip('/')
        self.output_file = output_file
        self.max_workers = max_workers
        self.results: List[EvaluationResult] = []
        self.processed_ids = set()
        
        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results to resume from where we left off"""
        if Path(self.output_file).exists():
            try:
                df = pd.read_csv(self.output_file)
                self.processed_ids = set(df['question_id'].tolist())
                logger.info(f"Loaded {len(self.processed_ids)} existing results from {self.output_file}")
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")

    def _save_result(self, result: EvaluationResult):
        """Save individual result immediately to prevent data loss"""
        result.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        file_exists = Path(self.output_file).exists()
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                # Write header
                headers = [
                    'question_id', 'question_text', 'subject',
                    'standard_explanation', 'standard_answer', 'standard_processing_time', 'standard_error',
                    'sim_explanation', 'sim_answer', 'sim_processing_time', 'sim_error',
                    'sim_score', 'standard_score', 'analysis_notes', 'analysis_error', 'timestamp'
                ]
                writer.writerow(headers)
            
            # Write data
            row = [
                result.question_id, result.question_text, result.subject,
                result.standard_explanation, result.standard_answer, 
                result.standard_processing_time, result.standard_error,
                result.sim_explanation, result.sim_answer, 
                result.sim_processing_time, result.sim_error,
                result.sim_score, result.standard_score, 
                result.analysis_notes, result.analysis_error, result.timestamp
            ]
            writer.writerow(row)

    def _call_solution_api(self, question: str, use_similar_questions: bool) -> Dict[str, Any]:
        """Make API call to solution builder"""
        try:
            response = requests.post(
                f"{self.solution_api_url}/solve",
                json={
                    "question": question,
                    "use_similar_questions": use_similar_questions
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Solution API error: {str(e)}")

    def _call_analysis_api(self, question: str, standard_result: Dict, sim_result: Dict) -> Dict[str, Any]:
        """Make API call to comparative analysis"""
        try:
            response = requests.post(
                f"{self.analysis_api_url}/analyse",
                json={
                    "question": question,
                    "sim_answer_explanation": sim_result['data']['explanation'],
                    "sim_answer_final_answer": sim_result['data']['final_answer'],
                    "no_sim_answer_explanation": standard_result['data']['explanation'],
                    "no_sim_answer_final_answer": standard_result['data']['final_answer']
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Analysis API error: {str(e)}")

    def _process_question(self, question_data: Dict[str, Any]) -> EvaluationResult:
        """Process a single question through both agents and analysis"""
        question_id = question_data['question_id']
        question_text = question_data['question_text']
        subject = question_data.get('subject', 'Unknown')
        
        result = EvaluationResult(
            question_id=question_id,
            question_text=question_text,
            subject=subject
        )
        
        # Get standard agent solution
        try:
            standard_response = self._call_solution_api(question_text, use_similar_questions=False)
            result.standard_explanation = standard_response['data']['explanation']
            result.standard_answer = standard_response['data']['final_answer']
            result.standard_processing_time = standard_response['processing_time_ms']
        except Exception as e:
            result.standard_error = str(e)
            logger.error(f"Standard agent failed for {question_id}: {e}")

        # Get similar questions agent solution
        try:
            sim_response = self._call_solution_api(question_text, use_similar_questions=True)
            result.sim_explanation = sim_response['data']['explanation']
            result.sim_answer = sim_response['data']['final_answer']
            result.sim_processing_time = sim_response['processing_time_ms']
        except Exception as e:
            result.sim_error = str(e)
            logger.error(f"Similar questions agent failed for {question_id}: {e}")

        # Perform comparative analysis if both solutions succeeded
        if result.standard_explanation and result.sim_explanation:
            try:
                analysis_response = self._call_analysis_api(question_text, standard_response, sim_response)
                result.sim_score = analysis_response['data']['sim_answer_score']
                result.standard_score = analysis_response['data']['no_sim_answer_score']
                result.analysis_notes = analysis_response['data']['notes']
            except Exception as e:
                result.analysis_error = str(e)
                logger.error(f"Analysis failed for {question_id}: {e}")
        else:
            result.analysis_error = "Skipped due to failed solution generation"

        return result

    def run_evaluation(self, data_file: str = 'similar_question_data.json'):
        """Run evaluation on all questions with concurrent processing"""
        
        # Load data
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)[:5]
            logger.info(f"Loaded {len(data)} questions from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            return

        # Filter out already processed questions
        remaining_data = [q for q in data if q['question_id'] not in self.processed_ids]
        logger.info(f"Processing {len(remaining_data)} remaining questions (skipping {len(self.processed_ids)} already processed)")

        if not remaining_data:
            logger.info("All questions have already been processed!")
            return

        # Process questions concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self._process_question, question_data): question_data['question_id']
                for question_data in remaining_data
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(remaining_data), desc="Processing questions") as pbar:
                for future in as_completed(future_to_question):
                    question_id = future_to_question[future]
                    try:
                        result = future.result()
                        self._save_result(result)
                        self.processed_ids.add(question_id)
                        
                        # Update progress bar description with latest result
                        if result.sim_score and result.standard_score:
                            pbar.set_postfix({
                                'ID': question_id[:6],
                                'Sim': result.sim_score,
                                'Std': result.standard_score
                            })
                        else:
                            pbar.set_postfix({'ID': question_id[:6], 'Status': 'Error'})
                            
                    except Exception as e:
                        logger.error(f"Failed to process question {question_id}: {e}")
                        
                        # Save error result
                        error_result = EvaluationResult(
                            question_id=question_id,
                            question_text="Error loading question",
                            subject="Unknown",
                            standard_error=f"Processing failed: {str(e)}",
                            sim_error=f"Processing failed: {str(e)}",
                            analysis_error=f"Processing failed: {str(e)}"
                        )
                        self._save_result(error_result)
                    
                    pbar.update(1)

        logger.info(f"Evaluation completed! Results saved to {self.output_file}")
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics"""
        try:
            df = pd.read_csv(self.output_file)
            total = len(df)
            successful_analysis = len(df[df['sim_score'].notna() & df['standard_score'].notna()])
            
            if successful_analysis > 0:
                avg_sim_score = df['sim_score'].mean()
                avg_standard_score = df['standard_score'].mean()
                sim_wins = len(df[df['sim_score'] > df['standard_score']])
                standard_wins = len(df[df['standard_score'] > df['sim_score']])
                ties = len(df[df['sim_score'] == df['standard_score']])
                
                print(f"\n{'='*50}")
                print(f"EVALUATION SUMMARY")
                print(f"{'='*50}")
                print(f"Total questions processed: {total}")
                print(f"Successful analyses: {successful_analysis}")
                print(f"Average similar questions score: {avg_sim_score:.2f}")
                print(f"Average standard score: {avg_standard_score:.2f}")
                print(f"Similar questions agent wins: {sim_wins}")
                print(f"Standard agent wins: {standard_wins}")
                print(f"Ties: {ties}")
                print(f"{'='*50}")
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

def main():
    # Configuration
    SOLUTION_API_URL = "http://localhost:8000"  # Your unified solution builder
    ANALYSIS_API_URL = "http://localhost:8003"   # Your comparative analysis API
    OUTPUT_FILE = "evaluation_results.csv"
    MAX_WORKERS = 4  # Adjust based on your system and API limits
    
    # Test API connectivity
    try:
        health_check_solution = requests.get(f"{SOLUTION_API_URL}/health", timeout=10)
        health_check_analysis = requests.get(f"{ANALYSIS_API_URL}/health", timeout=10)
        
        if health_check_solution.status_code != 200:
            raise Exception(f"Solution API not healthy: {health_check_solution.status_code}")
        if health_check_analysis.status_code != 200:
            raise Exception(f"Analysis API not healthy: {health_check_analysis.status_code}")
            
        logger.info("Both APIs are healthy and ready")
        
    except Exception as e:
        logger.error(f"API connectivity check failed: {e}")
        return

    # Run evaluation
    runner = EvaluationRunner(
        solution_api_url=SOLUTION_API_URL,
        analysis_api_url=ANALYSIS_API_URL,
        output_file=OUTPUT_FILE,
        max_workers=MAX_WORKERS
    )
    
    try:
        runner.run_evaluation()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()