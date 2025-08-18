#!/usr/bin/env python3
"""
Production-ready FastAPI server for Comparative Analysis with a Multi-Agent System
"""

import os
import logging
import time
import yaml
import json
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.genai import types
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.bridge.pydantic import BaseModel, Field, field_validator
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sub_agent_llm = None
orchestrator_llm = None
orchestrator = None
config_data = None

def load_config() -> Dict[str, Any]:
    """Load configuration from comparitive_analysis_prompts_mas.yaml"""
    try:
        with open('configs/comparitive_analysis_prompts_mas.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("configs/comparitive_analysis_prompts_mas.yaml file not found. Please ensure it exists in the current directory.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configs/comparitive_analysis_prompts_mas.yaml: {str(e)}")
        raise

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = "1.0.0"

class ComparisonRequest(BaseModel):
    """Request model for comparative analysis"""
    question: str = Field(
        ..., 
        min_length=5, 
        max_length=5000,
        description="The question to solve and compare answers for"
    )
    
    @field_validator('question')
    @classmethod
    def validate_not_empty_after_strip(cls, v: str) -> str:
        """Ensure field is not empty after stripping whitespace"""
        if not v.strip():
            raise ValueError('Question cannot be empty or contain only whitespace')
        return v.strip()

class SolutionModel(BaseModel):
    explanation: str
    final_answer: str

class SimilarQuestion(BaseModel):
    similar_question_text: str
    similarity_score: float
    summarized_solution_approach: str

class ComparitiveAnalysis(BaseModel):
    sim_answer_score: int
    no_sim_answer_score: int
    notes: str

class AgenticComparitiveAnalysis(BaseModel):
    sim_explanation: str
    sim_final_answer: str
    no_sim_explanation: str
    no_sim_final_answer: str
    sim_answer_score: int
    no_sim_answer_score: int
    notes: str

class ComparisonResponse(BaseModel):
    """Response model for comparative analysis"""
    success: bool = True
    data: AgenticComparitiveAnalysis
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: str
    request_id: Optional[str] = None

def validate_environment():
    """Validate required environment variables"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = config_data['messages']['errors']['missing_api_key']
        raise ValueError(error_msg)
    return api_key

def initialize_llms():
    """Initialize LLM instances"""
    global sub_agent_llm, orchestrator_llm
    
    try:
        api_key = validate_environment()
        
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=-1)  
        )
        
        sub_agent_llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            generation_config=config,
        )
        
        orchestrator_llm = GoogleGenAI(
            model="gemini-2.5-flash", 
            api_key=api_key,
            generation_config=config,
        )
        
        logger.info(config_data['messages']['startup']['llm_init'])
        
    except Exception as e:
        error_msg = config_data['messages']['startup']['failure'].format(error=str(e))
        logger.error(error_msg)
        raise

def initialize_agents():
    """Initialize all solution and comparison agents"""
    global orchestrator
    
    if sub_agent_llm is None or orchestrator_llm is None:
        raise RuntimeError("LLMs must be initialized before agents")
    
    try:
        agents_config = config_data['agents']
        
        # Initialize solution agents
        no_sim_solution_agent = FunctionAgent(
            name="NoSimSolutionAgent",
            description="Builds thorough solutions without using similar questions",
            system_prompt=agents_config['no_sim_solution']['system_prompt'],
            llm=sub_agent_llm,
            output_cls=SolutionModel,
            timeout=120,
            tools=[]
        )

        async def get_similar_questions(question: str) -> List[SimilarQuestion]:
            """Get most similar question from dataset and return its similar questions"""
            try:
                with open('similar_question_data.json') as f:
                    data = json.load(f)

                best_match = max(
                    data,
                    key=lambda q: SequenceMatcher(None, q['question_text'], question).ratio()
                )

                return [SimilarQuestion(**sq) for sq in best_match.get('similar_questions', [])]
            except FileNotFoundError:
                logger.warning("similar_question_data.json not found, returning empty list")
                return []
            except Exception as e:
                logger.error(f"Error loading similar questions: {str(e)}")
                return []

        sim_solution_agent = FunctionAgent(
            name="SimSolutionAgent", 
            description="Builds thorough solutions using similar questions",
            system_prompt=agents_config['sim_solution']['system_prompt'],
            llm=sub_agent_llm,
            output_cls=SolutionModel,
            timeout=120,
            tools=[get_similar_questions]
        )

        comparison_agent = FunctionAgent(
            name="ComparitiveAnalysisAgent",
            description="Evaluates the comparative analysis of solutions",
            system_prompt=agents_config['comparison']['system_prompt'],
            llm=sub_agent_llm,
            output_cls=ComparitiveAnalysis,
            timeout=120,
            tools=[]
        )

        async def solve_without_similar_questions(question: str) -> str:
            """
            Generates a solution for the given question using the standard solution builder agent.
            This agent does not use similar questions for context.
            Returns a solution object as a string.
            """
            result = await no_sim_solution_agent.run(user_msg=question)
            return str(result.structured_response)

        async def solve_with_similar_questions(question: str) -> str:
            """
            Generates a solution for the given question using the agent that can fetch
            and analyze similar questions via its tools.
            Returns a solution object as a string.
            """
            result = await sim_solution_agent.run(user_msg=question)
            return str(result.structured_response)

        async def evaluate_and_compare_solutions(
            question: str,
            sim_solution_explanation: str,
            sim_solution_final_answer: str,
            no_sim_solution_explanation: str,
            no_sim_solution_final_answer: str,
        ) -> str:
            """
            Compares and evaluates two different solutions for the same question.
            It takes the original question and the individual components (explanation and final answer)
            for both the 'similar question' and 'no similar question' based solutions.
            Returns a comparative analysis object as a string.
            """
            template = agents_config['comparison']['user_message_template']
            user_msg = template.format(
                question=question,
                sim_answer_explanation=sim_solution_explanation,
                sim_answer_final_answer=sim_solution_final_answer,
                no_sim_answer_explanation=no_sim_solution_explanation,
                no_sim_answer_final_answer=no_sim_solution_final_answer
            )
            result = await comparison_agent.run(user_msg=user_msg)
            return str(result.structured_response)

        orchestrator = FunctionAgent(
            system_prompt=agents_config['orchestrator']['system_prompt'],
            llm=orchestrator_llm,
            tools=[
                solve_without_similar_questions,
                solve_with_similar_questions,
                evaluate_and_compare_solutions,
            ],
            output_cls=AgenticComparitiveAnalysis,
        )
        
        logger.info(config_data['messages']['startup']['agents_init'])
        
    except Exception as e:
        error_msg = config_data['messages']['startup']['failure'].format(error=str(e))
        logger.error(error_msg)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global config_data
    logger.info("Starting up Comparative Analysis API")
    try:
        config_data = load_config()
        initialize_llms()
        initialize_agents()
        logger.info(config_data['messages']['startup']['success'])
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Comparative Analysis API")

def create_app():
    """Create FastAPI app with configuration from YAML"""
    temp_config = load_config()
    api_config = temp_config['api']
    
    return FastAPI(
        title=api_config['title'],
        description=api_config['description'],
        version=api_config['version'],
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this to the appropriate origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID and logging middleware"""
    request_id = f"{int(time.time())}-{id(request)}"
    request.state.request_id = request_id
    
    start_time = time.time()
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    response = await call_next(request)
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Request {request_id}: Completed in {processing_time:.2f}ms")
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Request {request_id}: Unhandled exception: {str(exc)}", exc_info=True)
    
    error_msg = config_data['messages']['errors']['internal_error'] if config_data else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=error_msg,
            error_code="INTERNAL_ERROR",
            request_id=request_id
        ).dict()
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        timestamp=int(time.time()),
        version=config_data['api']['version'] if config_data else "1.0.0"
    )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": config_data['api']['title'] if config_data else "Comparative Analysis API",
        "version": config_data['api']['version'] if config_data else "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/config", tags=["Configuration"])
async def get_comparison_config():
    """Get comparison configuration and descriptions"""
    if not config_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    return {
        "comparison_config": config_data['comparison_config'],
        "descriptions": config_data['comparison_config']['descriptions']
    }

@app.post("/compare", response_model=ComparisonResponse, tags=["Comparison"])
async def compare_answers(request_data: ComparisonRequest, request: Request):
    """
    Compare answers generated with and without similar questions for a given question
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        logger.info(config_data['messages']['comparison']['start'].format(request_id=request_id))
        
        if orchestrator is None:
            error_msg = config_data['messages']['errors']['service_unavailable']
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg
            )
        
        logger.info(f"Request {request_id}: Running orchestrator comparison for question: {request_data.question[:50]}...")
        orchestrator_response = await orchestrator.run(request_data.question)
        
        result = None
        
        if hasattr(orchestrator_response, 'get_pydantic_model'):
            try:
                result = orchestrator_response.get_pydantic_model(AgenticComparitiveAnalysis)
            except:
                result = orchestrator_response.get_pydantic_model()
        elif hasattr(orchestrator_response, 'structured_response'):
            result_dict = orchestrator_response.structured_response
            result = AgenticComparitiveAnalysis(**result_dict)
        else:
            raise ValueError(f"Unexpected orchestrator output type: {type(orchestrator_response)}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Request {request_id}: Comparison completed in {processing_time}ms")
        
        return ComparisonResponse(
            data=result,
            processing_time_ms=processing_time,
            request_id=request_id
        )
        
    except ValueError as e:
        error_msg = config_data['messages']['errors']['validation_error'].format(error=str(e))
        logger.error(f"Request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        error_msg = config_data['messages']['comparison']['failure'].format(error=str(e))
        logger.error(f"Request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

if __name__ == "__main__":
    server_config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8001)),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
        "workers": int(os.getenv("WORKERS", 1)) if os.getenv("ENVIRONMENT") == "production" else 1,
    }
    
    logger.info(f"Starting server with config: {server_config}")
    uvicorn.run("compare_answers_mas:app", **server_config)