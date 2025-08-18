#!/usr/bin/env python3
"""
Production-ready FastAPI server for Similar Questions Evaluation with a Single Agent System
"""

import os
import logging
import time
import yaml
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

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

evaluator_llm = None
evaluator_agent = None
config_data = None

def load_config() -> Dict[str, Any]:
    """Load configuration from prompts.yaml"""
    try:
        with open('configs/evaluation_prompts_s.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("configs/evaluation_prompts_s.yaml file not found. Please ensure it exists in the current directory.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configs/evaluation_prompts_s.yaml: {str(e)}")
        raise

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = "1.0.0"

class EvaluationRequest(BaseModel):
    """Request model for question evaluation"""
    question_text: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="The original question text to evaluate"
    )
    similar_question: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="The similar question text for comparison"
    )
    summarized_solution_approach: str = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="The solution approach for the similar question"
    )
    
    @field_validator('question_text', 'similar_question', 'summarized_solution_approach')
    @classmethod
    def validate_not_empty_after_strip(cls, v: str) -> str:
        """Ensure fields are not empty after stripping whitespace"""
        if not v.strip():
            raise ValueError('Field cannot be empty or contain only whitespace')
        return v.strip()

class SimilarQuestionsEvaluation(BaseModel):
    similar_question: str
    solution_approach: str
    conceptual_similarity_score: int = Field(..., ge=0, le=100)
    structural_similarity_score: int = Field(..., ge=0, le=100)
    difficulty_alignment_score: int = Field(..., ge=0, le=100)
    solution_approach_transferability_score: int = Field(..., ge=0, le=100)
    total_score: int = Field(..., ge=0, le=100)
    notes: str

class EvaluationResponse(BaseModel):
    """Response model for question evaluation"""
    success: bool = True
    data: SimilarQuestionsEvaluation
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

def initialize_llm():
    """Initialize LLM instance"""
    global evaluator_llm
    
    try:
        api_key = validate_environment()
        
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0)  
        )
        
        evaluator_llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            generation_config=config,
        )
        
        logger.info(config_data['messages']['startup']['llm_init'])
        
    except Exception as e:
        error_msg = config_data['messages']['startup']['failure'].format(error=str(e))
        logger.error(error_msg)
        raise

def initialize_agent():
    """Initialize the single evaluation agent"""
    global evaluator_agent
    
    if evaluator_llm is None:
        raise RuntimeError("LLM must be initialized before agent")
    
    try:
        agent_config = config_data['agent']['single_evaluator']
        
        evaluator_agent = FunctionAgent(
            system_prompt=agent_config['system_prompt'],
            llm=evaluator_llm,
            tools=[],
            output_cls=SimilarQuestionsEvaluation,
        )
        
        logger.info(config_data['messages']['startup']['agent_init'])
        
    except Exception as e:
        error_msg = config_data['messages']['startup']['failure'].format(error=str(e))
        logger.error(error_msg)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global config_data
    logger.info("Starting up Evaluation API (Single Agent)")
    try:
        config_data = load_config()
        initialize_llm()
        initialize_agent()
        logger.info(config_data['messages']['startup']['success'])
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Evaluation API (Single Agent)")

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
        "message": config_data['api']['title'] if config_data else "Similar Questions Evaluation API (Single Agent)",
        "version": config_data['api']['version'] if config_data else "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/config", tags=["Configuration"])
async def get_evaluation_config():
    """Get evaluation configuration and descriptions"""
    if not config_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    return {
        "evaluation_config": config_data['evaluation_config'],
        "descriptions": config_data['evaluation_config']['descriptions']
    }

@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_questions(request_data: EvaluationRequest, request: Request):
    """
    Evaluate the similarity and transferability between an original question and a similar question
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        logger.info(config_data['messages']['evaluation']['start'].format(request_id=request_id))
        
        if evaluator_agent is None:
            error_msg = config_data['messages']['errors']['service_unavailable']
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg
            )
        
        template = config_data['agent']['single_evaluator']['user_message_template']
        user_msg = template.format(
            original_question=request_data.question_text,
            similar_question=request_data.similar_question,
            solution_approach=request_data.summarized_solution_approach
        )
        
        logger.info(f"Request {request_id}: Running single agent evaluation")
        agent_output = await evaluator_agent.run(user_msg=user_msg)
        
        result = None
        
        if hasattr(agent_output, 'get_pydantic_model'):
            try:
                result = agent_output.get_pydantic_model(SimilarQuestionsEvaluation)
            except:
                result = agent_output.get_pydantic_model()
        elif hasattr(agent_output, 'structured_response'):
            result_dict = agent_output.structured_response
            result = SimilarQuestionsEvaluation(**result_dict)
        else:
            raise ValueError(f"Unexpected agent output type: {type(agent_output)}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Request {request_id}: Evaluation completed in {processing_time}ms")
        
        return EvaluationResponse(
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
        error_msg = config_data['messages']['evaluation']['failure'].format(error=str(e))
        logger.error(f"Request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

if __name__ == "__main__":
    server_config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8000)),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
        "workers": int(os.getenv("WORKERS", 1)) if os.getenv("ENVIRONMENT") == "production" else 1,
    }
    
    logger.info(f"Starting server with config: {server_config}")
    uvicorn.run("evaluate_similar_question:app", **server_config)