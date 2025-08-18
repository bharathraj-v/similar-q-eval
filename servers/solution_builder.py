#!/usr/bin/env python3
# UNIFIED SOLUTION BUILDER API

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

solution_llm = None
solution_agent = None
solution_agent_with_sim = None
config_data = None
config_data_sim = None

def load_config() -> Dict[str, Any]:
    try:
        with open('configs/solution_builder_prompts.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("configs/solution_builder_prompts.yaml not found. Please ensure it exists.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing solution_builder_prompts.yaml: {str(e)}")
        raise

def load_config_sim() -> Dict[str, Any]:
    try:
        with open('configs/solution_builder_prompts_sim.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("configs/solution_builder_prompts_sim.yaml not found. Please ensure it exists.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing solution_builder_prompts_sim.yaml: {str(e)}")
        raise

def validate_environment():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = config_data['messages']['errors']['missing_api_key'] if config_data else "GEMINI_API_KEY is not set."
        raise ValueError(error_msg)
    return api_key

class SimilarQuestion(BaseModel):
    similar_question_text: str
    similarity_score: float
    summarized_solution_approach: str

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = "1.0.0"

class SolutionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The academic question to be solved."
    )
    use_similar_questions: bool = Field(
        default=False,
        description="Whether to use similar questions for enhanced solution building."
    )

    @field_validator('question')
    @classmethod
    def validate_not_empty_after_strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question field cannot be empty or contain only whitespace')
        return v.strip()

class SolutionModel(BaseModel):
    explanation: str
    final_answer: str

class SolutionResponse(BaseModel):
    success: bool = True
    data: SolutionModel
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    request_id: Optional[str] = None
    agent_used: str = Field(..., description="Which agent was used for the solution")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    request_id: Optional[str] = None

async def get_similar_questions(question: str) -> List[SimilarQuestion]:
    try:
        with open('similar_question_data.json') as f:
            data = json.load(f)

        if not data:
            return []

        best_match = max(
            data,
            key=lambda q: SequenceMatcher(None, q.get('question_text', ''), question).ratio()
        )
        
        return [SimilarQuestion(**sq) for sq in best_match.get('similar_questions', [])]
    except FileNotFoundError:
        logger.error("similar_question_data.json not found in the root directory.")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error processing similar_question_data.json: {e}")
        return []

def initialize_llm():
    global solution_llm
    try:
        api_key = validate_environment()
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        )
        solution_llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            generation_config=config,
        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        raise

def initialize_agents():
    global solution_agent, solution_agent_with_sim
    if solution_llm is None:
        raise RuntimeError("LLM must be initialized before the agents.")
    
    try:
        agent_config = config_data['agent']['solution_builder']
        solution_agent = FunctionAgent(
            name="SolutionAgent",
            description="Builds thorough solutions for academic problems.",
            system_prompt=agent_config['system_prompt'],
            llm=solution_llm,
            output_cls=SolutionModel,
            timeout=120,
            tools=[]
        )
        
        agent_config_sim = config_data_sim['agent']['solution_builder']
        solution_agent_with_sim = FunctionAgent(
            name="SolutionAgentWithSim",
            description="It builds thorough solutions to the given problem by referencing similar questions.",
            system_prompt=agent_config_sim['system_prompt'],
            llm=solution_llm,
            output_cls=SolutionModel,
            timeout=120,
            tools=[get_similar_questions]
        )
        
        logger.info("Both agents initialized successfully")
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config_data, config_data_sim
    logger.info("Starting up Unified Solution Builder API")
    try:
        config_data = load_config()
        config_data_sim = load_config_sim()
        initialize_llm()
        initialize_agents()
        logger.info("Unified Solution Builder API started successfully")
    except Exception as e:
        logger.error(f"Critical startup failure: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down Unified Solution Builder API")

def create_app():
    temp_config = load_config()
    api_config = temp_config['api']
    return FastAPI(
        title="Unified Solution Builder API",
        description="API for solving academic problems with optional similar question enhancement",
        version=api_config['version'],
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = f"{int(time.time())}-{id(request)}"
    request.state.request_id = request_id
    start_time = time.time()
    logger.info(f"Request {request_id}: Received {request.method} {request.url}")
    response = await call_next(request)
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Request {request_id}: Completed in {processing_time:.2f}ms with status {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Request {request_id}: Unhandled exception: {str(exc)}", exc_info=True)
    error_msg = config_data['messages']['errors']['internal_error'] if config_data else "An internal server error occurred."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=error_msg,
            error_code="INTERNAL_ERROR",
            request_id=request_id
        ).model_dump()
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        timestamp=int(time.time()),
        version=config_data['api']['version'] if config_data else "1.0.0"
    )

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Unified Solution Builder API",
        "version": config_data['api']['version'] if config_data else "1.0.0",
        "docs": "/docs"
    }

@app.post("/solve", response_model=SolutionResponse, tags=["Solution"])
async def get_solution(request_data: SolutionRequest, request: Request):
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        if request_data.use_similar_questions:
            logger.info(f"Request {request_id}: Processing with similar questions agent")
            if solution_agent_with_sim is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=config_data_sim['messages']['errors']['service_unavailable']
                )
            
            template = config_data_sim['agent']['solution_builder']['user_message_template']
            user_msg = template.format(question=request_data.question)
            agent_output = await solution_agent_with_sim.run(user_msg=user_msg)
            agent_used = "solution_agent_with_similar_questions"
        else:
            logger.info(f"Request {request_id}: Processing with standard agent")
            if solution_agent is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=config_data['messages']['errors']['service_unavailable']
                )
            
            template = config_data['agent']['solution_builder']['user_message_template']
            user_msg = template.format(question=request_data.question)
            agent_output = await solution_agent.run(user_msg=user_msg)
            agent_used = "standard_solution_agent"
        
        result = agent_output.get_pydantic_model(SolutionModel)
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Request {request_id}: Completed successfully in {processing_time}ms using {agent_used}")
        
        return SolutionResponse(
            data=result,
            processing_time_ms=processing_time,
            request_id=request_id,
            agent_used=agent_used
        )
        
    except ValueError as e:
        active_config = config_data_sim if request_data.use_similar_questions else config_data
        error_msg = active_config['messages']['errors']['validation_error'].format(error=str(e))
        logger.warning(f"Request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        active_config = config_data_sim if request_data.use_similar_questions else config_data
        error_msg = active_config['messages']['solution']['failure'].format(error=str(e))
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
    }
    
    logger.info(f"Starting Uvicorn server with config: {server_config}")
    uvicorn.run("solution_builder:app", **server_config)