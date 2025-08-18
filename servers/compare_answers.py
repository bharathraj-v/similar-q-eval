#!/usr/bin/env python3
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

comparison_llm = None
comparison_agent = None
config_data = None

def load_config() -> Dict[str, Any]:
    try:
        with open('configs/comparitive_analysis_prompts.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("configs/comparitive_analysis_prompts.yaml not found. Please ensure it exists.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing comparitive_analysis_prompts.yaml: {str(e)}")
        raise

def validate_environment():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = config_data['messages']['errors']['missing_api_key'] if config_data else "GEMINI_API_KEY is not set."
        raise ValueError(error_msg)
    return api_key

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = "1.0.0"

class AnalysisRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=5000)
    sim_answer_explanation: str = Field(..., min_length=10, max_length=10000)
    sim_answer_final_answer: str = Field(..., min_length=1, max_length=1000)
    no_sim_answer_explanation: str = Field(..., min_length=10, max_length=10000)
    no_sim_answer_final_answer: str = Field(..., min_length=1, max_length=1000)

    @field_validator('*')
    @classmethod
    def validate_not_empty_after_strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Field cannot be empty or contain only whitespace')
        return v.strip()

class ComparitiveAnalysis(BaseModel):
    sim_answer_score: int = Field(..., ge=0, le=100)
    no_sim_answer_score: int = Field(..., ge=0, le=100)
    notes: str

class AnalysisResponse(BaseModel):
    success: bool = True
    data: ComparitiveAnalysis
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    request_id: Optional[str] = None

def initialize_llm():
    global comparison_llm
    try:
        api_key = validate_environment()
        config = types.GenerateContentConfig(max_output_tokens=8192)
        comparison_llm = GoogleGenAI(
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
    global comparison_agent
    if comparison_llm is None:
        raise RuntimeError("LLM must be initialized before the agent.")
    try:
        agent_config = config_data['agent']['comparitive_analyzer']
        comparison_agent = FunctionAgent(
            name="ComparitiveAnalysisAgent",
            description="It evaluates the comparitive analysis of the given question",
            system_prompt=agent_config['system_prompt'],
            llm=comparison_llm,
            output_cls=ComparitiveAnalysis,
            timeout=120,
            tools=[]
        )
        logger.info(config_data['messages']['startup']['agent_init'])
    except Exception as e:
        error_msg = config_data['messages']['startup']['failure'].format(error=str(e))
        logger.error(error_msg)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config_data
    logger.info("Starting up Comparative Analysis API")
    try:
        config_data = load_config()
        initialize_llm()
        initialize_agent()
        logger.info(config_data['messages']['startup']['success'])
    except Exception as e:
        logger.error(f"Critical startup failure: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down Comparative Analysis API")

def create_app():
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
        "message": config_data['api']['title'],
        "version": config_data['api']['version'],
        "docs": "/docs"
    }

@app.post("/analyse", response_model=AnalysisResponse, tags=["Analysis"])
async def get_analysis(request_data: AnalysisRequest, request: Request):
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        logger.info(config_data['messages']['analysis']['start'].format(request_id=request_id))
        
        if comparison_agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=config_data['messages']['errors']['service_unavailable']
            )
        
        template = config_data['agent']['comparitive_analyzer']['user_message_template']
        user_msg = template.format(
            question=request_data.question,
            sim_answer_explanation=request_data.sim_answer_explanation,
            sim_answer_final_answer=request_data.sim_answer_final_answer,
            no_sim_answer_explanation=request_data.no_sim_answer_explanation,
            no_sim_answer_final_answer=request_data.no_sim_answer_final_answer
        )
        
        agent_output = await comparison_agent.run(user_msg=user_msg)
        
        result = agent_output.get_pydantic_model(ComparitiveAnalysis)
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(config_data['messages']['analysis']['success'].format(request_id=request_id, processing_time=processing_time))
        
        return AnalysisResponse(
            data=result,
            processing_time_ms=processing_time,
            request_id=request_id
        )
        
    except ValueError as e:
        error_msg = config_data['messages']['errors']['validation_error'].format(error=str(e))
        logger.warning(f"Request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        error_msg = config_data['messages']['analysis']['failure'].format(error=str(e))
        logger.error(f"Request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

if __name__ == "__main__":
    server_config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8003)),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
    }
    
    logger.info(f"Starting Uvicorn server with config: {server_config}")
    uvicorn.run("compare_answers:app", **server_config)
