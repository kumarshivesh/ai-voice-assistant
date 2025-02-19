# main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from openai import OpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from typing import Dict, Any
from enum import Enum


# psql -h localhost -p 5432 -U kumarshivesh -d prototype_voice_assistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Intent Categories
class IntentCategory(str, Enum):
    # Greeting and Farewell
    GREETING = "greeting"
    FAREWELL = "farewell"
    
    # Questions and Information
    GENERAL_QUESTION = "general_question"
    WEATHER_QUERY = "weather_query"
    TIME_QUERY = "time_query"
    LOCATION_QUERY = "location_query"
    
    # Programming and Technical
    CODE_REQUEST = "code_request"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUG = "code_debug"
    TECHNICAL_QUESTION = "technical_question"
    
    # Task Management
    TASK_REMINDER = "task_reminder"
    TASK_CREATE = "task_create"
    TASK_UPDATE = "task_update"
    TASK_DELETE = "task_delete"
    
    # Help and Support
    HELP_REQUEST = "help_request"
    CLARIFICATION = "clarification"
    ERROR_REPORT = "error_report"
    
    # Calculations and Math
    MATH_CALCULATION = "math_calculation"
    CONVERSION_REQUEST = "conversion_request"
    
    # System and Settings
    SYSTEM_STATUS = "system_status"
    SETTINGS_CHANGE = "settings_change"
    
    # Fallback
    UNKNOWN = "unknown"

class IntentClassificationError(Exception):
    """Custom exception for intent classification errors"""
    pass

# Enhanced database initialization
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                user_input TEXT NOT NULL,
                intent TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence FLOAT,
                error_occurred BOOLEAN DEFAULT FALSE,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    init_db()
    yield
    # Shutdown: Clean up any resources if needed
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="AI Voice Assistant API", lifespan=lifespan)

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_HOST = os.getenv("DB_HOST", "db")
    DB_NAME = os.getenv("DB_NAME", "prototype_voice_assistant")
    DB_USER = os.getenv("DB_USER", "kumarshivesh")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Initialize OpenAI client
client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=Config.DB_HOST,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD
    )

# Models
class UserInput(BaseModel):
    text: str

class AssistantResponse(BaseModel):
    intent: str
    response: str
    confidence: float

# Intent recognition using OpenAI
async def recognize_intent(text: str) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are an advanced intent recognition system. Analyze user input and respond in EXACTLY this format:

INTENT
CONFIDENCE_SCORE
RESPONSE

Choose the most appropriate intent from these categories:

1. Greeting and Farewell:
   - greeting: General greetings
   - farewell: Goodbye messages

2. Questions and Information:
   - general_question: Generic queries
   - weather_query: Weather-related questions
   - time_query: Time/date queries
   - location_query: Location-based questions

3. Programming and Technical:
   - code_request: Requests for code examples
   - code_explanation: Requests to explain code
   - code_debug: Help with debugging
   - technical_question: Technical queries

4. Task Management:
   - task_reminder: Setting reminders
   - task_create: Creating new tasks
   - task_update: Updating existing tasks
   - task_delete: Removing tasks

5. Help and Support:
   - help_request: General help requests
   - clarification: Asking for clarification
   - error_report: Reporting issues

6. Calculations and Math:
   - math_calculation: Mathematical operations
   - conversion_request: Unit conversions

7. System and Settings:
   - system_status: System state queries
   - settings_change: Configuration changes

Examples:

User: "Hello, how are you?"
Response:
greeting
0.95
Hi! How can I help you today?

User: "Write a Python function to calculate factorial"
Response:
code_request
0.98
I'll help you write a factorial function in Python.

User: "What's wrong with my for loop?"
Response:
code_debug
0.90
I'll help you identify and fix the issues in your for loop.

User: "Convert 5 kilometers to miles"
Response:
conversion_request
0.95
I'll help you convert 5 kilometers to miles.

User: "Explain how binary search works"
Response:
code_explanation
0.92
I'll explain the binary search algorithm and how it works.

User: "Set a reminder for my meeting tomorrow"
Response:
task_reminder
0.94
I'll help you set a reminder for your meeting."""},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        # Parse OpenAI response
        analysis = response.choices[0].message.content.strip().split('\n')
        analysis = [line for line in analysis if line.strip()]
        
        if len(analysis) != 3:
            raise IntentClassificationError("Invalid response format from model")
            
        intent = analysis[0].strip()
        
        # Validate intent category
        if intent not in [category.value for category in IntentCategory]:
            logger.warning(f"Unknown intent category received: {intent}")
            intent = IntentCategory.UNKNOWN.value
        
        # Parse confidence with error handling
        try:
            confidence = float(analysis[1].strip())
            if not 0 <= confidence <= 1:
                raise ValueError("Confidence score out of range")
        except ValueError as e:
            logger.warning(f"Invalid confidence score: {str(e)}")
            confidence = 0.7
            
        return {
            "intent": intent,
            "confidence": confidence,
            "response": analysis[2].strip(),
            "error_occurred": False,
            "error_message": None
        }
        
    except IntentClassificationError as e:
        logger.error(f"Intent classification error: {str(e)}")
        return {
            "intent": IntentCategory.UNKNOWN.value,
            "confidence": 0.5,
            "response": "I'm having trouble understanding your request. Could you please rephrase it?",
            "error_occurred": True,
            "error_message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in intent recognition: {str(e)}")
        return {
            "intent": IntentCategory.UNKNOWN.value,
            "confidence": 0.3,
            "response": "I apologize, but I encountered an unexpected error. Please try again.",
            "error_occurred": True,
            "error_message": str(e)
        }

# Store interaction with error handling
async def store_interaction(user_input: str, result: Dict[str, Any]):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO interactions 
            (user_input, intent, response, confidence, error_occurred, error_message)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user_input,
            result["intent"],
            result["response"],
            result["confidence"],
            result["error_occurred"],
            result["error_message"]
        ))
        
        conn.commit()
        logger.info(f"Stored interaction with intent: {result['intent']}")
    except Exception as e:
        logger.error(f"Failed to store interaction: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

# Routes
@app.post("/process", response_model=AssistantResponse)
async def process_input(user_input: UserInput):
    try:
        # Recognize intent
        result = await recognize_intent(user_input.text)
        
        # Store interaction
        await store_interaction(user_input.text, result)
        
        return AssistantResponse(
            intent=result["intent"],
            response=result["response"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interactions")
async def get_interactions():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 100")
    interactions = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return interactions

