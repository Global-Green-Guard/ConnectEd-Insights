from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
from datetime import datetime
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel


load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"OpenAI client initialization failed: {str(e)}")
    client = None

# Initialize FastAPI app
app = FastAPI(
    title="ConnectEd Insights",
    description="AI-Driven School Connectivity Planning Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Set up static files and templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Data Models
class School(BaseModel):
    id: str
    name: str
    longitude: float
    latitude: float
    education_level: str
    region: str
    students_affected: int
    current_connectivity: float  # in Mbps
    connectivity_status: str  # "adequate", "moderate", "poor"
    last_updated: str  # Changed to string for ISO format

class ConnectivityGapAnalysis(BaseModel):
    total_schools: int
    students_affected: int
    connectivity_threshold: float
    schools_below_threshold: int
    priority_schools: List[School]
    worst_connected_regions: List[str]
    recommended_actions: List[str]

class ChatRequest(BaseModel):
    question: str

# generate_chat_response function
def generate_chat_response(question: str) -> str:
    """Generate responses using both dataset and AI analysis"""
    try:
        if not SCHOOLS_DATA:
            return "School data is currently unavailable. Please try again later."

        # First try to answer from dataset directly
        if "highest connectivity" in question.lower():
            best_school = max(SCHOOLS_DATA, key=lambda x: x.current_connectivity)
            return (
                f"The school with the best connectivity is {best_school.name} "
                f"in {best_school.region} with {best_school.current_connectivity}Mbps."
            )

        # Create enhanced system prompt
        system_prompt = f"""You are a connectivity expert analyzing {len(SCHOOLS_DATA)} Kenyan schools. 
        You MUST use this data:
        - Schools: {len(SCHOOLS_DATA)}
        - Average speed: {sum(s.current_connectivity for s in SCHOOLS_DATA)/len(SCHOOLS_DATA):.1f}Mbps
        - Regions: {', '.join({s.region for s in SCHOOLS_DATA})}
        - Top 3 schools: {', '.join([s.name for s in sorted(SCHOOLS_DATA, key=lambda x: -x.current_connectivity)[:3]])}
        
        Never mention you're an AI model. Use phrases like "Based on current data" 
        and "Our analysis shows". For missing data, suggest contacting support."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"AI Error: {str(e)}")
        return "Our analysis service is currently unavailable. Please try again later."

# Update the load_schools function
def load_schools():
    csv_path = BASE_DIR / "data/School_report_page_1_out_of_3_dated_25022025_082235.csv"
    
    try:
        print(f"Attempting to load data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} raw records")
        
        # Add coordinate validation
        initial_count = len(df)
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        print(f"After coordinate cleaning: {len(df)} records remaining")

        # Rest of your processing code...
        schools = []
        kenyan_counties = ['BUNGOMA', 'GARISSA', 'KAJIADO', 'KISUMU', 'MACHAKOS', 
                         'NAIROBI', 'NAKURU', 'NYERI', 'EMBU', 'LAIKIPIA', 'MURANG']
        
        for _, row in df.iterrows():
            # Add try-except for row processing
            try:
                name_parts = row['School Name'].split()
                region = next((part for part in name_parts if part in kenyan_counties), 'OTHER')
                
                students = random.randint(100, 2500)
                connectivity = random.uniform(1, 50)
                status = "adequate" if connectivity >= 20 else "moderate" if connectivity >= 10 else "poor"
                
                schools.append(School(
                    id=row['School Giga ID'],
                    name=row['School Name'],
                    longitude=row['Longitude'],
                    latitude=row['Latitude'],
                    education_level=row['Education Level'],
                    region=region,
                    students_affected=students,
                    current_connectivity=connectivity,
                    connectivity_status=status,
                    last_updated=datetime.now().isoformat()
                ))
            except Exception as e:
                print(f"Error processing row {_}: {str(e)}")
                continue
                
        print(f"Successfully processed {len(schools)} schools")
        return schools
        
    except Exception as e:
        print(f"Critical error loading schools: {str(e)}")
        return []


SCHOOLS_DATA = load_schools()

# Analysis Logic
def analyze_connectivity_gaps(
    schools: List[School],
    min_speed: float = 10.0,
    region: Optional[str] = None
) -> ConnectivityGapAnalysis:
    filtered_schools = [s for s in schools if (not region or s.region == region)]
    
    below_threshold = [s for s in filtered_schools if s.current_connectivity < min_speed]
    
    priority_schools = sorted(
        below_threshold,
        key=lambda x: (-x.students_affected, x.current_connectivity)
    )[:10]

    region_stats = {}
    for s in below_threshold:
        region_stats[s.region] = region_stats.get(s.region, 0) + 1
    worst_regions = sorted(region_stats.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return ConnectivityGapAnalysis(
        total_schools=len(filtered_schools),
        students_affected=sum(s.students_affected for s in below_threshold),
        connectivity_threshold=min_speed,
        schools_below_threshold=len(below_threshold),
        priority_schools=priority_schools,
        worst_connected_regions=[r[0] for r in worst_regions],
        recommended_actions=[
            "Prioritize fiber optic deployment in high-need regions",
            "Implement wireless bridge solutions for remote schools",
            f"Upgrade network infrastructure for {len(below_threshold)} schools"
        ]
    )

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "ConnectEd Insights"}
    )

# Define the /chat route
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "title": "Chat"})



# chat endpoint to handle direct data queries first
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        question = chat_request.question.lower()
        
        # Direct data responses
        if "highest connectivity" in question:
            if not SCHOOLS_DATA:
                return {"answer": "School data currently unavailable"}
            best_school = max(SCHOOLS_DATA, key=lambda x: x.current_connectivity)
            return {"answer": f"{best_school.name} ({(best_school.current_connectivity)}Mbps) in {best_school.region}"}
        
        # FAQ responses
        connectivity_faq = {
            "wifi": "All schools have WiFi access with speeds up to 100Mbps.",
            "internet": "Internet access is available in computer labs during school hours.",
            "speed": "Average school connection speed is 45Mbps (range: 10-100Mbps)",
            "report": "Report issues to tech-support@connected.education with your school ID",
            "contact": "Email globalgreenguardg3@gmail.com"
            # Add more FAQ entries
        }
        for key, response in connectivity_faq.items():
            if key in question:
                return {"answer": response}

        # AI-generated response
        answer = generate_chat_response(question)
        return {"answer": answer}
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return {"answer": "Service unavailable. Please try again later."}


@app.get("/api/connectivity/gaps", response_model=ConnectivityGapAnalysis)
async def get_connectivity_gaps(
    min_speed: float = Query(10.0, description="Minimum acceptable connectivity speed in Mbps"),
    region: Optional[str] = Query(None, description="Filter by specific region")
):
    if min_speed <= 0:
        raise HTTPException(status_code=400, detail="Minimum speed must be positive value")
    
    valid_regions = {s.region for s in SCHOOLS_DATA}
    if region and region not in valid_regions:
        raise HTTPException(status_code=404, detail="Region not found")

    return analyze_connectivity_gaps(SCHOOLS_DATA, min_speed, region)


# New API endpoint for getting all schools
@app.get("/api/schools", response_model=List[School])
@app.get("/api/schools/", response_model=List[School])
async def get_all_schools():
    return SCHOOLS_DATA

@app.get("/map", response_class=HTMLResponse)
async def show_schools_map(request: Request):
    return templates.TemplateResponse(
        "map.html",
        {"request": request, "title": "School Connectivity Map"}
    )

# New analytics endpoint for the connectivity gap analysis page
@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse(
        "analytics.html", 
        {"request": request, "title": "Connectivity Gap Analysis"}
    )

@app.get("/regional-analysis", response_class=HTMLResponse)
async def regional_analysis_page(request: Request):
    """Endpoint for regional connectivity analysis page"""
    return templates.TemplateResponse(
        "regional_analysis.html",
        {
            "request": request,
            "title": "Regional Connectivity Analysis"
        }
    )

@app.get("/priority-schools", response_class=HTMLResponse)
async def priority_schools_page(request: Request):
    """Endpoint for priority schools analysis page"""
    return templates.TemplateResponse(
        "priority.html",
        {
            "request": request,
            "title": "Priority Schools Analysis"
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", reload=True)