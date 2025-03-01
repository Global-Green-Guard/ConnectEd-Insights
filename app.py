from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
from datetime import datetime
import random

# Initialize FastAPI app
app = FastAPI(
    title="ConnectEd Insights",
    description="AI-Driven School Connectivity Planning Platform",
    version="1.0.0"
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

# Load and process school data
def load_schools():
    csv_path = BASE_DIR / "data/School_report_page_1_out_of_3_dated_25022025_082235.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"School data file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    schools = []
    kenyan_counties = ['BUNGOMA', 'GARISSA', 'KAJIADO', 'KISUMU', 'MACHAKOS', 
                      'NAIROBI', 'NAKURU', 'NYERI', 'EMBU', 'LAIKIPIA', 'MURANG']
    
    for _, row in df.iterrows():
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
            last_updated=datetime.now().isoformat()  # ISO formatted string
        ))
    return schools

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
async def get_all_schools():
    return SCHOOLS_DATA

@app.get("/map", response_class=HTMLResponse)
async def show_schools_map(request: Request):
    return templates.TemplateResponse(
        "map.html",
        {"request": request}
    )

# Other endpoints remain the same...

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", reload=True)