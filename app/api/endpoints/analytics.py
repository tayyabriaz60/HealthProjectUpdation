"""
Analytics and Dashboard API endpoints.
Provides weekly glucose data for patient dashboards.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db import get_db
from app.models import GlucoseReading

# Router for analytics endpoints
api_router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@api_router.get("/glucose/weekly")
async def get_weekly_glucose(
    user_id: str = Query(..., description="Firebase user ID to get glucose data for"),
    days: int = Query(7, ge=1, le=30, description="Number of days to retrieve (default: 7, max: 30)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get weekly glucose data for a patient dashboard.
    
    Returns day-wise glucose values suitable for Flutter bar charts.
    
    **Response Format:**
    ```json
    {
        "user_id": "firebase_uid",
        "period_days": 7,
        "start_date": "2025-01-01",
        "end_date": "2025-01-07",
        "daily_data": [
            {
                "date": "2025-01-01",
                "day_name": "Monday",
                "average_value": 5.2,
                "unit": "mmol/L",
                "reading_count": 3,
                "readings": [
                    {"value": 5.2, "unit": "mmol/L", "taken_at": "2025-01-01T08:00:00"}
                ]
            },
            ...
        ]
    }
    ```
    
    **Usage:**
    - Default (last 7 days): `GET /api/analytics/glucose/weekly?user_id=YOUR_UID`
    - Custom period: `GET /api/analytics/glucose/weekly?user_id=YOUR_UID&days=14`
    """
    try:
        # Calculate date range (last N days including today)
        today = datetime.utcnow().date()
        dates_to_show = [today - timedelta(days=i) for i in range(days)]
        dates_to_show.sort()  # Ascending order
        
        start_date_limit = dates_to_show[0]
        
        # Query glucose readings for this user within the date range
        # Start from the beginning of the first day to capture all relevant readings
        query_start = datetime.combine(start_date_limit, datetime.min.time())
        query_end = datetime.utcnow()
        
        stmt = select(GlucoseReading).where(
            and_(
                GlucoseReading.user_id == user_id,
                GlucoseReading.taken_at >= query_start,
                GlucoseReading.taken_at <= query_end
            )
        ).order_by(GlucoseReading.taken_at.asc())
        
        result = await db.execute(stmt)
        readings = result.scalars().all()
        
        # Determine unit (default to mmol/L if no readings found)
        unit = "mmol/L"
        if readings:
            unit = readings[0].unit
        
        # Group readings by day
        daily_groups = {}
        for reading in readings:
            # Get date (YYYY-MM-DD) as key
            date_key = reading.taken_at.date().isoformat()
            
            if date_key not in daily_groups:
                daily_groups[date_key] = []
            
            daily_groups[date_key].append(reading)
        
        # Build daily data, ensuring all days in range are included
        daily_data = []
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for d in dates_to_show:
            date_str = d.isoformat()
            day_name = day_names[d.weekday()]
            
            readings_for_day = daily_groups.get(date_str, [])
            
            if readings_for_day:
                values = [r.value for r in readings_for_day]
                average_value = sum(values) / len(values)
                day_unit = readings_for_day[0].unit
                
                reading_dicts = [
                    {
                        "value": r.value,
                        "unit": r.unit,
                        "taken_at": r.taken_at.isoformat()
                    } for r in readings_for_day
                ]
                
                daily_data.append({
                    "date": date_str,
                    "day_name": day_name,
                    "average_value": round(average_value, 2),
                    "unit": day_unit,
                    "reading_count": len(readings_for_day),
                    "readings": reading_dicts
                })
            else:
                # Add empty entry for days with no data
                daily_data.append({
                    "date": date_str,
                    "day_name": day_name,
                    "average_value": 0.0,
                    "unit": unit,
                    "reading_count": 0,
                    "readings": []
                })
        
        return {
            "user_id": user_id,
            "period_days": days,
            "start_date": dates_to_show[0].isoformat(),
            "end_date": dates_to_show[-1].isoformat(),
            "daily_data": daily_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving glucose data: {str(e)}"
        )


@api_router.get("/glucose/summary")
async def get_glucose_summary(
    user_id: str = Query(..., description="Firebase user ID"),
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary statistics for glucose readings.
    
    Returns overall stats like average, min, max, total readings.
    
    **Response Format:**
    ```json
    {
        "user_id": "firebase_uid",
        "period_days": 7,
        "total_readings": 15,
        "average_value": 5.3,
        "min_value": 4.8,
        "max_value": 6.1,
        "unit": "mmol/L"
    }
    ```
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query all readings for this period
        stmt = select(GlucoseReading).where(
            and_(
                GlucoseReading.user_id == user_id,
                GlucoseReading.taken_at >= start_date,
                GlucoseReading.taken_at <= end_date
            )
        )
        
        result = await db.execute(stmt)
        readings = result.scalars().all()
        
        if not readings:
            return {
                "user_id": user_id,
                "period_days": days,
                "total_readings": 0,
                "average_value": 0.0,
                "min_value": 0.0,
                "max_value": 0.0,
                "unit": "mmol/L",
                "message": "No glucose readings found for this period"
            }
        
        values = [r.value for r in readings]
        unit = readings[0].unit
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_readings": len(readings),
            "average_value": round(sum(values) / len(values), 2),
            "min_value": round(min(values), 2),
            "max_value": round(max(values), 2),
            "unit": unit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving glucose summary: {str(e)}"
        )
