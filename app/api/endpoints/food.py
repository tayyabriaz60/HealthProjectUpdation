"""
Image analysis endpoint (auto-detect glucose vs food).
"""
import mimetypes
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.gemini_service import GeminiService
from app.core.config import BASE_DIR
from app.db import get_db
from app.models import ChatSession, Message, GlucoseReading, FoodEvent


api_router = APIRouter(prefix="/api/ai", tags=["ai"])

_gemini_service_instance = None


def get_gemini_service() -> GeminiService:
    """Get or create Gemini service instance."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance


def _save_image_to_disk(data: bytes, content_type: str) -> str:
    """Save uploaded image to media/chat_images and return relative path."""
    media_root = BASE_DIR / "media" / "chat_images"
    media_root.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(content_type) or ".jpg"
    file_name = f"{uuid.uuid4()}{ext}"
    rel_path = Path("chat_images") / file_name
    abs_path = media_root / file_name
    with open(abs_path, "wb") as f:
        f.write(data)
    return rel_path.as_posix()


async def _get_or_create_session(db: AsyncSession, chat_id: Optional[str], user_id: Optional[str]) -> ChatSession:
    session: Optional[ChatSession] = None
    if chat_id:
        session = await db.get(ChatSession, chat_id)
    if session is None:
        chat_id = chat_id or str(uuid.uuid4())
        session = ChatSession(id=chat_id, user_id=user_id)
        db.add(session)
    elif user_id and not session.user_id:
        session.user_id = user_id
    return session


@api_router.post("/analyze-image")
async def ai_analyze_image(
    image: UploadFile = File(...),
    health_context: Optional[str] = Query(None, description="Optional health context for food analysis"),
    chat_id: Optional[str] = Query(None, description="Optional chat session ID to attach this analysis to"),
    user_id: Optional[str] = Query(None, description="Optional user ID (e.g. Firebase UID)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Smart image analysis:
    - If the image is a glucose meter, returns glucose reading and analysis.
    - If the image is food, returns meal info, calories, and diabetes-friendly recommendation.
    Stores the image on disk and persists user+assistant messages in the database.
    """
    try:
        # Validate content type
        content_type = image.content_type
        if not content_type or content_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(image.filename or "")
            content_type = guessed or "image/jpeg"
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Image file is empty")

        # Save image to disk
        image_path = _save_image_to_disk(data, content_type)

        # Analyze with Gemini (auto-detect glucose vs food)
        gemini_service = get_gemini_service()
        
        # Auto-fetch latest AND average glucose reading if health_context is missing and user_id is provided
        if not health_context and user_id:
            from sqlalchemy import select, func
            from app.models import GlucoseReading as GRModel
            from datetime import datetime, timedelta
            
            # 1. Latest Reading
            stmt = (
                select(GRModel)
                .where(GRModel.user_id == user_id)
                .order_by(GRModel.taken_at.desc())
                .limit(1)
            )
            res = await db.execute(stmt)
            latest = res.scalar_one_or_none()
            
            context_parts = []
            if latest:
                context_parts.append(f"Latest glucose reading: {latest.value} {latest.unit} at {latest.taken_at}.")
                
                # 2. Average Reading (only if we have a latest reading to know the unit)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                avg_stmt = (
                    select(func.avg(GRModel.value))
                    .where(
                        GRModel.user_id == user_id,
                        GRModel.taken_at >= thirty_days_ago
                    )
                )
                avg_res = await db.execute(avg_stmt)
                avg_val = avg_res.scalar()
                
                if avg_val:
                    context_parts.append(f"Average glucose (last 30 days): {avg_val:.1f} {latest.unit}.")
            
            if context_parts:
                health_context = " ".join(context_parts)

        result = gemini_service.analyze_image_auto(
            image_data=data,
            mime_type=content_type,
            health_context=health_context
        )

        # Persist to DB (session + messages)
        session = await _get_or_create_session(db, chat_id, user_id)
        chat_id = session.id

        assistant_text = ""
        if result.get("type") == "glucose":
            reading = result.get("reading") or {}
            value = reading.get("value")
            unit = reading.get("unit")
            analysis = result.get("analysis") or ""
            parts = [
                "Detected: Glucose Meter",
                f"Glucose: {value} {unit}" if value is not None and unit else None,
                f"Analysis: {analysis}" if analysis else None,
            ]
            assistant_text = "\n".join([p for p in parts if p])
        elif result.get("type") == "food":
            meal = result.get("meal") or {}
            name = meal.get("meal_name") or "Unidentified Meal"
            calories = meal.get("calories")
            rec = result.get("recommendation") or ""
            parts = [
                "Detected: Food",
                f"Meal: {name}",
                f"Calories: {calories}" if calories is not None else None,
                f"Recommendation: {rec}" if rec else None,
            ]
            assistant_text = "\n".join([p for p in parts if p])
        else:
            assistant_text = "Could not detect if this is glucose meter or food."

        # Create message records and flush to get their IDs
        user_message = Message(
            chat_session_id=chat_id,
            role="user",
            text="",
            image_path=image_path,
        )
        assistant_message = Message(
            chat_session_id=chat_id,
            role="assistant",
            text=assistant_text,
        )
        db.add_all([user_message, assistant_message])
        await db.flush()

        # Store structured analytics
        if result.get("type") == "glucose":
            reading = result.get("reading") or {}
            value = reading.get("value")
            unit = reading.get("unit")
            if value is not None and unit is not None:
                glucose_row = GlucoseReading(
                    user_id=user_id,
                    chat_session_id=chat_id,
                    message_id=assistant_message.id,
                    image_path=image_path,
                    value=float(value),
                    unit=str(unit),
                )
                db.add(glucose_row)

        elif result.get("type") == "food":
            meal = result.get("meal") or {}
            name = meal.get("meal_name") or "Unidentified Meal"
            calories = meal.get("calories")
            carbs_g = meal.get("carbs_g")
            recommendation_level = result.get("recommendation_level")
            recommendation = result.get("recommendation") or ""

            # Attempt to link to latest glucose reading for this user
            latest_glucose_id = None
            if user_id:
                from sqlalchemy import select
                from app.models import GlucoseReading as GRModel

                stmt = (
                    select(GRModel)
                    .where(GRModel.user_id == user_id)
                    .order_by(GRModel.taken_at.desc())
                    .limit(1)
                )
                res = await db.execute(stmt)
                latest = res.scalar_one_or_none()
                if latest is not None:
                    latest_glucose_id = latest.id

            food_event = FoodEvent(
                user_id=user_id,
                chat_session_id=chat_id,
                message_id=assistant_message.id,
                image_path=image_path,
                meal_name=name,
                calories=calories,
                carbs_g=carbs_g,
                recommendation_level=recommendation_level,
                glucose_reading_id=latest_glucose_id,
            )
            db.add(food_event)

        await db.commit()

        return {
            "success": True,
            "chat_id": chat_id,
            "image_path": image_path,
            **result
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

