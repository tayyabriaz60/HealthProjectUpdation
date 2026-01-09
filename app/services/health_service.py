"""
Health Data Service.
Handles fetching and formatting patient health context from the database.
"""
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import GlucoseReading, FoodEvent

class HealthService:
    @staticmethod
    async def get_patient_context_string(db: AsyncSession, user_id: str) -> str:
        """
        Fetches latest readings, averages, and recent meals for a user.
        Returns a formatted string for injection into the AI system prompt/context.
        """
        if not user_id:
            return ""

        context_parts = []

        # 1. Fetch Latest Glucose Reading
        stmt = (
            select(GlucoseReading)
            .where(GlucoseReading.user_id == user_id)
            .order_by(GlucoseReading.taken_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        latest_reading = result.scalar_one_or_none()
        
        unit = "mg/dL" # Default
        if latest_reading:
            unit = latest_reading.unit
            context_parts.append(
                f"Latest glucose: {latest_reading.value} {unit} (at {latest_reading.taken_at.strftime('%Y-%m-%d %H:%M')})."
            )

        # 2. Fetch Average Glucose (Last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        avg_stmt = (
            select(func.avg(GlucoseReading.value))
            .where(
                GlucoseReading.user_id == user_id,
                GlucoseReading.taken_at >= thirty_days_ago
            )
        )
        avg_result = await db.execute(avg_stmt)
        avg_value = avg_result.scalar()
        
        if avg_value:
            context_parts.append(f"30-day average: {avg_value:.1f} {unit}.")

        # 3. Fetch Recent Food History (Last 3 meals)
        food_stmt = (
            select(FoodEvent)
            .where(FoodEvent.user_id == user_id)
            .order_by(FoodEvent.created_at.desc())
            .limit(3)
        )
        food_result = await db.execute(food_stmt)
        recent_meals = food_result.scalars().all()
        
        if recent_meals:
            meal_list = []
            for meal in recent_meals:
                m_str = f"- {meal.meal_name}"
                if meal.calories: m_str += f" ({meal.calories} kcal)"
                meal_list.append(m_str)
            
            context_parts.append("Recent meals:\n" + "\n".join(meal_list))

        if not context_parts:
            return ""

        # Construct final context block
        joined_parts = "\n".join(context_parts)
        return (
            "\n\n[SYSTEM CONTEXT: USER HEALTH DATA]\n"
            f"{joined_parts}\n"
            "[END CONTEXT]"
        )
