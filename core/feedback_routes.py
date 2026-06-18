"""
core/feedback_routes.py
-----------------------
FastAPI router:
  - POST /api/feedback (submit feedback → reward model)
  - GET /api/feedback/stats (aggregate counts)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from core.rlhf import FeedbackCollector, RewardModel

router = APIRouter()


class FeedbackRequest(BaseModel):
    query: str
    strategy: str
    thumbs_up: bool | None = None
    thumbs_down: bool | None = None
    rating: int | None = None
    correction: str | None = None
    dwell_time: float | None = None
    feedback_details: dict[str, Any] = {}


@router.post("/api/feedback", tags=["rlhf"])
async def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    # Calculate reward
    reward = RewardModel.calculate_reward(
        thumbs_up=req.thumbs_up,
        thumbs_down=req.thumbs_down,
        rating=req.rating,
        correction=req.correction,
        dwell_time=req.dwell_time,
    )

    # Save feedback and learn
    await FeedbackCollector.collect_feedback(
        query=req.query,
        strategy=req.strategy,
        reward=reward,
        feedback_details={
            "thumbs_up": req.thumbs_up,
            "thumbs_down": req.thumbs_down,
            "rating": req.rating,
            "correction": req.correction,
            "dwell_time": req.dwell_time,
            **req.feedback_details,
        },
    )

    return {"status": "success", "reward": reward}


@router.get("/api/feedback/stats", tags=["rlhf"])
async def get_feedback_stats() -> dict[str, Any]:
    from rag.vectorstore import get_qdrant_client

    client = get_qdrant_client()
    try:
        response = await client.scroll(
            collection_name="feedback_memory",
            limit=100,
            with_payload=True,
        )
        points = response[0]
        total = len(points)
        positive = sum(1 for p in points if p.payload and p.payload.get("reward", 0.0) > 0)
        negative = sum(1 for p in points if p.payload and p.payload.get("reward", 0.0) < 0)
        avg_reward = (
            sum(p.payload.get("reward", 0.0) for p in points if p.payload) / total
            if total > 0
            else 0.0
        )
        return {
            "total_feedback_count": total,
            "positive_count": positive,
            "negative_count": negative,
            "average_reward": round(avg_reward, 3),
        }
    except Exception as e:
        return {
            "total_feedback_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "average_reward": 0.0,
            "error": str(e),
        }
