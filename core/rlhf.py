"""
core/rlhf.py
------------
RLHF self-learning loop:
  - RewardModel: converts user feedback into a -1..+1 reward.
  - FeedbackCollector: persists to feedback memory and updates procedural memory.
  - StrategyAdvisor: reads procedural memory at query time to bias routing/complexity decisions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

log = structlog.get_logger()


class RewardModel:
    @staticmethod
    def calculate_reward(
        thumbs_up: bool | None = None,
        thumbs_down: bool | None = None,
        rating: int | None = None,
        correction: str | None = None,
        dwell_time: float | None = None,
    ) -> float:
        score = 0.0
        weights = 0.0

        if thumbs_up is not None:
            score += 1.0 if thumbs_up else 0.0
            weights += 1.0
        if thumbs_down is not None:
            score += -1.0 if thumbs_down else 0.0
            weights += 1.0

        if rating is not None:
            # Map 1-5 to -1..1
            r_score = (rating - 3) / 2.0
            score += r_score
            weights += 1.0

        if correction:
            # A correction implies the answer was wrong
            score += -0.5
            weights += 0.5

        if dwell_time is not None:
            # Dwell time > 30s is positive, < 5s is negative
            if dwell_time > 30:
                score += 0.3
                weights += 0.3
            elif dwell_time < 5:
                score += -0.3
                weights += 0.3

        if weights == 0:
            return 0.0
        return max(-1.0, min(1.0, score / weights))


class FeedbackCollector:
    @staticmethod
    async def collect_feedback(
        query: str,
        strategy: str,
        reward: float,
        feedback_details: dict[str, Any],
    ) -> None:
        from rag.vectorstore import upsert_document

        # Save to feedback_memory
        fb_text = f"Query: {query}\nStrategy: {strategy}\nReward: {reward}\nDetails: {json.dumps(feedback_details)}"
        fb_hash = hashlib.sha256(fb_text.encode("utf-8")).hexdigest()
        fb_id = f"{fb_hash[0:8]}-{fb_hash[8:12]}-{fb_hash[12:16]}-{fb_hash[16:20]}-{fb_hash[20:32]}"

        await upsert_document(
            collection_name="feedback_memory",
            text=fb_text,
            metadata={"query": query, "strategy": strategy, "reward": reward, **feedback_details},
            id=fb_id,
        )
        log.info("feedback_saved", query=query[:60], reward=reward)

        # If reward is negative, we want to recommend a different strategy (complexity upgrade)
        # If positive, we recommend the current strategy
        recommended_strategy = strategy
        if reward < -0.3:
            if strategy == "SIMPLE":
                recommended_strategy = "MEDIUM"
            elif strategy == "MEDIUM":
                recommended_strategy = "COMPLEX"
            elif strategy in ("COMPLEX", "web_only", "local_only", "hybrid"):
                recommended_strategy = "RESEARCH"

        proc_text = query
        proc_hash = hashlib.sha256(f"proc:{query}".encode("utf-8")).hexdigest()
        proc_id = f"{proc_hash[0:8]}-{proc_hash[8:12]}-{proc_hash[12:16]}-{proc_hash[16:20]}-{proc_hash[20:32]}"

        await upsert_document(
            collection_name="procedural_memory",
            text=proc_text,
            metadata={
                "query": query,
                "strategy": recommended_strategy,
                "reward": reward,
            },
            id=proc_id,
        )
        log.info("procedural_memory_updated", query=query[:60], strategy=recommended_strategy)


class StrategyAdvisor:
    @staticmethod
    async def advise(query: str) -> str | None:
        """
        Reads procedural memory at query time to bias future routing/complexity decisions.
        """
        try:
            from rag.vectorstore import similarity_search

            results = await similarity_search("procedural_memory", query, k=1)
            if results:
                doc, score = results[0]
                # If similarity score is high, recommend the strategy
                if score >= 0.75:
                    recommended = doc.metadata.get("strategy")
                    log.info("strategy_advisor_match", query=query[:60], recommended=recommended, score=score)
                    return recommended
        except Exception as e:
            log.warning("strategy_advisor_error", error=str(e))
        return None


async def get_strategy_advisor_recommendation(query: str) -> str | None:
    return await StrategyAdvisor.advise(query)
