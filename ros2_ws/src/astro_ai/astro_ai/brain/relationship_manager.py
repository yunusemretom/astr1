"""ASTRO V1 — Social Relationship and Familiarity Manager."""

import json
import time
from typing import Any, Dict, List, Optional

from astro_ai.contracts.intent_emotion_types import RelationshipRole
from astro_ai.memory_v2.relationship_memory import RelationshipMemory


class RelationshipManager:
    """Evaluates interpersonal dynamics, trust, familiarity, and communication tone preferences."""

    def __init__(self, relationship_mem: RelationshipMemory):
        self.memory = relationship_mem

    def assess_relationship(self, person_name: str, formal_title: str = "") -> Dict[str, Any]:
        """Returns the current relationship assessment and conversational guidance for a person."""
        prof = self.memory.get_or_create_profile(person_name, formal_title)
        role = prof["role"]
        fam = prof["familiarity"]
        trust = prof["trust"]
        count = prof["interaction_count"]

        # Determine appropriate tone and formality
        if person_name.lower() == "baran":
            suggested_tone = "playful_and_loyal_partner"
            formality_level = "informal_best_friend"
        elif role == RelationshipRole.FRIEND or fam >= 0.70:
            suggested_tone = "warm_and_humorous"
            formality_level = "friendly_informal"
        elif role == RelationshipRole.REGULAR_GUEST:
            suggested_tone = "warm_and_respectful"
            formality_level = "semi_formal"
        else:
            suggested_tone = "polite_welcoming_and_helpful"
            formality_level = "polite_formal"

        return {
            "person_id": prof["person_id"],
            "name": prof["name"],
            "formal_title": prof["formal_title"],
            "role": role,
            "familiarity": fam,
            "trust": trust,
            "interaction_count": count,
            "suggested_tone": suggested_tone,
            "formality_level": formality_level,
            "shared_topics": prof["shared_topics"],
        }

    def record_turn_interaction(
        self,
        person_name: str,
        valence: float = 0.0,
        topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Updates trust and familiarity scores based on conversational sentiment and turn count."""
        prof = self.memory.get_or_create_profile(person_name)
        new_count = prof["interaction_count"] + 1

        # Familiarity increases asymptotically with each turn
        new_fam = min(0.98, prof["familiarity"] + 0.03) if prof["name"].lower() != "baran" else 1.0

        # Trust adjusts according to emotional valence / cooperative interaction
        # Positive sentiment (valence > 0.2) builds trust; hostile sentiment degrades it
        trust_delta = 0.02 if valence > 0.2 else (-0.03 if valence < -0.4 else 0.005)
        new_trust = max(0.1, min(1.0, prof["trust"] + trust_delta)) if prof["name"].lower() != "baran" else 1.0

        # Role progression
        role = prof["role"]
        if role == RelationshipRole.NEW_USER and new_count >= 3:
            role = RelationshipRole.REGULAR_GUEST
        if role == RelationshipRole.REGULAR_GUEST and new_count >= 8 and new_trust >= 0.60:
            role = RelationshipRole.FRIEND

        existing_topics = set(prof["shared_topics"])
        if topics:
            existing_topics.update(topics)

        now = time.time()
        self.memory.storage.execute_write(
            """
            UPDATE relationship_profiles
            SET interaction_count = ?, familiarity = ?, trust = ?, role = ?, last_seen = ?, shared_topics_json = ?
            WHERE person_id = ?
            """,
            (
                new_count,
                round(new_fam, 3),
                round(new_trust, 3),
                role.value,
                now,
                json.dumps(list(existing_topics)),
                prof["person_id"],
            ),
        )
        return self.assess_relationship(person_name)
