"""ASTRO V1 — Proactive Memory Consolidation and Distillation Engine.

Extracts durable semantic facts, preference updates, and relationship milestones
from raw session conversations without polluting persistent memory with casual small-talk.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from astro_ai.contracts.intent_emotion_types import MemorySourceType
from astro_ai.contracts.memory_models import MemoryType
from astro_ai.memory_v2.episodic_memory import EpisodicMemoryV2
from astro_ai.memory_v2.relationship_memory import RelationshipMemory
from astro_ai.memory_v2.semantic_memory import SemanticMemory


class ConsolidationEngine:
    """Extracts distilled memories and updates relationship profiles at session end."""

    def __init__(
        self,
        semantic_mem: SemanticMemory,
        relationship_mem: RelationshipMemory,
        episodic_mem: EpisodicMemoryV2,
    ):
        self.semantic = semantic_mem
        self.relationship = relationship_mem
        self.episodic = episodic_mem

    def consolidate_session(
        self,
        person_name: str,
        dialogue_turns: List[Dict[str, Any]],
        inferred_topics: Optional[List[str]] = None,
        emotional_arc: str = "neutral",
    ) -> Dict[str, Any]:
        """Processes completed session into durable memory records and archives the session."""
        if not dialogue_turns:
            return {"archived": False, "extracted_facts": 0}

        topics = inferred_topics or []
        person_norm = person_name.strip() if person_name else "Misafir"

        # 1. Heuristic / Pattern-based Semantic Fact Extraction (zero token cost)
        extracted_facts_count = 0
        for turn in dialogue_turns:
            if turn.get("role") != "user":
                continue
            text = str(turn.get("content", ""))
            extracted = self._extract_facts_from_user_text(person_norm, text)
            for subj, pred, val in extracted:
                rec = self.semantic.store_fact(
                    subject=subj,
                    predicate=pred,
                    value=val,
                    memory_type=MemoryType.FACT,
                    source_type=MemorySourceType.EXPLICIT_USER_STATEMENT,
                    evidence=text,
                    created_by_person=person_norm,
                )
                if rec is not None:
                    extracted_facts_count += 1

        # 2. Generate Concise Summary
        user_utterances = [
            t.get("content", "") for t in dialogue_turns if t.get("role") == "user"
        ]
        summary = f"{len(dialogue_turns)} turluk sohbet yapıldı."
        if user_utterances:
            first_topic = user_utterances[0][:50]
            summary = f"Kullanıcı '{first_topic}...' hakkında konuştu. {len(dialogue_turns)} tur sürdü."

        # 3. Archive Episodic Session
        self.episodic.archive_session(
            person_id=f"rel_{person_norm.lower()}",
            summary=summary,
            topics=topics,
            emotional_arc=emotional_arc,
        )

        # 4. Increment Relationship Interactivity
        if person_norm.lower() != "misafir":
            self.relationship.increment_interaction(person_norm, topics=topics)

        return {
            "archived": True,
            "extracted_facts": extracted_facts_count,
            "summary": summary,
        }

    def _extract_facts_from_user_text(
        self, person_name: str, text: str
    ) -> List[Tuple[str, str, str]]:
        """Lightweight regex-based extraction of explicit preferences and assertions."""
        results = []
        t_low = text.lower()

        # Turkish preference patterns
        # 1. Favorite Team
        m_team = re.search(r"(?:tuttuğum takım|tuttugum takim|en sevdiğim takım|en sevdigim takim)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ\s]+)", t_low)
        if m_team:
            results.append((person_name, "favorite_team", m_team.group(1).strip()))

        # 2. Favorite Food
        m_food = re.search(r"(?:en sevdiğim yemek|en sevdigim yemek|favori yemeğim)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ\s]+)", t_low)
        if m_food:
            results.append((person_name, "favorite_food", m_food.group(1).strip()))

        # 3. Hometown / Living City
        m_city = re.search(r"(?:memleketim|yaşadığım şehir|yasadigim sehir)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ\s]+)", t_low)
        if m_city:
            results.append((person_name, "hometown", m_city.group(1).strip()))

        # 4. Profession / Role
        m_prof = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ]+)\s+(?:mühendisiyim|muhendisiyim|doktoruyum|öğretmeniyim|ogretmeniyim|öğrencisiyim|ogrencisiyim)", t_low)
        if m_prof:
            results.append((person_name, "profession", f"{m_prof.group(1).strip()} mühendisi" if "mühendis" in m_prof.group(0) else m_prof.group(1).strip()))
        else:
            m_meslek = re.search(r"(?:mesleğim|meslegim)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ\s]+)", t_low)
            if m_meslek:
                results.append((person_name, "profession", m_meslek.group(1).strip()))

        # 5. Likes / Interest
        m_like = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]+?)\s+(?:seviyorum|beğeniyorum|begeniyorum|tutkunuyum)", t_low)
        if m_like:
            candidate = m_like.group(1).strip()
            candidate = re.sub(r"\b(?:çok|cok|fazla|gerçekten|gercekten)\b", "", candidate).strip()
            if candidate and len(candidate) > 2:
                results.append((person_name, "likes", candidate))

        return results
