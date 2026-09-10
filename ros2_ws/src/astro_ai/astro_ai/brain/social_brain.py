"""ASTRO V1 — Master Social Brain Cognitive Orchestrator.

Integrates Perception, World Model, Self Model, Memory V2, Intent, Emotion,
Attention, Relationship Evolution, Social FSM, Initiative, and Response Planning.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from astro_ai.brain.attention_manager import AttentionManager
from astro_ai.brain.emotion_engine import EmotionEngine
from astro_ai.brain.initiative_engine import InitiativeEngine
from astro_ai.brain.intent_engine import IntentEngine
from astro_ai.brain.relationship_manager import RelationshipManager
from astro_ai.brain.response_planner import ResponsePlanner
from astro_ai.brain.self_model import SelfModel
from astro_ai.brain.social_fsm import SocialFSM
from astro_ai.brain.world_model import WorldModel
from astro_ai.contracts.intent_emotion_types import (
    ConversationPhase,
    EmotionSignal,
    IntentType,
    RelationshipRole,
)
from astro_ai.contracts.person_state import UnifiedPersonState
from astro_ai.contracts.social_context import SocialContext, SocialDecision
from astro_ai.memory_v2.autobiographical_memory import AutobiographicalMemory
from astro_ai.memory_v2.consolidation_engine import ConsolidationEngine
from astro_ai.memory_v2.episodic_memory import EpisodicMemoryV2
from astro_ai.memory_v2.migration import MemoryMigrator
from astro_ai.memory_v2.relationship_memory import RelationshipMemory
from astro_ai.memory_v2.retrieval_engine import MemoryRetrievalEngine
from astro_ai.memory_v2.semantic_memory import SemanticMemory
from astro_ai.memory_v2.spatial_memory import SpatialMemory
from astro_ai.memory_v2.sqlite_storage import SQLiteMemoryStorage
from astro_ai.spatial.spatial_fusion import SpatialFusionEngine


class SocialBrain:
    """Master Cognitive Engine for ASTRO Social Robot."""

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()

        # 1. Epistemic & World Models
        self.self_model = SelfModel()
        self.world_model = WorldModel()

        # 2. Memory V2 Cognitive Architecture
        self.storage = SQLiteMemoryStorage(db_path)
        self.semantic_memory = SemanticMemory(self.storage)
        self.episodic_memory = EpisodicMemoryV2(self.storage)
        self.autobiographical_memory = AutobiographicalMemory(self.storage)
        self.spatial_memory = SpatialMemory(self.storage)
        self.relationship_memory = RelationshipMemory(self.storage)
        self.retrieval_engine = MemoryRetrievalEngine(self.storage, self.semantic_memory)
        self.consolidation_engine = ConsolidationEngine(
            self.semantic_memory, self.relationship_memory, self.episodic_memory
        )

        # Automatic Migration on Startup
        self.migrator = MemoryMigrator(
            self.storage, self.semantic_memory, self.relationship_memory, self.spatial_memory
        )
        self.migrator.migrate_if_needed()

        # 3. Spatial & Sensory Fusion
        self.spatial_fusion = SpatialFusionEngine()

        # 4. Cognitive Reasoning Components
        self.intent_engine = IntentEngine()
        self.emotion_engine = EmotionEngine()
        self.attention_manager = AttentionManager()
        self.relationship_manager = RelationshipManager(self.relationship_memory)
        self.social_fsm = SocialFSM()
        self.initiative_engine = InitiativeEngine()
        self.response_planner = ResponsePlanner()

    def process_dialogue_turn(
        self,
        user_text: str,
        person_state: Optional[UnifiedPersonState] = None,
        active_persona: str = "playful",
        acoustic_energy_rms: float = 500.0,
    ) -> Tuple[SocialContext, SocialDecision, str]:
        """Executes full cognitive reasoning loop for an incoming dialogue turn.

        Returns (SocialContext, SocialDecision, StructuredSystemPrompt).
        """
        with self._lock:
            # 1. Identity & Attention Resolution
            person = person_state or self.spatial_fusion.get_fused_primary_person()
            p_name = person.name if person else "Misafir"
            p_title = person.formal_title if person else "Misafir"
            is_looking = person.is_looking_at_robot if person else True
            distance_m = person.distance_m if person else 1.2

            # 2. Relationship Assessment
            rel_assessment = self.relationship_manager.assess_relationship(p_name, p_title)
            role = rel_assessment["role"]
            fam = rel_assessment["familiarity"]
            trust = rel_assessment["trust"]

            # 3. Intent & Emotion Classification
            intent, intent_conf = self.intent_engine.classify_intent(user_text)
            vis_emo = person.visual_emotion if person else EmotionSignal.NEUTRAL
            affect = self.emotion_engine.estimate_affect(
                visual_emotion=vis_emo,
                is_looking=is_looking,
                acoustic_energy_rms=acoustic_energy_rms,
            )

            # 4. Contextual Memory Retrieval
            relevant_mems = self.retrieval_engine.retrieve_relevant_memories(
                person_name=p_name,
                user_query=user_text,
                top_k=5,
            )

            # 5. Social FSM Transition
            phase = self.social_fsm.step(
                primary_person=person,
                is_user_speaking=True,
                is_robot_speaking=False,
                silence_duration_s=0.0,
            )

            # 6. Formulate Normalized Social Context
            context = SocialContext(
                person_id=person.person_id if person else "person_guest",
                person_name=p_name,
                formal_title=p_title,
                relationship_role=role,
                familiarity=fam,
                trust=trust,
                conversation_phase=phase,
                user_intent=intent,
                user_mood=affect["primary_mood"],
                user_valence=affect["valence"],
                user_arousal=affect["arousal"],
                engagement_level=affect["engagement"],
                is_looking_at_robot=is_looking,
                distance_m=distance_m,
                relevant_memories=relevant_mems,
                active_persona=active_persona,
            )

            # 7. Formulate Strategic Decision
            decision = self.response_planner.plan_response_strategy(context)

            # 8. Construct Modular System Prompt
            prompt = self._build_modular_prompt(context, decision, user_text)

            # Record turn in episodic memory
            self.episodic_memory.record_turn("user", user_text)

            # Evolve relationship trust and familiarity
            if p_name and p_name.lower() != "misafir":
                self.relationship_manager.record_turn_interaction(p_name, valence=affect["valence"])

            return context, decision, prompt

    def _build_modular_prompt(
        self,
        context: SocialContext,
        decision: SocialDecision,
        user_text: str,
    ) -> str:
        """Constructs modularized system prompt without dumping unparsed database blobs."""
        parts = []

        # Part 1: Epistemic Self Model
        parts.append(self.self_model.get_self_description_prompt())

        # Part 2: Social Context & Interlocutor
        parts.append(
            f"=== ETKİLEŞİM VE SOSYAL BAĞLAM ===\n"
            f"- Muhatap: {context.person_name} ({context.formal_title})\n"
            f"- İlişki Durumu: {context.relationship_role.value} (Aşinalık: {context.familiarity:.2f}, Güven: {context.trust:.2f})\n"
            f"- Kullanıcı Niyeti: {context.user_intent.value}\n"
            f"- Kullanıcı Ruh Hali: {context.user_mood} (Valence: {context.user_valence}, Arousal: {context.user_arousal})\n"
            f"- Mesafe: {context.distance_m:.2f} metre, Bakış: {'Robota Bakıyor' if context.is_looking_at_robot else 'Başka Yere Bakıyor'}"
        )

        # Part 3: Relevant Retrieved Memories
        if context.relevant_memories:
            mem_lines = [
                f"- [{m.memory_type.value}] {m.subject} -> {m.predicate}: {m.value} (Güven: {m.confidence:.2f})"
                for m in context.relevant_memories
            ]
            parts.append("=== İLGİLİ HAFIZA BİLGİLERİ ===\n" + "\n".join(mem_lines))

        # Part 4: Strategic Response Directives
        if decision.response_strategy:
            strat_lines = [f"- {s}" for s in decision.response_strategy]
            parts.append(
                f"=== YANIT STRATEJİSİ VE TALİMATLAR ===\n"
                f"- Önerilen Ton: {decision.suggested_tone}\n"
                f"- Ayrıntı Seviyesi: {decision.recommended_verbosity}\n"
                + "\n".join(strat_lines)
            )

        return "\n\n".join(parts)
