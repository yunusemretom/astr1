"""ASTRO V1 — Phase 2 Acceptance & Verification Test Suite.

Verifies:
  1. RelationshipManager dynamic trust and familiarity evolution over conversation turns
  2. Automatic Episodic Memory Consolidation into persistent SQLite
  3. ActionManager physical head gesture execution (nod, shake, tilt, scan, center)
  4. AstroRealtimeNode turn buffering and session end consolidation wiring
"""

import json
import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch

# Ensure test import paths
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [
    os.path.join(pkg_root, "astro_ai"),
    os.path.join(pkg_root, "astro_ai", "astro_ai"),
    os.path.join(pkg_root, "astro_audio"),
    os.path.join(pkg_root, "astro_audio", "astro_audio"),
    os.path.join(pkg_root, "astro_vision"),
    os.path.join(pkg_root, "astro_vision", "astro_vision"),
    os.path.join(pkg_root, "astro_base"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_ai.contracts.intent_emotion_types import RelationshipRole
from astro_ai.brain.relationship_manager import RelationshipManager
from astro_ai.memory_v2.sqlite_storage import SQLiteMemoryStorage
from astro_ai.memory_v2.relationship_memory import RelationshipMemory
from astro_ai.memory_v2.semantic_memory import SemanticMemory
from astro_ai.memory_v2.episodic_memory import EpisodicMemoryV2
from astro_ai.memory_v2.consolidation_engine import ConsolidationEngine
from astro_ai.action_manager import ActionManager


class TestPhase2RelationshipEvolution(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteMemoryStorage(":memory:")
        self.rel_mem = RelationshipMemory(self.storage)
        self.manager = RelationshipManager(self.rel_mem)

    def test_01_familiarity_and_trust_evolution(self):
        """Verify familiarity and trust evolve with conversational turns and valence."""
        # Initial assessment for a new user
        prof0 = self.manager.assess_relationship("Ahmet")
        self.assertEqual(prof0["role"], RelationshipRole.NEW_USER)
        self.assertAlmostEqual(prof0["familiarity"], 0.10, places=2)
        self.assertAlmostEqual(prof0["trust"], 0.50, places=2)

        # Turn 1: positive interaction
        prof1 = self.manager.record_turn_interaction("Ahmet", valence=0.5)
        self.assertGreater(prof1["familiarity"], prof0["familiarity"])
        self.assertGreater(prof1["trust"], prof0["trust"])
        self.assertEqual(prof1["interaction_count"], 2)

        # Multiple turns: should advance role to REGULAR_GUEST (count >= 3)
        self.manager.record_turn_interaction("Ahmet", valence=0.4)
        prof3 = self.manager.record_turn_interaction("Ahmet", valence=0.6)
        self.assertGreaterEqual(prof3["interaction_count"], 3)
        self.assertEqual(prof3["role"], RelationshipRole.REGULAR_GUEST)

        # Baran is creator: invariant highest trust and familiarity
        prof_baran = self.manager.record_turn_interaction("Baran", valence=0.1)
        self.assertEqual(prof_baran["role"], RelationshipRole.CREATOR)
        self.assertEqual(prof_baran["familiarity"], 1.0)
        self.assertEqual(prof_baran["trust"], 1.0)


class TestPhase2MemoryConsolidation(unittest.TestCase):
    def setUp(self):
        self.storage = SQLiteMemoryStorage(":memory:")
        self.semantic = SemanticMemory(self.storage)
        self.rel_mem = RelationshipMemory(self.storage)
        self.episodic = EpisodicMemoryV2(self.storage)
        self.engine = ConsolidationEngine(self.semantic, self.rel_mem, self.episodic)

    def test_02_session_consolidation_and_fact_extraction(self):
        """Verify session turns produce distilled semantic facts and episodic record."""
        turns = [
            {"role": "user", "content": "Benim adım Zeynep, robotik mühendisiyim."},
            {"role": "assistant", "content": "Memnun oldum Zeynep, ne güzel bir meslek!"},
            {"role": "user", "content": "Python programlama dilini çok seviyorum."},
            {"role": "assistant", "content": "Python robotik için harika bir seçim."},
        ]
        result = self.engine.consolidate_session(
            person_name="Zeynep",
            dialogue_turns=turns,
            inferred_topics=["yazılım", "robotik"],
            emotional_arc="happy",
        )
        self.assertTrue(result["archived"])
        self.assertGreaterEqual(result["extracted_facts"], 1)

        # Check semantic facts saved
        facts = self.semantic.query_active_facts_for_subject("Zeynep")
        self.assertGreaterEqual(len(facts), 1)

        # Check relationship profile evolved
        prof = self.rel_mem.get_or_create_profile("Zeynep")
        self.assertGreaterEqual(prof["interaction_count"], 2)


class TestPhase2ActionManagerGestures(unittest.TestCase):
    def setUp(self):
        self.mock_node = MagicMock()
        self.mock_node._arduino_heartbeat_healthy = True
        self.mock_node._last_heartbeat_ack_time = time.monotonic()
        self.mock_node.pub_head_cmd = MagicMock()
        self.action_manager = ActionManager(logger=MagicMock(), node=self.mock_node)

    def test_03_execute_gesture_nod_and_aliases(self):
        """Verify gesture execution publishes HeadCmd angles and returns verified ActionResult."""
        # Update joint states to simulate active encoders
        self.action_manager.update_joint_states(["head_yaw_joint"], [0.0], [0.0])

        res = self.action_manager.execute_gesture("nod", duration_ms=200)
        self.assertTrue(res.success)
        self.assertEqual(res.action, "execute_gesture")
        self.assertEqual(res.actual_direction, "nod")
        self.assertTrue(res.hardware_ack)
        self.assertTrue(res.verified)

        # Test alias resolution: 'yes' -> 'nod'
        res_alias = self.action_manager.execute_gesture("yes", duration_ms=200, action_id="test_yes_alias")
        self.assertTrue(res_alias.success)
        self.assertEqual(res_alias.actual_direction, "nod")

    def test_04_execute_gesture_shake_tilt_scan(self):
        """Verify shake, tilt, and scan gestures."""
        for g_name in ["shake", "tilt", "scan", "center"]:
            res = self.action_manager.execute_gesture(g_name, duration_ms=200, action_id=f"test_{g_name}")
            self.assertTrue(res.success)
            self.assertEqual(res.actual_direction, g_name)

    def test_05_invalid_gesture_fails_gracefully(self):
        """Verify unknown gesture returns INVALID_GESTURE error."""
        res = self.action_manager.execute_gesture("do_a_backflip")
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "INVALID_GESTURE")


class TestPhase2RealtimeNodeSessionWiring(unittest.TestCase):
    @patch.dict(os.environ, {"ASTRO_TEST_MODE": "1", "OPENAI_API_KEY": ""})
    def test_06_realtime_node_buffers_turns_and_grounds_gestures(self):
        """Verify AstroRealtimeNode buffers dialogue turns and maps speech to gestures."""
        from astro_ai.astro_realtime_node import AstroRealtimeNode

        node = AstroRealtimeNode()
        node.action_manager = MagicMock()

        # Test speech gesture grounding
        node._ground_speech_gesture("Evet, kesinlikle haklısın!")
        node.action_manager.execute_gesture.assert_called_with("nod")

        node._ground_speech_gesture("Acaba bu nasıl çalışıyor?")
        node.action_manager.execute_gesture.assert_called_with("tilt")

        node._ground_speech_gesture("Hayır, öyle bir şey söylemedim.")
        node.action_manager.execute_gesture.assert_called_with("shake")

        # Test turn buffering
        node._session_turns_buffer.append({"role": "user", "content": "Merhaba Astro", "timestamp": time.time()})
        node._session_turns_buffer.append({"role": "assistant", "content": "Merhaba!", "timestamp": time.time()})
        self.assertEqual(len(node._session_turns_buffer), 2)

        # Trigger session end consolidation callback
        node._active_person_name = "Baran"
        with patch.object(node.social_brain.consolidation_engine, "consolidate_session") as mock_consolidate:
            node._on_conversation_session_ended()
            # Buffer should be cleared
            self.assertEqual(len(node._session_turns_buffer), 0)


if __name__ == "__main__":
    unittest.main()
