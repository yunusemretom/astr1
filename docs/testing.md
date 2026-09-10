# ASTRO Gaze & Head Control Testing Suite

## 1. Automated Test Architecture
The test suite consists of 89 automated tests covering unit math, perception, fusion, FSM arbitration, kinematics, serial bridging, and hardware feedback synchronization.

### Key Regressions & Scenarios
- **`test_fsm_does_not_hold_before_actual_head_reaches_target`**: Verifies that when an audio cue triggers `ORIENTING` at $-35^\circ$ and then disappears, the FSM strictly prevents transitioning to `HOLD` or `VISUAL_ACQUIRE` while `actual_head_yaw_deg` is at $-9.97^\circ$ or $-25^\circ$. It only enters `VISUAL_ACQUIRE` once the encoder confirms arrival at $-35.0^\circ$ and settles for 3 consecutive cycles.
- **`test_all_motion_phases_with_closed_loop_feedback`**: Validates $0 \to -35^\circ$, $0 \to +35^\circ$, $0 \to 60^\circ$, $60 \to -60^\circ$, and $-60 \to 60^\circ$.
- **`test_doa_wrap_and_inversion`**: Validates $0^\circ \dots 360^\circ$ ReSpeaker clockwise mapping to $[-180^\circ \dots +180^\circ]$ REP-103 coordinate frame.

## 2. Running Tests
```bash
pytest ros2_ws/src/astro_base/test/
```
Output: `89 passed in 0.68s`
