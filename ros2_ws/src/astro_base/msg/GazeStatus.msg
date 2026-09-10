std_msgs/Header header

# FSM State Enum:
# 0=IDLE, 1=SEARCHING, 2=AUDIO_ACQUIRE, 3=ORIENTING, 4=VISUAL_ACQUIRE,
# 5=TRACKING, 6=HOLD, 7=TARGET_LOST, 8=RETURNING
uint8 state

# Priority Source Enum:
# 0=IDLE, 1=VISUAL_PERSON, 2=ACTIVE_SPEAKER, 3=DIALOGUE, 4=GESTURE, 5=SAFETY
uint8 priority

# Three distinct spatial angles
float64 desired_yaw_deg
float64 planned_yaw_deg
float64 actual_yaw_deg

# Target metrics
float64 target_confidence
bool target_valid
bool motion_active
bool at_target
string active_target_id
