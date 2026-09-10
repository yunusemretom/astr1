std_msgs/Header header

# Actual physical state from encoder feedback
float64 position_deg
float64 velocity_deg_s

# Commanded target state
float64 target_position_deg

# Motion and settling indicators
bool moving
bool at_target

# Hardware & safety status
bool enabled
bool watchdog_healthy
bool encoder_valid
uint8 fault_code
