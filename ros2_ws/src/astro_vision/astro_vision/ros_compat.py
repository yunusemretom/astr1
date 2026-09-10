"""Headless stand-ins for the rclpy pieces the vision nodes use.

These activate only when rclpy cannot be imported, which is how the nodes' logic —
capture rates, cadence, parameter plumbing — gets exercised without a ROS graph.
`declare_parameter` remembers the default it was given, so a headless node reports
the same thresholds it would run with on the robot instead of empty stubs.
"""


class MockRclpy:
    @staticmethod
    def ok():
        return True

    @staticmethod
    def init(*args, **kwargs):
        pass

    @staticmethod
    def shutdown():
        pass

    @staticmethod
    def spin(*args, **kwargs):
        pass


class MockParam:
    def __init__(self, value):
        self.value = value

    def get_parameter_value(self):
        class _Val:
            def __init__(self, v):
                self.string_value = str(v) if v is not None else ""
                self.double_value = float(v) if isinstance(v, (int, float)) else 0.0
                self.integer_value = int(v) if isinstance(v, (int, float)) else 0
                self.bool_value = bool(v)

        return _Val(self.value)


class MockPublisher:
    """Captures published messages so headless tests can assert on node output."""

    def __init__(self, topic):
        self.topic = topic
        self.last_msg = None
        self.count = 0

    def publish(self, msg):
        self.last_msg = msg
        self.count += 1


class MockClock:
    def now(self):
        class _Time:
            nanoseconds = 0

            def to_msg(self):
                return None

        return _Time()


class MockMsg:
    """Assignable stand-in for std_msgs / sensor_msgs types."""

    def __init__(self, data=None, **kwargs):
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockNode:
    def __init__(self, *args, **kwargs):
        self._declared_parameters = {}

    def create_publisher(self, msg_type, topic, *args, **kwargs):
        return MockPublisher(topic)

    def create_subscription(self, *args, **kwargs):
        return None

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return MockClock()

    def get_logger(self):
        import logging

        return logging.getLogger(self.__class__.__name__)

    def declare_parameter(self, name, value=None, *args, **kwargs):
        self._declared_parameters[name] = value
        return MockParam(value)

    def get_parameter(self, name):
        return MockParam(self._declared_parameters.get(name))

    def destroy_node(self):
        pass
