from enum import Enum

class StepMode(str, Enum):
    SINGLE = "single"
    MULTI_STATELESS = "multi_stateless"
    MULTI_STATEFUL = "multi_stateful"

