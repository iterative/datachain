from enum import Enum


class JobQueryStepStatus(int, Enum):
    RUNNING = 10
    FINISHED = 20
    ERROR = 30
