from enum import Enum, auto

class Command(Enum):
    SEGMENT = auto()
    RETRAIN = auto()
    FINE_TUNE = auto()
    STOP_TRAINING = auto()
    EXPORT = auto()
    LOAD_MODEL = auto()
    TEST_MODEL = auto()
    SEGMENT_FOLDER = auto()
    LOAD_IMAGE = auto()
