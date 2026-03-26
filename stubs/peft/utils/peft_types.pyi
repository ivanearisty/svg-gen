import enum

class TaskType(enum.StrEnum):
    CAUSAL_LM = "CAUSAL_LM"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    SEQ_CLS = "SEQ_CLS"
    TOKEN_CLS = "TOKEN_CLS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
