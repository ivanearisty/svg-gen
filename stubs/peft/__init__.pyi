from peft.config import PeftConfig as PeftConfig
from peft.mapping import get_peft_model as get_peft_model
from peft.peft_model import PeftModel as PeftModel
from peft.tuners.lora.config import LoraConfig as LoraConfig
from peft.utils.peft_types import TaskType as TaskType

def prepare_model_for_kbit_training(model: object, **kwargs: object) -> object: ...

__all__ = [
    "LoraConfig",
    "PeftConfig",
    "PeftModel",
    "TaskType",
    "get_peft_model",
    "prepare_model_for_kbit_training",
]
