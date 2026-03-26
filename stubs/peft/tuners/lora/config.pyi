from typing import Literal

from peft.utils.peft_types import TaskType

class LoraConfig:
    def __init__(
        self,
        r: int = ...,
        lora_alpha: int = ...,
        lora_dropout: float = ...,
        target_modules: list[str] | None = ...,
        bias: Literal["none", "all", "lora_only"] = ...,
        task_type: TaskType | str | None = ...,
        **kwargs: object,
    ) -> None: ...
