from typing import Any

class SFTTrainer:
    def __init__(
        self,
        model: Any = ...,
        args: Any = ...,
        train_dataset: Any = ...,
        eval_dataset: Any = ...,
        processing_class: Any = ...,
        packing: bool = ...,
        dataset_text_field: str | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def train(self, **kwargs: Any) -> Any: ...
    def save_model(self, output_dir: str, **kwargs: Any) -> None: ...
