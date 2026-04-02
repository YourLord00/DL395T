from .base_llm import BaseLLM
from .sft import test_model
from .data import Dataset
from .sft import TokenizedDataset, test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(prompt, answer, response):

    return {"question": prompt, "answer": response}


def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import get_peft_model, LoraConfig

    llm = BaseLLM()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=80,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(llm.model, lora_config)
    model.enable_input_require_grads()

    # dataloader
    train_data = Dataset("rft")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_data, format_example)

    from transformers import Trainer, TrainingArguments

    #model config
    args = TrainingArguments(
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=32,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=False,
        report_to="tensorboard",
        output_dir=output_dir,
        logging_dir=output_dir,

    )

    # train
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset)
    trainer.train()

    #save
    model.save_pretrained(output_dir)

    test_model(output_dir)



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
