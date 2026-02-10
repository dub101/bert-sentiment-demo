import os
import inspect
import numpy as np

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

def _make_training_args(**kwargs) -> TrainingArguments:
    """
    Create TrainingArguments while tolerating minor API changes across Transformers versions.
    - Handles known renames
    - Drops unsupported kwargs (prints what it dropped)
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    # evaluation_strategy -> eval_strategy (newer versions)
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    # Drop unsupported kwargs to avoid version-specific crashes
    dropped = [k for k in list(kwargs.keys()) if k not in params]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"[train.py] Dropping unsupported TrainingArguments kwargs: {dropped}")

    return TrainingArguments(**kwargs)


def main() -> None:
    set_seed(42)
    model_ckpt = os.environ.get("MODEL_CKPT", "distilbert-base-uncased")
    output_dir = os.environ.get("OUTPUT_DIR", "models/distilbert-sst2")

    max_length = int(os.environ.get("MAX_LENGTH", "256"))
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "2000"))
    max_eval_samples = int(os.environ.get("MAX_EVAL_SAMPLES", "500"))
    per_device_train_batch_size = int(os.environ.get("TRAIN_BS", "16"))
    per_device_eval_batch_size = int(os.environ.get("EVAL_BS", "32"))
    max_steps = int(os.environ.get("MAX_STEPS", "200"))

    print(f"Loading dataset glue/sst2 ...")
    ds = load_dataset("glue", "sst2")

    print(f"Loading tokenizer/model: {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    # Tokenize + trim columns
    tokenized = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("label",)])

    train_ds = tokenized["train"].select(range(min(max_train_samples, len(tokenized["train"]))))
    eval_ds = tokenized["validation"].select(range(min(max_eval_samples, len(tokenized["validation"]))))


    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    args = _make_training_args(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=1,
        max_steps=max_steps,               # caps runtime for CPU
        warmup_ratio=0.06,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=25,
        load_best_model_at_end=False,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model + tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
