from typing import Tuple, Optional
import numpy as np
import click
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from src.data import load_data, get_tokens_for_seq_classification
from src.utils import (
    load_sequence_classification_model,
    freeze_all_bert_layers,
    unfreeze_bert_adapters,
    AdapterConfig,
    add_adapters_to_bert_layers,
)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
    """
    Compute metrics for evaluation.
    :param eval_pred: The evaluation predictions.
    :return: The metrics.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "eval_accuracy": accuracy_score(labels, predictions),
        "eval_f1": f1_score(labels, predictions, average="macro"),
    }


@click.command()
@click.option("--n_epochs", default=3, help="Number of epochs.")
@click.option("--batch_size", default=8, help="Batch size.")
@click.option("--lr", default=0.00002, help="Learning rate.")
@click.option("--wt_decay", default=0.01, help="Weight decay.")
@click.option(
    "--adapter_init_range", default=0.01, help="Adapter initialization range."
)
@click.option("--adapter_dim", default=64, help="Adapter dimension.")
def train(
    n_epochs: Optional[int] = 3,
    batch_size: Optional[int] = 8,
    lr: Optional[float] = 2e-5,
    wt_decay: Optional[float] = 0.01,
    adapter_init_range: Optional[float] = 1e-2,
    adapter_dim: Optional[int] = 64,
):
    model_name = "bert-base-uncased"
    metric_name = "eval_f1"

    # Load data and tokenizer
    sst2 = load_data(("glue", "sst2"), "cache\\data\\")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation=True,
        padding=True,
        use_fast=True,
        cache_dir="cache\\tokenizer",
    )
    sst2_tokens = get_tokens_for_seq_classification(tokenizer, sst2)

    # Load model
    bert_model = load_sequence_classification_model(model_name, 2)

    # Add adapters to BERT layers
    adapter_cfg = AdapterConfig(
        hidden_dim=bert_model.config.hidden_size,
        adapter_dim=adapter_dim,
        adapter_act=bert_model.config.hidden_act,
        adapter_initializer_range=adapter_init_range,
    )

    bert_model.bert = add_adapters_to_bert_layers(bert_model.bert, adapter_cfg)
    bert_model.bert = freeze_all_bert_layers(bert_model.bert)
    bert_model.bert = unfreeze_bert_adapters(bert_model.bert)

    # Train model
    training_args = TrainingArguments(
        learning_rate=lr,
        weight_decay=wt_decay,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model=metric_name,
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=500,
        output_dir="cache\\checkpoints\\adapter_bert_sst2",
        logging_dir="cache\\logs\\adapter_bert_sst2",
    )

    trainer = Trainer(
        model=bert_model,
        train_dataset=sst2_tokens["train"],
        eval_dataset=sst2_tokens["validation"],
        args=training_args,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate model
    trainer.evaluate()

    # Save model
    trainer.save_model("cache\\model\\adapter-bert-sentiment-analysis")


if __name__ == "__main__":
    train()
