import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from src.model import AdapterBertForSequenceClassification
from src.data import SequenceClassificationDataModule
from constants import (
    SEED,
    MODEL_NAME,
    ADAPTER_DIM,
    ADAPTER_INIT_RANGE,
    DATASET_NAME,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    DEVICES,
)


def train():
    seed_everything(SEED)
    model = AdapterBertForSequenceClassification(
        MODEL_NAME, ADAPTER_DIM, ADAPTER_INIT_RANGE, LR, WEIGHT_DECAY
    )
    sst2 = SequenceClassificationDataModule(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    sst2.prepare_data()
    sst2.setup()
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="val/acc",
            mode="max",
        )
    ]
    logger = pl.loggers.CSVLogger(
        "cache/logs/", name="adapter-bert-sequence-classification"
    )

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        devices=DEVICES,
        accelerator="gpu",
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=20,
        val_check_interval=0.5,
    )

    start = time.time()
    trainer.fit(model, datamodule=sst2)
    end = time.time()
    print(f"\nTraining took {(end-start)/60:.2f} minutes.\n")

    print("\nBest Model Path: ", trainer.checkpoint_callback.best_model_path)
    print("Loading best model...\n")
    model = AdapterBertForSequenceClassification.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path
    )

    train_results = trainer.validate(
        model, dataloaders=sst2.train_dataloader(), verbose=False
    )[0]
    val_results = trainer.validate(
        model, dataloaders=sst2.val_dataloader(), verbose=False
    )[0]

    print("\nTrain results:: ")
    print(f"Train Loss: {train_results['val/loss']}")
    print(f"Train Accuracy: {train_results['val/acc']}")
    print(f"Train F1: {train_results['val/f1']}")

    print("\nValidation results:: ")
    print(f"Validation Loss: {val_results['val/loss']}")
    print(f"Validation Accuracy: {val_results['val/acc']}")
    print(f"Validation F1: {val_results['val/f1']}")


if __name__ == "__main__":
    train()
