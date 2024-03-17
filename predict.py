import os
from src.model import AdapterBertForSequenceClassification
from src.data import SequenceClassificationDataModule
from constants import (
    MODEL_NAME,
    DATASET_NAME,
    BATCH_SIZE,
    NUM_WORKERS,
    CKPT_PATH,
)


def inference_sequence_classification(
    text: str,
) -> str:
    model = AdapterBertForSequenceClassification.load_from_checkpoint(
        checkpoint_path=CKPT_PATH
    )
    device = model.device
    sst2 = SequenceClassificationDataModule(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    sst2.setup()
    encoding = sst2.tokenizer(
        text,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    output = model.model(**encoding)
    predicted_class_idx = output.logits.argmax(dim=1)
    return sst2.classes[predicted_class_idx]


if __name__ == "__main__":
    print("Ensure that the model checkpoint is present at", CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {CKPT_PATH}")

    query = input("Enter a sentence: ")

    prediction = inference_sequence_classification(query)
    print("Prediction:", prediction)
