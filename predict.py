from typing import Tuple
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification


def inference_sequence_classification(
    text: str, model: BertModel, tokenizer: AutoTokenizer, class_labels: Tuple[str, ...]
) -> str:
    """
    Perform Sequene classification inference on a text using a BERT model.
    :param text: The text to perform inference on.
    :param model: The BERT model.
    :param tokenizer: The tokenizer.
    :param class_labels: The class labels.
    :return: The predicted class.
    """
    device = model.device
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    output = model(**encoding)
    predicted_class_idx = output.logits.argmax(dim=1)
    return class_labels[predicted_class_idx]


if __name__ == "__main__":
    MODEL_NAME = "cache\\model\\adapter-bert-sentiment-analysis\\"
    CLASS_LABELS = ("negative", "positive")
    query = input("Enter a sentence: ")
    _tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        truncation=True,
        padding=True,
        use_fast=True,
        cache_dir="cache\\tokenizer",
    )
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    prediction = inference_sequence_classification(
        query, _model, _tokenizer, CLASS_LABELS
    )
    print(prediction)
