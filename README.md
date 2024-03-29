# AdapterBERT

AdapterBERT is a project that utilizes the BERT model with adapter layers for fine-tuning on specific downstream tasks. This project is an implementation of the paper: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751), Houlsby [Google], ICML 2019.

## Setup

To set up the project, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/nikhil-chigali/AdapterBERT.git
    ```

2. Navigate to the project directory:

    ```shell
    cd AdapterBERT
    ```

3. Install Poetry:

    ```shell
    pip install poetry
    ```

4. Set up the Poetry environment:

    ```shell
    poetry install --no-root
    ```

## Training

Update the training hyperparameters and settings as needed in the `constants.py` file.

To train the model, run the `train.py` script. Here's an example command:

```shell
python train.py
```

## Prediction

Before running any inferences, update the path to the model checkpoint `CKPT_PATH` in `constants.py`

To make predictions on custom input prompt using the trained model, run the `predict.py` script:

```shell
python predict.py
```

## Acknowledgements

I would like to acknowledge the following repositories and papers used as reference for this project:

- [bert_adapter](https://github.com/strawberrypie/bert_adapter.git) repo by [@strawberrypie](https://github.com/strawberrypie)
- [transformers](https://github.com/huggingface/transformers.git) repo by [@huggingface](https://github.com/huggingface)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751), Houlsby [Google], ICML 2019.
