import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from transformers import BertModel
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from .adapter import AdapterConfig, BertAdaptedOutput, BertAdaptedSelfOutput


class AdapterBertForSequenceClassification(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        adapter_dim: int,
        adapter_init_range: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir="cache/model"
        )
        self.adapter_config = AdapterConfig(
            hidden_dim=self.model.config.hidden_size,
            adapter_dim=adapter_dim,
            adapter_act=self.model.config.hidden_act,
            adapter_initializer_range=adapter_init_range,
        )
        self.model.bert = _add_adapters_to_bert_layers(
            self.model.bert, self.adapter_config
        )
        self.model.bert = _freeze_bert_params(self.model.bert)
        self.model.bert = _unfreeze_adapters(self.model.bert)

        self.acc_metric = {
            "train": BinaryAccuracy().cuda(),
            "val": BinaryAccuracy().cuda(),
            "test": BinaryAccuracy().cuda(),
        }
        self.f1_metric = {
            "train": BinaryF1Score().cuda(),
            "val": BinaryF1Score().cuda(),
            "test": BinaryF1Score().cuda(),
        }

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        log_dict = {
            "train/loss": output["loss"],
            "train/acc": self.acc_metric["train"](
                output["logits"][:, 1], batch["labels"]
            ),
            "train/f1": self.f1_metric["train"](
                output["logits"][:, 1], batch["labels"]
            ),
        }

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": output["loss"]}

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        acc = self.acc_metric["val"](output["logits"][:, 1], batch["labels"])
        f1 = self.f1_metric["val"](output["logits"][:, 1], batch["labels"])
        log_dict = {"val/loss": output["loss"], "val/acc": acc, "val/f1": f1}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return log_dict

    def test_step(self, batch, batch_idx):
        output = self(**batch)
        acc = self.acc_metric["test"](output["logits"][:, 1], batch["labels"])
        f1 = self.f1_metric["test"](output["logits"][:, 1], batch["labels"])
        log_dict = {"test/acc": acc, "test/f1": f1}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return log_dict

    def configure_optimizers(self):
        return optim.Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def _add_adapters_to_bert_layers(
    model: BertModel, adapter_cfg: AdapterConfig
) -> BertModel:
    for layer in model.encoder.layer:
        layer.attention.output = BertAdaptedSelfOutput(
            layer.attention.output, adapter_cfg
        )
        layer.output = BertAdaptedOutput(layer.output, adapter_cfg)
    return model


def _freeze_bert_params(model: BertModel) -> BertModel:
    for param in model.parameters():
        param.requires_grad = False
    return model


def _unfreeze_adapters(model: BertModel) -> BertModel:
    for i, layer in enumerate(model.encoder.layer):
        assert hasattr(layer.attention.output, "adapter") and hasattr(
            layer.output, "adapter"
        ), f"Adapter not found in layer-{i}"

        for param in layer.attention.output.adapter.parameters():
            param.requires_grad = True

        for param in layer.output.adapter.parameters():
            param.requires_grad = True
    return model
