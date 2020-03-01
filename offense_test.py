from transformers import EnsembleForOffensiveClassification
from transformers import PretrainedConfig, BertConfig, RobertaConfig
from torchsummaryX import summary
import torch
import numpy as np
if __name__ == "__main__":
    max_length =128
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    roberta_config = RobertaConfig.from_pretrained("roberta-base")
    batch_size = 32
    # roberta_config = RobertaConfig(max_length=128)
    # ensemble = EnsembleForOffensiveClassification(bert_config, roberta_config)
    model_dir = "E:\\Emory\\Research\\offensEval2020\\models"
    ensemble = EnsembleForOffensiveClassification.from_pretrained(model_dir, bert_config=bert_config, roberta_config=roberta_config)
    input_ids = np.ones(shape=(batch_size, max_length), dtype=np.int32)
    label_ids = np.full(shape=(batch_size, ), dtype=np.int32, fill_value=0)
    ensemble.cpu()
    ensemble.float()
    summary(ensemble, torch.tensor(input_ids.astype(np.int64)), torch.tensor(label_ids.astype(np.int64)))

