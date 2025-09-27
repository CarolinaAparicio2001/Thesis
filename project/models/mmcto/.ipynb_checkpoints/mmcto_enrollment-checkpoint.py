import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class MMCTO(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        smoe_encoder: nn.Module = None,
        smiles_transformer_encoder="seyonec/ChemBERTa-zinc-base-v1",
        final_input_parts=None,
        gate_input_parts=None,
        aux_loss=False,
        aux_loss_share_fc=False,
        weighted_aux_loss=False,
        moe_method="weighted",
        pretrain=False,
        vocab_size: int = 28996,
        model_dim: int = 768,
        num_labels: int = 1,
        augment_prob=0.0,
        augment_eps=0.1,
        piror_init=0,
        multiply_disturb=False,
        contrastive_loss=False,
        inverse_consistency_loss=False,
        use_cosin_simiarity_loss=False,
    ):
        super().__init__()
        self.encoders = encoders
        self.final_input_parts = final_input_parts or [
            "table", "summarization", "smiles", "description", "criteria",
            "enrollment", "diseases", "drugs"
        ]
        self.gate_input_parts = gate_input_parts or ["drugs", "diseases"]
        self.input_parts = set(self.final_input_parts + self.gate_input_parts)
        self.aux_loss_share_fc = aux_loss_share_fc
        self.weighted_aux_loss = weighted_aux_loss
        self.moe_method = moe_method
        self.pretrain = pretrain
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.augment_prob = augment_prob
        self.augment_eps = augment_eps
        self.contrastive_loss = contrastive_loss
        self.inverse_consistency_loss = inverse_consistency_loss
        self.use_cosin_simiarity_loss = use_cosin_simiarity_loss

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_tokens = nn.Parameter(torch.empty(len(self.input_parts), model_dim))
        self.piror = nn.Parameter(torch.empty(len(self.final_input_parts)))
        self.piror_init = piror_init
        self.multiply_disturb = multiply_disturb

        for key in list(self.encoders):
            if key not in self.input_parts:
                del self.encoders[key]

        self.smoe_encoder = smoe_encoder

        self.smiles_transformer_encoder = nn.ModuleDict({
            k: nn.ModuleList([
                AutoModel.from_pretrained(smiles_transformer_encoder),
                nn.Linear(768, model_dim),
            ])
            for k in [f"smiles_transformer{p}" for p in ["", "_concat", "_summarization"]]
            if k in self.input_parts
        })

        if "criteria" in self.input_parts:
            self.encoders["criteria"] = nn.Sequential(
                nn.Linear(768 * 2, model_dim), nn.ReLU(), nn.LayerNorm(model_dim)
            )

        if aux_loss:
            self.aux_loss_fc = nn.ModuleDict({
                part: nn.Linear(model_dim, num_labels)
                for part in self.final_input_parts if part != "enrollment"
            })
        else:
            self.aux_loss_fc = {}

        self.scalar_proj = nn.ModuleDict({
            "enrollment": nn.Linear(1, model_dim)
        }) if "enrollment" in self.input_parts else None

        if not self.pretrain:
            if moe_method == "weighted":
                self.gate_fc = nn.Linear(len(self.gate_input_parts) * model_dim, len(self.final_input_parts))
            else:
                self.gate_fc = None

            if moe_method == "concat":
                self.final_fc = nn.Linear(model_dim * len(self.final_input_parts), num_labels)
            else:
                self.final_fc = nn.Linear(model_dim, num_labels)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.aux_loss = nn.BCELoss(reduction="none")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cls_tokens)
        nn.init.normal_(self.piror, self.piror_init, 0.1)

    def add_embedding(self, input_ids, attention_mask=None, embedding_index=0):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        embedding = self.embedding(input_ids)
        cls_tokens = self.cls_tokens[embedding_index][None, None, :].expand(embedding.shape[0], 1, -1)
        embedding = torch.cat([cls_tokens, embedding], dim=1)

        _, seq_len, d_model = embedding.size()
        pos_embedding = embedding.new_zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        embedding = embedding + pos_embedding.unsqueeze(0)

        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones(attention_mask.size(0), 1, device=device),
                attention_mask
            ], dim=1)
            attention_mask = ~attention_mask.bool()
            return embedding, attention_mask
        else:
            return embedding

    def to_device(self, x):
        return {k: v.to(self.embedding.weight.device) for k, v in x.items()}

    def encode(self, embedding, attention_mask, key):
        feature = self.encoders[key](embedding, src_key_padding_mask=attention_mask)
        if self.smoe_encoder is not None:
            return self.smoe_encoder(feature[:, 0])
        return {
            "logits": feature[:, 0],
            "importance_loss": feature.new_zeros([]),
            "importances": feature.new_zeros([]),
        }

    def forward(self, data):
        features = {}
        losses = {}
        metrics = {}
        hidden_states = {"input_parts": self.final_input_parts, "smoe_weights": {}}
        embedding_index = -1

        if "criteria" in self.input_parts and "criteria" in data:
            feature = self.encoders["criteria"](data["criteria"])
            if self.smoe_encoder is not None:
                feature = self.smoe_encoder(feature)
                losses["criteria_importance_loss"] = feature["importance_loss"]
                hidden_states["smoe_weights"]["criteria"] = feature["importances"]
                feature = feature["logits"]
            features["criteria"] = feature

        for key in [f"smiles_transformer_{p}" for p in ["concat", "summarization"]]:
            if key in self.input_parts and key in data:
                feature = self.smiles_transformer_encoder[key][0](**data[key])
                feature = self.smiles_transformer_encoder[key][1](feature.pooler_output)
                if self.smoe_encoder is not None:
                    feature = self.smoe_encoder(feature)
                    losses[f"{key}_importance_loss"] = feature["importance_loss"]
                    hidden_states["smoe_weights"][key] = feature["importances"]
                    feature = feature["logits"]
                features[key] = feature

        for key in ["table", "summarization", "description", "smiles", "drugs", "diseases", "enrollment"]:
            if key not in self.input_parts or key not in data:
                continue
            embedding_index += 1
            if isinstance(data[key], list):
                embed_list = []
                for item in data[key]:
                    embedding, attention_mask = self.add_embedding(**self.to_device(item), embedding_index=embedding_index)
                    encode_result = self.encode(embedding, attention_mask, key)
                    embed_list.append(encode_result["logits"].mean(dim=0, keepdim=True))
                features[key] = torch.cat(embed_list, dim=0)
            elif isinstance(data[key], torch.Tensor) and data[key].dim() == 2 and data[key].shape[1] == 1:
                sanitized = data[key].clone()
                sanitized[torch.isnan(sanitized)] = 0.0
                features[key] = self.scalar_proj[key](sanitized.to(self.embedding.weight.device))
            else:
                embedding, attention_mask = self.add_embedding(**self.to_device(data[key]), embedding_index=embedding_index)
                encode_result = self.encode(embedding, attention_mask, key)
                features[key] = encode_result["logits"]

        for part in self.final_input_parts:
            if part in features and part in self.aux_loss_fc:
                aux_pred = self.sigmoid(self.aux_loss_fc[part](features[part])).squeeze(-1)
                aux_loss_val = self.aux_loss(aux_pred, data["label"].float())
                losses[f"{part}_loss"] = aux_loss_val.mean()
                metrics[part] = aux_pred

        modality_keys = [k for k in self.final_input_parts if k in features]
        modality_features = [features[k] for k in modality_keys]
        stacked_features = torch.stack(modality_features, dim=-1)

        gate_inputs = torch.cat([features[k] for k in self.gate_input_parts if k in features], dim=-1)
        weights = self.gate_fc(gate_inputs)
        weights = torch.softmax(weights[:, :len(modality_features)], dim=-1)

        final_features = (stacked_features * weights.unsqueeze(1)).sum(dim=-1)
        preds = self.sigmoid(self.final_fc(final_features)).squeeze(-1)

        losses["loss"] = self.loss(preds, data["label"].float())
        metrics["preds"] = preds
        metrics["target"] = data["label"]

        return {
            "loss_dict": losses,
            "metric_dict": metrics,
            "hidden_state_dict": hidden_states,
        }
