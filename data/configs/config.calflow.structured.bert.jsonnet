local embedding_dim = 768;
local encoder_output_dim = 768;
local num_train_examples = 25436;  // should change depending on the training dataset
local num_epoch = 30;
local batch_size = 32;    // this setting will likely require 32GB GPU memory.
local num_gradient_accumulation_steps = 2;
local total_train_steps = num_train_examples * num_epoch / batch_size / num_gradient_accumulation_steps;

local attention_regularization = "mse:all:src_normalize:2.0";  // choices from: null, mse:token:src_normalize:{weight}, {mse:all:src_normalize:weight}
local child_derivation_use_root_utterance_encoding = true;


local dataset_reader = {
    "type": "decompositional",
    "pretrained_encoder_name": "bert-base-uncased",
    "attention_regularization": attention_regularization,
    "child_derivation_use_root_utterance_encoding": child_derivation_use_root_utterance_encoding
};

{
  "train_data_path": 'data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/train.sketch.floating.subtree_idx_info.jsonl',
  "validation_data_path": 'data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/valid.sketch.floating.subtree_idx_info.jsonl',
  "test_data_path": "data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/test.sketch.floating.subtree_idx_info.jsonl",
  "dataset_reader": dataset_reader,
  "model": {
    "type": "decompositional",
    "dataset_reader": dataset_reader,
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": "bert-base-uncased",
        }
      }
    },
    "encoder": {
      "type": "pass_through",
      "input_dim": embedding_dim,
    },
    "attention": {
      "type": "span_masking",
      "vector_dim": 256,
      "matrix_dim": encoder_output_dim,
      "span_mask_weight": 0.1
    },
    "target_embedding_dim": 128,
    "decoder_hidden_dim": 256,
    "num_decoder_layers": 2,
    "decoder_dropout": 0.2,
    "attention_regularization": attention_regularization,
    "child_derivation_use_root_utterance_encoding": true,
    "sketch_level_attention_regularization_only": false,
    "root_derivation_compute_span_scores": false,
    "child_derivation_compute_span_scores": false,
    "beam_size": 5,
    "max_decoding_steps": 250
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["source_tokens"],
      "batch_size": batch_size,
    },
    "num_workers": 4
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["source_tokens"],
      "batch_size": batch_size * 2,
    }
  },
  "trainer": {
      "num_epochs": num_epoch,
      "optimizer": {
          "type": "huggingface_adamw",
          "parameter_groups": [
            [["_(child_)?source_embedder\\..*\\.bias", "_(child_)?source_embedder\\..*\\.LayerNorm\\.weight"], {"weight_decay": 0.0}],
            [["^(?!_source_embedder)(?!_child_source_embedder)"], {"lr": 1e-3}],
          ],
          "lr": 3e-5,
          "weight_decay": 0.01
      },
      "learning_rate_scheduler": {
          "type": "polynomial_decay",
          "end_learning_rate": 0.0,
          "warmup_steps": total_train_steps * 0.1,
      },
      "grad_norm": 5.0,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "validation_metric": "+events_with_orgchart_em",
      "patience": 10,
      "cuda_device": 0,
      //"batch_callbacks": [{"type": "batch_logger"}]
  },
  "evaluate_on_test": true,
  "pytorch_seed": 0,
}
