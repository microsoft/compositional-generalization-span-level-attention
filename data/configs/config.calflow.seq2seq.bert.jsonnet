local embedding_dim = 768;
local encoder_output_dim = 768;
local num_train_examples = 25436;
local num_epoch = 30;
local batch_size = 32;
local num_gradient_accumulation_steps = 2;
local total_train_steps = num_train_examples * num_epoch / batch_size / num_gradient_accumulation_steps;  # ~ 11920

local attention_regularization = "mse:all:src_normalize:2.0"; // null, mse:token:src_normalize:weight, mse:all:src_normalize:weight


{
  "train_data_path": "data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/train.alignment.jsonl",
  "validation_data_path": "data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/valid.alignment.jsonl",
  "test_data_path": "data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num32/test.alignment.jsonl",
  "dataset_reader": {
    "type": "seq2seq_with_copy",
    "pretrained_encoder_name": "bert-base-uncased",
    "attention_regularization": attention_regularization,
  },
  "vocabulary": {
    "min_count": {
        "source_tokens": 2
    }
  },
  "model": {
    "type": "seq2seq_with_copy",
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
      "type": "bilinear",
      "vector_dim": 256,
      "matrix_dim": encoder_output_dim
    },
    "attention_regularization": attention_regularization,
    "target_embedding_dim": 128,
    "decoder_hidden_dim": 256,
    "num_decoder_layers": 2,
    "decoder_dropout": 0.2,
    "beam_size": 5,
    "max_decoding_steps": 250,
    "use_sketch_metric": true
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,
      "sorting_keys": ["source_tokens"],
    },
    "num_workers": 4
  },
  "trainer": {
      "num_epochs": num_epoch,
      "optimizer": {
          "type": "huggingface_adamw",
          "parameter_groups": [
            [["_source_embedder\\..*\\.bias", "_source_embedder\\..*\\.LayerNorm\\.weight"], {"weight_decay": 0.0}],
            [["^(?!_source_embedder)"], {"lr": 1e-3}],
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
  },
  "evaluate_on_test": true,
  "pytorch_seed": 0
}



