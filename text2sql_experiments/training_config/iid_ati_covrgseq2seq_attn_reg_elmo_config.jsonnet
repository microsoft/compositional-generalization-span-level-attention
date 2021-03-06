local attention_regularization = "mse:inner:4.0";

local local_dir = "";
{
  "dataset_reader": {
    "type": "text2sql_seq2seq_reader_att_reg",
    "database_path": null,
    "remove_unneeded_aliases": false,
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "source_token_indexers": {
      "elmo": { "type": "elmo_characters" },
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "source_tokenizer": { "type": "whitespace" },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    },
    "target_tokenizer": { "type": "standard" },
    "use_prelinked_entities": true,
    "attention_regularization": attention_regularization
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 4,
    "padding_noise": 0,
    "sorting_keys": [ [ "target_tokens", "num_tokens" ] ]
  },
  "model": {
    "type": "coverage_seq2seq",
    "attention_regularization": attention_regularization,
    "beam_size": 5,
    "coverage_lambda": 0,
    "dec_dropout": 0.2,
    "emb_dropout": 0.5,
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "dropout": 0,
      "hidden_size": 300,
      "input_size": 1124,
      "num_layers": 1
    },
    "max_decoding_steps": 300,
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "source_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "do_layer_norm": false,
        "dropout": 0,
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
      },
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "trainable": true,
        "vocab_namespace": "source_tokens"
      }
    },
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",
    "use_bleu": true
  },
  "train_data_path": local_dir +"data/sql data/atis/schema_full_split/aligned_train.parse_comp_heuristics_false_analyze_nested_false.consecutive_utt.jsonl",
  "validation_data_path": local_dir +"data/sql data/atis/schema_full_split/aligned_final_dev.jsonl",
  "test_data_path": local_dir +"data/sql data/atis/schema_full_split/final_test.jsonl",
  "trainer": {
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 600,
      "warmup_steps": 800
    },
    "num_epochs": 60,
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "patience": 15,
    "validation_metric": "+seq_acc",
  },
  "evaluate_on_test": true
}
