LajsConfig = {
              "pretrain_model_dir": '../model/bert',  # 预测时加载训练好的模型；训练时加载预训练模型
              "model_dir": "./save_model/",
              "init_checkpoint": "",

              "do_train": True,
              "warm_ratio": 0.1,
              "max_len": 512,
              "chunk_size": 254,
              "random_k": 4,
              "train_file": "./data/large_train_segment_selection_law1.json",
              "dev_file": "./data/small_dev_l.json",
              "train_epoch": 500,
              "train_batch_size": 1,
              "weight_decay": 0.01,
              "learning_rate": 3e-5,
              "accum_steps": 4,
              "print_steps": 107,
              "eval_epochs": 1,
              "save_ckpt_steps": 40000000,
              "max_grad": 1,

              "predict_file": "./data.json",
              "predict_batch_size": 1
              }

if __name__ == "__main__":
    pass