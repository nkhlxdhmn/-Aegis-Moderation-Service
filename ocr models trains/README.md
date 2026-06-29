# test-content-moderation

## Service vs Promotional SMS Classifier

Train the binary classifier on `promotional/FINAL_MERGED_SMS_DATASET.csv`:

```bash
/home/ubuntu/miniconda3/envs/torch_env/bin/python promotional_sms_train.py
```

The script saves the model and metrics under `models/promotional_service_model/`.

To use all four GPUs, launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/ubuntu/miniconda3/envs/torch_env/bin/torchrun --nproc_per_node=4 promotional_sms_train.py --cuda-devices 0,1,2,3 --batch-size 8
```

If you want a quicker smoke test on all GPUs, add `--max-samples 20000`.
