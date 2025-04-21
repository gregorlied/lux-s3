# Train Immitation Learning Agent

Step 1: Prepare dataset. Please refer to `data/README.md` for more information.

Step 2: Train Action UNet.
```bash
python train.py --config "./configs/config.yaml"
```

Step 3: Train Sap UNet
```bash
python train.py --config "./configs/sap_config.yaml"
```
