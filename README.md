# NeurIPS 2024 - Lux AI Season3 8th Place Solution

This repository contains the code for the 8th place solution in the NeurIPS 2024 - Lux AI Season 3 competition.

For a summary of the solution, please refer to [this discussion](https://www.kaggle.com/competitions/lux-ai-season-3/discussion/570673).

## Setup

The code has been tested with Python 3.10.12.

```bash
pip install -r requirements.txt
```

## Data Preperation

Please refer to `data/README.md`.

## Training

Please refer to `model/README.md`.

## Submission

Run the following command and upload the archive to Kaggle.

```bash
tar --exclude='data' --exclude='model' -czf submission.tar.gz *
```

You can find my final submissions here:
- https://github.com/gregorlied/lux-s3/releases/tag/v73a
- https://github.com/gregorlied/lux-s3/releases/tag/v73b
