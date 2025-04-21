# Prepare Dataset for Immitation Learning

Step 1: Download `Episodes.csv` and `EpisodeAgents.csv` from the Meta Kaggle dataset. \[1\]

Step 2: Move the datasets into `luxai-s3/data/meta-kaggle`.

Step 3: Download episodes using
```bash
python download_episodes.py
```

The raw episodes will be stored in `luxai-s3/data/raw-episodes`.

Step 4: Process episodes using
```bash
python process_episodes.py
```

The processed episodes will be stored in `luxai-s3/data/dataset`.

\[1\]: https://www.kaggle.com/datasets/kaggle/meta-kaggle
