import os
import json
import json
import time
import datetime
import requests
import polars as pl
from pathlib import Path


META_DIR = Path("./meta-kaggle")
OUTPUT_DIR = Path("./raw-episodes")
BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"

LOWEST_SCORE_THRESH = 1600
EPISODE_LIMIT_SIZE = None # Kaggle says don't do more than 3600 per day and 1 per second
COMPETITION_ID = 86411

TARGET_SUBMISSION_IDS = [
    43330490, 43330358, 43320130, 43317109,
    43294433, 43293811,
    43276830,
    43212846, 43212163,
    42704976
]


def saveEpisode(epid:int, sub_id:int) -> None:
    # request
    re = requests.post(GET_URL, json = {"episodeId": int(epid)})
        
    # save replay
    replay = re.json()
    
    with open(OUTPUT_DIR / f'{sub_id}/{sub_id}_{epid}.json', 'w') as f:
        json.dump(replay, f)


episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
episodes_df = (
    episodes_df
    .filter(pl.col('CompetitionId')==COMPETITION_ID)
    .with_columns(
        pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
        pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
    )
    .sort("Id")
    .collect()
)

# Remove episodes with leaderboard bug
episodes_df = episodes_df.filter((pl.col("Id") >= 67674959) | (pl.col("Id") <= 66881073))


agents_df = pl.scan_csv(
    META_DIR / "EpisodeAgents.csv", 
    schema_overrides={'Reward':pl.Float32, 'UpdatedConfidence': pl.Float32, 'UpdatedScore': pl.Float32}
)

agents_df = (
    agents_df
    .filter(pl.col("EpisodeId").is_in(episodes_df['Id'].to_list()))
    .collect()
)


start_time = datetime.datetime.now()
episode_count = 0
downloaded_episodes = 0
target_episodes_df = agents_df.filter(pl.col("SubmissionId").is_in(TARGET_SUBMISSION_IDS))
target_episodes_df = target_episodes_df.filter((pl.col("Reward") >= 3))
for _sub_id, df in target_episodes_df.group_by('SubmissionId'):
    sub_id = _sub_id[0]
    ep_ids = df['EpisodeId'].unique()
    target_dir = os.path.join(OUTPUT_DIR, str(sub_id))
    os.makedirs(target_dir, exist_ok=True)
    for epid in ep_ids:
        episode_count+=1
        
        if os.path.exists(OUTPUT_DIR / f'{sub_id}/{sub_id}_{epid}.json'):
            print(str(episode_count) + f': episode #{epid} already exists')
            continue
        
        saveEpisode(epid, sub_id); 
        downloaded_episodes += 1

        try:
            size = os.path.getsize(OUTPUT_DIR / f'{sub_id}/{sub_id}_{epid}.json') / 1e6
            print(str(episode_count) + f': saved episode #{epid}')
        except:
            print(f'  file {sub_id}_{epid}.json did not seem to save')

        # process 1 episode/sec
        spend_seconds = (datetime.datetime.now() - start_time).seconds
        if downloaded_episodes > spend_seconds:
            print(f"Sleep for {downloaded_episodes - spend_seconds:.2f}s")
            time.sleep(downloaded_episodes - spend_seconds)
            
        if EPISODE_LIMIT_SIZE and downloaded_episodes > EPISODE_LIMIT_SIZE:
            break 

    if EPISODE_LIMIT_SIZE and downloaded_episodes > EPISODE_LIMIT_SIZE:
            break

print(f'Episodes saved: {downloaded_episodes}')
