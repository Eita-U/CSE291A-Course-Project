from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "session_id",
    "step",
    "input",
    "output_length",
    "pre_gap",
}


def check_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def normalize_messages(value):
    if isinstance(value, np.ndarray):
        return normalize_messages(value.tolist())
    if isinstance(value, dict):
        return {key: normalize_messages(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_messages(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def load_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    check_columns(df)
    df["input"] = df["input"].map(normalize_messages)
    return df

def group_tasks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {task_id: task.sort_values("step") for task_id, task in df.groupby("session_id", sort=False)}