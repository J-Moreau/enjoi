from pathlib import Path

import polars as pl
from wandb import Api


def attach_sampled_x_paths(
    selected_runs: pl.DataFrame,
    download_root: Path = Path("results/artifacts/sampled_x"),
) -> pl.DataFrame:
    api = Api()
    filters = {"name": {"$in": selected_runs["run_id"].to_list()}}
    downloaded_runs: list[dict[str, str]] = []

    for run in api.runs("pdediff", filters=filters):
        for artifact in run.logged_artifacts():
            if not any(file.name.endswith("sampled_x.npy") for file in artifact.files()):
                continue
            artifact_dir = Path(artifact.download(root=download_root / run.id))
            run_id = next(
                run_id
                for run_id in selected_runs["run_id"]
                if run_id == run.id
            )
            downloaded_runs.append(
                {
                    "run_id": run_id,
                    "sampled_x_path": str(next(artifact_dir.rglob("sampled_x.npy"))),
                }
            )

    return selected_runs.join(
        pl.DataFrame(
            downloaded_runs,
            schema={"run_id": pl.String, "sampled_x_path": pl.String},
        ),
        on="run_id",
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col.eval.struct.field("autoregressive").struct.field("variant"),
    )
