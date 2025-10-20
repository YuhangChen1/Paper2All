import json
import os
from collections import defaultdict
from contextlib import contextmanager
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


@contextmanager
def science_plot(font_size=16):
    import scienceplots

    with plt.style.context(["ieee", "grid", "no-latex", "light"]):
        plt.rcParams.update({"font.size": font_size})
        yield


def statistic_humaneval(eval_file: str, print_diff: bool = False):
    llm_eval = json.load(open(eval_file))
    llm_data = []
    for dimension, files in llm_eval.items():
        for filename, values in files.items():
            try:
                setting, basename = filename.split("/", 2)[1:]
                basename = basename.split("/")[0]
                if not isinstance(values["score"], int):
                    raise ValueError(f"score is not int: {values['score']}")
                llm_data.append(
                    {
                        "setting": setting,
                        "sample": basename,
                        "dimension": dimension,
                        "score": values["score"],
                    }
                )
            except:
                continue

    file_path = "human_eval/human_scores_2024-12-13.xlsx"
    human_eval = pd.read_excel(file_path).to_dict("records")
    human_data = []
    for record in human_eval:
        setting = record.get("setting")
        score = record[dimension]
        basename = record["PPT"]
        try:
            score = int(score)
        except:
            continue
        human_data.append(
            {
                "setting": setting,
                "sample": basename,
                "dimension": dimension,
                "score": score,
            }
        )

    # Compare and output differences between human and llm evaluations
    llm_df = pd.DataFrame(llm_data)
    human_df = pd.DataFrame(human_data)
    merged = pd.merge(
        llm_df,
        human_df,
        on=["setting", "sample", "dimension"],
        suffixes=("_llm", "_human"),
        how="outer",
        indicator=True,
    )
    # Calculate and print correlation coefficients for common records
    common_records = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
    dimensions = common_records["dimension"].unique()

    for dimension in dimensions:
        scores_human = common_records[common_records["dimension"] == dimension][
            "score_human"
        ]
        scores_llm = common_records[common_records["dimension"] == dimension][
            "score_llm"
        ]
        pearson_correlation = pearsonr(scores_human, scores_llm)
        spearman_correlation = spearmanr(scores_human, scores_llm)
        print(
            f"{dimension}, pearson: {pearson_correlation}, spearman: {spearman_correlation}"
        )
        if print_diff:
            difference_df = common_records[
                common_records["score_human"] != common_records["score_llm"]
            ]
            for _, row in difference_df.iterrows():
                print(row)


def statistic_ppteval():
    data = []
    eval_files = glob("./data/evals/PPTCrew*")
    for eval_file in eval_files:
        setting = eval_file.split("/")[-1].removesuffix(".json")
        eval_stats = json.load(open(eval_file))
        for dimension, files in eval_stats.items():
            if dimension == "vision":
                dimension = "design"
            for filename, score in files.items():
                domain = filename.split("/")[1]
                if isinstance(score, dict):
                    score = score["score"]
                if isinstance(score, str):
                    continue
                if score > 5000 or score < 0:
                    continue
                data.append(
                    {
                        "setting": setting,
                        "dimension": dimension,
                        "sample": filename,
                        "score": score,
                        "domain": domain,
                    }
                )
    return pd.DataFrame(data)


def setting_perfomance(df: pd.DataFrame):
    df = df.drop(columns=["domain"])
    for setting, dimension in df[["setting", "dimension"]].drop_duplicates().values:
        avg_score = df[(df["setting"] == setting) & (df["dimension"] == dimension)][
            "score"
        ].mean()
        print(f"{setting}, {dimension}, {avg_score}")


def plot_correlation(df: pd.DataFrame):
    df = df.drop(columns=["domain"])
    correlation_matrix = df[df["setting"] == "PPTCrew-gpt-4o+gpt-4o+gpt-4o"][
        ["ppl", "fid", "content", "design"]
    ].corr()
    # Plot the heatmap with axis limits set from -1 to 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 15},
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("correlation.pdf", bbox_inches="tight")
    plt.show()


def domain_perfomance(df: pd.DataFrame):
    df = df[df["setting"] == "PPTCrew-gpt-4o+gpt-4o+gpt-4o"]
    for domain, scores in df.groupby("domain")["score"]:
        print(f"{domain}, {scores.mean()}")


if __name__ == "__main__":
    statistic_humaneval("./test-logic.json")
