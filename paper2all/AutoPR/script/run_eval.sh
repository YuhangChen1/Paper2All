#!/bin/bash

# ==============================================================================
# Script to run the academic promotion benchmark.
#
# This script configures and executes the Python evaluation module.
# You can set paths, global overrides, and select specific evaluations to run or reset.
#
# Usage:
# 1. Ensure your API credentials are in a .env file in the project root.
# 2. Configure the variables in the "Configuration" section below.
# 3. Make the script executable: chmod +x run_all_evals.sh
# 4. Run from the project's root directory: ./run_evals.sh
# ==============================================================================

# --- Configuration ---
# Set the core paths and parameters for the benchmark run.

# Path to the main data file containing original promotion data items.
DATA_PATH="eval/data/academic_promotion_data_core.json"
# Directory containing the YAML configuration files for each evaluation.
CONFIGS_DIR="eval/configs"
# File path where the evaluation metric results will be saved (in JSONL format).
METRIC_OUTPUT_PATH="llm_metric_result.jsonl"
# Number of concurrent requests to send to the LLM API.
CONCURRENCY=10

# Determines the source of the data to be evaluated.
# Options:
# "original": Use the data from the main JSON file specified in DATA_PATH.
# "pr_test": Use the data found in the directory specified by PR_TEST_DIR.
TARGET_DATA_SOURCE="pr_test"

# Path to the directory containing test data.
# This is only used if TARGET_DATA_SOURCE is set to "pr_test".
PR_TEST_DIR="output"


# --- Global Override Settings ---
# These settings will override the corresponding values in the YAML config files for all evaluations.
# Leave a variable empty or set to `false` to use the settings from the individual YAML files.

# Override the model for all evaluations (e.g., "gemini-1.5-pro-latest").
OVERRIDE_MODEL=""

# Override the image handling strategy for all applicable evals.
# This will NOT change evals that are originally set to "none" in their YAML config.
# Options: "real", "placeholder"
# This option has been deprecated; please do not modify it
OVERRIDE_IMAGES=""

# Force JSON output via prompt injection. Useful for models without native JSON mode support.
# Options: true, false
FORCE_JSON_PROMPT=true


# --- Evaluation Selection & Reset ---
# Control which evaluations are run or have their previous results cleared.

# To run ONLY specific evaluations, list their 'eval_name' from the YAML files here.
# To run ALL evaluations found in the configs directory, leave this list empty: EVALS_TO_RUN=()
EVALS_TO_RUN=(
    "S1_Authorship_and_Title_Accuracy"
    "S2_Logic_Attractiveness"
    "S3_Contextual_Relevance"
    "S4_Visual_Attractiveness"
    "S5_Optimal_Visual_to_Text_Ratio"
    "S7_Engagement_Hook_Strength"
    "S8_Hashtag_and_Mention_Strategy"
    "S9_CTA_Checklist_Score"
    "P1_Overall_Preference_Comparison"
    "P2_Professional_Interest_Preference"
    "P3_SciComm_Strategy_Preference"
    # "Traditional Content Similarity (ROUGE & BERTScore)"
    "Factual Accuracy Assessment"
)

# To reset specific evaluations before running, list their names here.
# This deletes their entries from the results file, forcing a re-run.
# To reset no evaluations, leave this list empty: EVALS_TO_RESET=()
EVALS_TO_RESET=()


# --- Execution ---
# This section builds and runs the command based on the configuration above. Do not modify.
echo "--- Starting Benchmark ---"

# --- Validity Check ---
# If targeting 'pr_test' data, ensure the specified directory actually exists.
if [ "$TARGET_DATA_SOURCE" = "pr_test" ]; then
    if [ ! -d "$PR_TEST_DIR" ]; then
        echo "❌ Error: TARGET_DATA_SOURCE is 'pr_test', but the directory '$PR_TEST_DIR' does not exist." >&2
        exit 1
    fi
fi

# --- Command Building ---
# Start with the base command arguments.
CMD_ARGS=(
    "--data-path" "$DATA_PATH"
    "--configs-dir" "$CONFIGS_DIR"
    "--metric-output-path" "$METRIC_OUTPUT_PATH"
    "--concurrency" "$CONCURRENCY"
    "--target-data-source" "$TARGET_DATA_SOURCE"
)

# Add the PR test directory path, but only if it's the target data source.
if [ "$TARGET_DATA_SOURCE" = "pr_test" ]; then
    CMD_ARGS+=("--pr-test-dir" "$PR_TEST_DIR")
fi

# Add the model override argument if the variable is set.
if [ -n "$OVERRIDE_MODEL" ]; then
    echo "🔵 Overriding model for all evaluations to: $OVERRIDE_MODEL"
    CMD_ARGS+=("--model" "$OVERRIDE_MODEL")
fi

# Add the image handling override argument if the variable is set.
if [ -n "$OVERRIDE_IMAGES" ]; then
    echo "🖼️  Overriding image handling for applicable evaluations to: $OVERRIDE_IMAGES"
    CMD_ARGS+=("--include-images-override" "$OVERRIDE_IMAGES")
fi

# Add the force JSON prompt flag if it's set to true.
if [ "$FORCE_JSON_PROMPT" = true ]; then
    echo "📄 Forcing JSON output via prompt injection."
    CMD_ARGS+=("--force-json-prompt")
fi

# Add the reset argument if the reset list is not empty.
if [ ${#EVALS_TO_RESET[@]} -gt 0 ]; then
    echo "🔄 Resetting SELECTED evaluations:"
    printf " - %s\n" "${EVALS_TO_RESET[@]}"
    CMD_ARGS+=("--reset-metrics" "${EVALS_TO_RESET[@]}")
fi

# Add the run argument if the run list is not empty.
if [ ${#EVALS_TO_RUN[@]} -gt 0 ]; then
    echo "▶️  Running SELECTED evaluations:"
    printf " - %s\n" "${EVALS_TO_RUN[@]}"
    CMD_ARGS+=("--run-evals" "${EVALS_TO_RUN[@]}")
else
    # If the run list is empty, the script will run all evaluations by default.
    echo "▶️  Running ALL available evaluations."
fi

echo "-------------------------------"

# Execute the python script with all the dynamically constructed arguments.
python -m eval.main_eval "${CMD_ARGS[@]}"

# Capture the exit code of the python script.
EXIT_CODE=$?

# --- Completion ---
# Report the final status of the benchmark run based on the exit code.
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ --- Benchmark run finished successfully. ---"
    echo "Results are saved in $METRIC_OUTPUT_PATH"
else
    echo "❌ --- Benchmark run failed with exit code $EXIT_CODE. ---"
fi

exit $EXIT_CODE
