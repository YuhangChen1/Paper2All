import json
from collections import defaultdict
from typing import List, Optional, Set
import statistics
import argparse
from prettytable import PrettyTable, ALL


def calculate_average(scores: List[float]) -> float:
    """Safely calculates the average of a list of scores."""
    if not scores:
        return 0.0
    return statistics.mean(scores)


def calculate_metrics(metric_results_path: str, selected_ids_path: Optional[str] = None):
    """
    Calculates and prints overall evaluation results from a metric_results.jsonl file.

    Optionally filters the results to include only IDs specified in a given file.

    This script requires the 'prettytable' library to be installed.
    You can install it with: pip install prettytable
    """
    selected_ids: Set[str] = set()
    if selected_ids_path:
        try:
            with open(selected_ids_path, 'r', encoding='utf-8') as f:
                selected_ids = {line.strip() for line in f if line.strip()}
            if not selected_ids:
                print(f"WARNING: The file {selected_ids_path} was found but is empty. No filtering will be applied.")
            else:
                print(f"INFO: Filtering metrics for {len(selected_ids)} specific IDs from {selected_ids_path}.")
        except FileNotFoundError:
            print(f"ERROR: Selected IDs file not found at {selected_ids_path}. Aborting.")
            return
        except Exception as e:
            print(f"ERROR: An error occurred while reading the selected IDs file: {e}. Aborting.")
            return

    universal_scores = defaultdict(list)
    fg_obtained_scores: List[float] = []
    fg_possible_scores: List[float] = []
    preference_results = defaultdict(lambda: defaultdict(int))
    traditional_metrics_data = defaultdict(lambda: defaultdict(list))

    try:
        with open(metric_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)

                    if selected_ids:
                        item_id = item.get('id')
                        if not item_id or item_id not in selected_ids:
                            continue

                    eval_name = item.get('eval_name')
                    status = item.get('status')
                    results = item.get('evaluation_results', {})

                    if status != 'completed':
                        continue

                    if 'traditional_metrics' in results:
                        metrics = results['traditional_metrics']
                        rouge = metrics.get('rouge_scores', {})
                        bert = metrics.get('bert_score', {})
                        if 'ROUGE-1' in rouge:
                            traditional_metrics_data[eval_name]['ROUGE-1 F-measure'].append(rouge['ROUGE-1'].get('fmeasure', 0))
                        if 'ROUGE-2' in rouge:
                            traditional_metrics_data[eval_name]['ROUGE-2 F-measure'].append(rouge['ROUGE-2'].get('fmeasure', 0))
                        if 'ROUGE-L' in rouge:
                            traditional_metrics_data[eval_name]['ROUGE-L F-measure'].append(rouge['ROUGE-L'].get('fmeasure', 0))
                        if 'f1' in bert:
                            traditional_metrics_data[eval_name]['BERTScore F1'].append(bert.get('f1', 0))

                    elif 'vote_summary' in results:
                        vote_summary = results.get('vote_summary', {})
                        preference_results[eval_name]['pr_test_wins'] += vote_summary.get('pr_test', 0)
                        preference_results[eval_name]['original_wins'] += vote_summary.get('original', 0)
                        preference_results[eval_name]['ties'] += vote_summary.get('tie', 0)

                    elif 'fine_grained_assessment' in results:
                        fg_data = results.get('fine_grained_assessment')
                        if fg_data:
                            fg_obtained_scores.append(fg_data.get('total_obtained_score', 0))
                            fg_possible_scores.append(fg_data.get('total_possible_score', 0))

                    elif 'assessments' in results:
                        item_scores = []
                        for assessment in results['assessments']:
                            if isinstance(assessment, dict) and 'score' in assessment:
                                item_scores.append(assessment['score'])
                        
                        if item_scores:
                            avg_item_score = calculate_average(item_scores)
                            normalized_score = (avg_item_score - 1) / 4.0 if avg_item_score >= 1 else 0.0
                            universal_scores[eval_name].append(normalized_score)

                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed JSON line: {line.strip()}. Error: {e}")
                except Exception as e:
                    print(f"WARNING: An unexpected error occurred processing line: {line.strip()}. Error: {e}")

    except FileNotFoundError:
        print(f"ERROR: Metric results file not found at {metric_results_path}")
        return
    except Exception as e:
        print(f"ERROR: An error occurred while reading the metric file: {e}")
        return

    # --- Result Presentation (Unchanged) ---
    print("\n--- Universal Evaluation Dimensions (Overall Quality) ---")
    all_universal_scores = [score for scores_list in universal_scores.values() for score in scores_list]
    if all_universal_scores:
        table = PrettyTable()
        table.field_names = ["Dimension", "Average Score (0-1)", "Count"]
        table.align["Dimension"] = "l"
        table.align["Average Score (0-1)"] = "r"
        table.align["Count"] = "r"
        
        overall_avg = calculate_average(all_universal_scores)
        table.add_row(["Overall", f"{overall_avg:.4f}", len(all_universal_scores)])
        table.add_row(["-"*10, "-"*20, "-"*7], divider=True)

        for eval_name, scores in sorted(universal_scores.items()):
            if scores:
                avg_score = calculate_average(scores)
                table.add_row([eval_name, f"{avg_score:.4f}", len(scores)])
        print(table)
    else:
        print("No completed universal evaluations found.")

    print("\n--- Fine-Grained Evaluation (Factual Checklist Score) ---")
    if fg_obtained_scores:
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"
        avg_obtained = calculate_average(fg_obtained_scores)
        avg_possible = calculate_average(fg_possible_scores)
        avg_fg_score = avg_obtained / avg_possible if avg_possible > 0 else 0.0
        table.add_row(["Average Normalized Score", f"{avg_fg_score:.4f}"])
        table.add_row(["Total Fine-Grained Evaluations", len(fg_obtained_scores)])
        print(table)
    else:
        print("No completed fine-grained evaluations found.")

    print("\n--- Traditional Similarity Metrics (Averages) ---")
    if not traditional_metrics_data:
        print("No completed traditional metric evaluations found.")
    else:
        for eval_name, data in sorted(traditional_metrics_data.items()):
            count = len(next(iter(data.values()), []))
            if count == 0:
                print(f"\n--- {eval_name} ---")
                print("      No data to report.")
                continue
            table = PrettyTable()
            table.title = f"{eval_name} (Total Comparisons: {count})"
            table.field_names = ["Metric", "Average Score"]
            table.align["Metric"] = "l"
            table.align["Average Score"] = "r"
            table.hrules = ALL
            for metric_name, scores in sorted(data.items()):
                avg_score = calculate_average(scores)
                table.add_row([metric_name, f"{avg_score:.4f}"])
            print(table)

    print("\n--- Preference Evaluation ---")
    if not preference_results:
        print("No completed preference evaluations found.")
    else:
        for eval_name, counts in sorted(preference_results.items()):
            total_votes = sum(counts.values())
            if total_votes == 0:
                print(f"\n--- {eval_name} ---")
                print("      No completed comparisons.")
                continue
            pr_wins = counts.get('pr_test_wins', 0)
            ties = counts.get('ties', 0)
            orig_wins = counts.get('original_wins', 0)
            table = PrettyTable()
            table.title = f"{eval_name} (Total Votes Cast: {total_votes})"
            table.field_names = ["Outcome", "Votes", "Percentage"]
            table.align["Outcome"] = "l"
            table.align["Votes"] = "r"
            table.align["Percentage"] = "r"
            table.hrules = ALL
            table.add_row(["PR Test Wins", pr_wins, f"{pr_wins / total_votes:.2%}"])
            table.add_row(["Ties", ties, f"{ties / total_votes:.2%}"])
            table.add_row(["Original Wins", orig_wins, f"{orig_wins / total_votes:.2%}"])
            print(table)
            
    # --- Composite Metrics Calculation and LaTeX Output ---
    print("\n--- Composite Metrics Summary (New Format) ---")
    
    # Define metric keys (aliases)
    S1 = "S1_Authorship_and_Title_Accuracy"
    S2 = "S2_Logic_Attractiveness"
    S3 = "S3_Contextual_Relevance"
    S4 = "S4_Visual_Attractiveness"
    S5 = "S5_Optimal_Visual_to_Text_Ratio"
    S7 = "S7_Engagement_Hook_Strength"
    S8 = "S8_Hashtag_and_Mention_Strategy"
    S9 = "S9_CTA_Checklist_Score"
    P1 = "P1_Overall_Preference_Comparison"
    P2 = "P2_Professional_Interest_Preference"
    P3 = "P3_SciComm_Strategy_Preference"

    # Helper function to get win rate, returns None if no data
    def get_win_rate(eval_name: str, pref_results: defaultdict) -> Optional[float]:
        counts = pref_results.get(eval_name, {})
        total_votes = sum(counts.values())
        if total_votes == 0:
            return None
        return counts.get('pr_test_wins', 0) / total_votes

    # 1. Calculate Fidelity Metrics
    at_acc = calculate_average(universal_scores.get(S1, []))
    avg_fg_obtained = calculate_average(fg_obtained_scores)
    avg_fg_possible = calculate_average(fg_possible_scores)
    factual_score = avg_fg_obtained / avg_fg_possible if avg_fg_possible > 0 else 0.0
    
    # Fidelity is now a simple average
    fidelity_overall = calculate_average([at_acc, factual_score])

    # 2. Calculate Engagement Metrics
    hook = calculate_average(universal_scores.get(S7, []))
    logical_attr = calculate_average(universal_scores.get(S2, []))
    cta = calculate_average(universal_scores.get(S9, []))
    
    # Handle potentially missing S4 data
    visual_attr_scores = universal_scores.get(S4, [])
    visual_attr = calculate_average(visual_attr_scores)
    
    prof_pref = get_win_rate(P2, preference_results)
    broad_pref = get_win_rate(P3, preference_results)
    
    # Build list for overall average, excluding missing values
    eng_scores_for_avg = [hook, logical_attr, cta]
    if visual_attr_scores: # Only include if S4 data exists
        eng_scores_for_avg.append(visual_attr)
    if prof_pref is not None: # Only include if P2 data exists
        eng_scores_for_avg.append(prof_pref)
    if broad_pref is not None: # Only include if P3 data exists
        eng_scores_for_avg.append(broad_pref)
    engagement_overall = calculate_average(eng_scores_for_avg)

    # 3. Calculate Alignment Metrics
    context_rel = calculate_average(universal_scores.get(S3, []))
    hashtag = calculate_average(universal_scores.get(S8, []))
    
    # Handle potentially missing S5 data
    vis_txt_integ_scores = universal_scores.get(S5, [])
    vis_txt_integ = calculate_average(vis_txt_integ_scores)
    
    plat_pref = get_win_rate(P1, preference_results)
    
    # Build list for overall average, excluding missing values
    align_scores_for_avg = [context_rel, hashtag]
    if vis_txt_integ_scores: # Only include if S5 data exists
        align_scores_for_avg.append(vis_txt_integ)
    if plat_pref is not None: # Only include if P1 data exists
        align_scores_for_avg.append(plat_pref)
    alignment_overall = calculate_average(align_scores_for_avg)
    
    # Create a PrettyTable for the new composite metrics format
    comp_table = PrettyTable()
    comp_table.title = "Composite Metrics (Based on Table)"
    comp_table.field_names = ["Category", "Metric", "Score"]
    comp_table.align["Category"] = "l"
    comp_table.align["Metric"] = "l"
    comp_table.align["Score"] = "r"
    comp_table.hrules = ALL
    
    # Fidelity
    comp_table.add_row(["Fidelity", "A&T Acc.", f"{at_acc:.2f}"])
    comp_table.add_row(["Fidelity", "Factual Score", f"{factual_score:.2f}"])
    comp_table.add_row(["Fidelity", "Overall", f"{fidelity_overall*100:.2f}"])
    # Engagement
    comp_table.add_row(["Engagement", "Hook", f"{hook:.2f}"])
    comp_table.add_row(["Engagement", "Logical Attr.", f"{logical_attr:.2f}"])
    comp_table.add_row(["Engagement", "Visual Attr.", f"{visual_attr:.2f}" if visual_attr_scores else "-"])
    comp_table.add_row(["Engagement", "CTA", f"{cta:.2f}"])
    comp_table.add_row(["Engagement", "Prof. Pref.", f"{prof_pref*100:.1f}%" if prof_pref is not None else "-"])
    comp_table.add_row(["Engagement", "Broad Pref.", f"{broad_pref*100:.1f}%" if broad_pref is not None else "-"])
    comp_table.add_row(["Engagement", "Overall", f"{engagement_overall*100:.2f}"])
    # Alignment
    comp_table.add_row(["Alignment", "Context Rel.", f"{context_rel:.2f}"])
    comp_table.add_row(["Alignment", "Vis-Txt Integ.", f"{vis_txt_integ:.2f}" if vis_txt_integ_scores else "-"])
    comp_table.add_row(["Alignment", "Hashtag", f"{hashtag:.2f}"])
    comp_table.add_row(["Alignment", "Plat. Pref.", f"{plat_pref*100:.1f}%" if plat_pref is not None else "-"])
    comp_table.add_row(["Alignment", "Overall", f"{alignment_overall*100:.2f}"])
    print(comp_table)
    
    # --- LaTeX Output ---
    # Prepare strings for LaTeX output, using '-' for missing values
    visual_attr_str = f"{visual_attr*100:.2f}" if visual_attr_scores else "-"
    vis_txt_integ_str = f"{vis_txt_integ*100:.2f}" if vis_txt_integ_scores else "-"
    prof_pref_str = f"{prof_pref * 100:.2f}" if prof_pref is not None else "-"
    broad_pref_str = f"{broad_pref * 100:.2f}" if broad_pref is not None else "-"
    plat_pref_str = f"{plat_pref * 100:.2f}" if plat_pref is not None else "-"
    
    #print("\n--- LaTeX Output Line (Original) ---")
    latex_line = (
        f"Model & {at_acc:.4f} & {factual_score:.4f} & {fidelity_overall:.4f} & "
        f"{hook:.4f} & {logical_attr:.4f} & {visual_attr_str} & {cta:.4f} & "
        f"{prof_pref_str} & {broad_pref_str} & {engagement_overall:.4f} & "
        f"{context_rel:.4f} & {vis_txt_integ_str} & {hashtag:.4f} & "
        f"{plat_pref_str} & {alignment_overall:.4f} \\\\"
    )
    #print(latex_line)

    # --- LaTeX Output Line (All Metrics Average) ---
    print("\n--- LaTeX Output Line ---")
    all_metrics_for_avg = []
    
    # Collect all available scores
    all_metrics_for_avg.append(at_acc)
    all_metrics_for_avg.append(factual_score)
    all_metrics_for_avg.append(hook)
    all_metrics_for_avg.append(logical_attr)
    if visual_attr_scores: all_metrics_for_avg.append(visual_attr)
    all_metrics_for_avg.append(cta)
    if prof_pref is not None: all_metrics_for_avg.append(prof_pref)
    if broad_pref is not None: all_metrics_for_avg.append(broad_pref)
    all_metrics_for_avg.append(context_rel)
    if vis_txt_integ_scores: all_metrics_for_avg.append(vis_txt_integ)
    all_metrics_for_avg.append(hashtag)
    if plat_pref is not None: all_metrics_for_avg.append(plat_pref)
    
    overall_average = calculate_average(all_metrics_for_avg)
    
    # Prepare strings for the new LaTeX line
    at_acc_str = f"{at_acc*100:.2f}"
    factual_score_str = f"{factual_score*100:.2f}"
    hook_str = f"{hook*100:.2f}"
    logical_attr_str = f"{logical_attr*100:.2f}"
    # visual_attr_str is already defined
    cta_str = f"{cta*100:.2f}"
    # prof_pref_str is already defined
    # broad_pref_str is already defined
    context_rel_str = f"{context_rel*100:.2f}"
    # vis_txt_integ_str is already defined
    hashtag_str = f"{hashtag*100:.2f}"
    # plat_pref_str is already defined
    overall_average_str = f"{overall_average*100:.2f}"

    latex_line_2 = (
        f"Model & {at_acc_str} & {factual_score_str} & "
        f"{hook_str} & {logical_attr_str} & {visual_attr_str} & {cta_str} & "
        f"{prof_pref_str} & {broad_pref_str} & "
        f"{context_rel_str} & {vis_txt_integ_str} & {hashtag_str} & "
        f"{plat_pref_str} & {overall_average_str} \\\\"
    )
    print(latex_line_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics from metric_results.jsonl.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--metric_output_path", 
        type=str, 
        default="metric_results.jsonl",
        help="Path to the output metric results file generated by main_eval.py."
    )
    parser.add_argument(
        "--selected_ids_path", 
        type=str, 
        default=None,
        help="Optional. Path to a .txt file containing one ID per line.\n"
             "If provided, metrics will only be calculated for these specific IDs."
    )
    args = parser.parse_args()
    
    calculate_metrics(args.metric_output_path, args.selected_ids_path)