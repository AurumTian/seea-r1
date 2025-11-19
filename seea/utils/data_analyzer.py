import os
import re
import json
import shutil
import random
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
# Try to import seaborn, but make its usage optional in analyze_confusion_matrix
try:
    import seaborn as sns
except ImportError:
    sns = None

from seea.utils.logger import get_logger # Assuming this path is correct

logger = get_logger(__name__)

def read_jsonl_data(dataset_path):
    """
    Read data from a JSONL file.
    This is a helper, potentially needed by analyze_confusion_matrix if it reads files,
    or if aggregate_reward_data needs it before passing data.
    """
    data_items = []
    try:
        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    data_items.append(json.loads(stripped_line))
        return data_items
    except Exception as e:
        logger.error(f'Failed to read data file: {dataset_path}. Error: {str(e)}', exc_info=True)
        raise

def analyze_confusion_matrix(data_items, prefix="", output_dir=None):
    """
    Analyze prediction accuracy - simplified version
    Calculate accuracy for each class and overall accuracy
    
    Parameters:
    data_items: list - Data content list (list of dicts)
    prefix: str - Prefix for output file names
    output_dir: str - Output directory, uses current working directory if None
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'analysis_results') # Changed default
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f'Starting confusion matrix analysis. Output directory: {output_dir}, Prefix: {prefix}')

    # Counters
    confusion_matrix_data = defaultdict(lambda: defaultdict(int))
    class_counts = {'success': 0, 'continue': 0, 'failure': 0}
    class_correct = {'success': 0, 'continue': 0, 'failure': 0}
    total_samples = 0
    total_correct = 0
    
    # Analyze data
    for data_index, data in enumerate(data_items, 1):
        try:
            prediction_val = data.get('prediction', None)
            solution_val = data.get('solution', None)

            if prediction_val is not None and solution_val is not None:
                pred = str(prediction_val).lower()
                gt = str(solution_val).lower()

                valid_states_map = {
                    'state.success': 'success', 'state.continue': 'continue', 'state.failure': 'failure',
                    'success': 'success', 'continue': 'continue', 'failure': 'failure'
                }
                
                pred_mapped = valid_states_map.get(pred)
                gt_mapped = valid_states_map.get(gt)

                if pred_mapped is None:
                    # Use concatenation for complex strings if f-string quoting is an issue
                    log_msg = 'Item ' + str(data_index) + ' has invalid prediction value \'' + str(prediction_val) + '\' (processed to \'' + str(pred) + '\'). Skipping.'
                    logger.warning(log_msg)
                    continue
                
                if gt_mapped is None:
                    log_msg = 'Item ' + str(data_index) + ' has invalid solution value \'' + str(solution_val) + '\' (processed to \'' + str(gt) + '\'). Skipping.'
                    logger.warning(log_msg)
                    continue
                
                pred = pred_mapped
                gt = gt_mapped
                
                confusion_matrix_data[pred][gt] += 1
                class_counts[gt] += 1
                
                if pred == gt:
                    class_correct[gt] += 1
                    total_correct += 1
                
                total_samples += 1
            else:
                missing = []
                if prediction_val is None: missing.append("'prediction'")
                if solution_val is None: missing.append("'solution'")
                logger.warning(f'Item {data_index} is missing {", ".join(missing)} key(s). Skipping.')
                continue
        except Exception as e:
            logger.error(f'Error processing item {data_index} for confusion matrix: {e}', exc_info=True)
            continue
    
    if total_samples == 0:
        logger.warning("No valid samples found for confusion matrix analysis. Aborting analysis.")
        return {}, {}, 0

    class_accuracy = {}
    for cls_name in ['success', 'continue', 'failure']:
        class_accuracy[cls_name] = class_correct[cls_name] / class_counts[cls_name] if class_counts[cls_name] > 0 else 0
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    classes_order = ['success', 'continue', 'failure']
    matrix_display_data = np.zeros((len(classes_order), len(classes_order)), dtype=np.int32)
    for i, pred_cls in enumerate(classes_order):
        for j, gt_cls in enumerate(classes_order):
            matrix_display_data[i, j] = confusion_matrix_data[pred_cls][gt_cls]
    
    if sns:
        try:
            plt.figure(figsize=(10, 7)) # Adjusted figure size
            sns.heatmap(matrix_display_data, annot=True, fmt='.0f', cmap='YlOrRd',
                        xticklabels=classes_order, yticklabels=classes_order)
            title_prefix_viz = "Original " if prefix == "original_" else "Balanced " if "balanced_" in prefix else prefix
            plt.title(f'{title_prefix_viz}Confusion Matrix (Rows: Predicted, Cols: GT)')
            plt.xlabel('Ground Truth')
            plt.ylabel('Predicted')
            viz_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f'Confusion matrix visualization saved to: {viz_path}')
        except Exception as e:
            logger.error(f'Failed to save confusion matrix visualization: {str(e)}', exc_info=True)
    else:
        logger.warning("Seaborn library not found. Skipping confusion matrix heatmap visualization.")

    output_buffer = StringIO()
    title_prefix_report = "Original " if prefix == "original_" else "Balanced " if "balanced_" in prefix else prefix
    output_buffer.write(f'=== {title_prefix_report}Accuracy Analysis ===\n\n')
    output_buffer.write("Confusion Matrix (Rows: Predicted, Columns: Ground Truth):\n\n")
    header = f"{'Pred/GT':10}" + ''.join([f'{gt_class:>10}' for gt_class in classes_order])
    output_buffer.write(header + "\n")
    for pred_cls in classes_order:
        row = f'{pred_cls:10}' + ''.join([f'{confusion_matrix_data[pred_cls][gt_cls]:>10}' for gt_cls in classes_order])
        output_buffer.write(row + "\n")
    
    output_buffer.write("\nClass Accuracy (Correct / Total in class):\n")
    for cls_name in classes_order:
        output_buffer.write(f'{cls_name:10} - {class_accuracy[cls_name]:.2%} ({class_correct[cls_name]}/{class_counts[cls_name]})\n')
    output_buffer.write(f'\nOverall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})\n')
    output_buffer.write("\nData Distribution (Ground Truth):\n")
    for cls_name in classes_order:
        percentage = class_counts[cls_name] / total_samples * 100 if total_samples > 0 else 0
        output_buffer.write(f'{cls_name:10} - {class_counts[cls_name]} ({percentage:.1f}%)\n')
    
    report_path = os.path.join(output_dir, f'{prefix}accuracy_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f_report:
            f_report.write(output_buffer.getvalue())
        logger.info(f'Accuracy report saved to: {report_path}')
    except Exception as e:
        logger.error(f'Failed to save accuracy report: {str(e)}', exc_info=True)
    
    return confusion_matrix_data, class_accuracy, overall_accuracy

def aggregate_reward_data(main_save_dir, enable_ttrl_reward=False, perform_analysis=True):
    """
    Aggregates reward_model_data.jsonl files from subdirectories of main_save_dir,
    balances the data, and writes it into a single reward_model_data.jsonl file.
    Optionally performs confusion matrix analysis on the final aggregated data.
    """
    aggregated_file_path = os.path.join(main_save_dir, "reward_model_data.jsonl")
    temp_aggregated_file_path = aggregated_file_path + ".tmp"

    logger.info(f'Starting aggregation and balancing of reward_model_data.jsonl into {aggregated_file_path}')
    
    all_data_items = []
    found_files_count = 0

    if not os.path.isdir(main_save_dir):
        logger.error(f"Main save directory '{main_save_dir}' does not exist or is not a directory.") # Corrected quote
        return

    for item_name in os.listdir(main_save_dir):
        sub_dir_path = os.path.join(main_save_dir, item_name)
        if os.path.isdir(sub_dir_path):
            source_jsonl_path = os.path.join(sub_dir_path, "reward_model_data.jsonl")
            if os.path.exists(source_jsonl_path):
                logger.info(f'Processing file: {source_jsonl_path}')
                found_files_count += 1
                try:
                    with open(source_jsonl_path, 'r', encoding='utf-8-sig') as f_source: # utf-8-sig
                        for line_number, line in enumerate(f_source, 1):
                            try:
                                stripped_line = line.strip()
                                if stripped_line:
                                    data_item = json.loads(stripped_line)
                                    all_data_items.append(data_item)
                            except json.JSONDecodeError as e:
                                logger.warning(f'Skipping line {line_number} in {source_jsonl_path} due to JSON decode error: {e}')
                                continue
                except Exception as e:
                    logger.error(f'Error reading from {source_jsonl_path}: {e}', exc_info=True)

    if not all_data_items:
        logger.warning("No data items found to aggregate and balance.")
        if os.path.exists(temp_aggregated_file_path):
            try:
                os.remove(temp_aggregated_file_path)
            except OSError: pass
        return

    # --- Balancing Logic ---
    success_data, failure_data, continue_data = [], [], []
    state_map = {
        'state.success': 'success', 'state.continue': 'continue', 'state.failure': 'failure',
        'success': 'success', 'continue': 'continue', 'failure': 'failure'
    }
    
    data_source_for_categorization = []
    if enable_ttrl_reward:
        logger.info("enable_ttrl_reward is True. Using 'prediction' as 'solution'.")
        for item_index, item in enumerate(all_data_items):
            new_item = copy.deepcopy(item) 
            if 'prediction' in new_item:
                new_item['solution'] = new_item.pop('prediction')
                data_source_for_categorization.append(new_item)
            else:
                # Corrected f-string for logger.warning
                log_msg = f"Item at index {item_index} missing 'prediction' key (TTRL). Original item kept if 'solution' exists."
                logger.warning(log_msg)
                if 'solution' in new_item:
                     data_source_for_categorization.append(new_item)
                else:
                     logger.warning(f"Skipping item at index {item_index} as it has no 'prediction' for TTRL and no 'solution'.")
    else:
        data_source_for_categorization = all_data_items

    for data in data_source_for_categorization:
        if not isinstance(data, dict):
            logger.warning(f'Skipping non-dictionary item: {str(data)[:200]}')
            continue
        solution = data.get('solution')
        if solution is None:
            logger.warning(f"Item missing 'solution' key. Skipping: {str(data)[:200]}") # Corrected
            continue
        
        mapped_solution = state_map.get(str(solution).lower())
        if mapped_solution:
            if mapped_solution == 'success': success_data.append(data)
            elif mapped_solution == 'continue': continue_data.append(data)
            elif mapped_solution == 'failure': failure_data.append(data)
        else:
            # Corrected f-string for logger.warning
            log_msg = f"Item with invalid solution value '{solution}'. Skipping: {str(data)[:200]}"
            logger.warning(log_msg)
            continue

    success_failure_data = success_data + failure_data
    success_failure_count = len(success_failure_data)
    continue_count = len(continue_data)
    logger.info(f'Data counts before balancing: Success={len(success_data)}, Failure={len(failure_data)}, Continue={continue_count}')

    if continue_count > success_failure_count and success_failure_count > 0:
        logger.info(f"Sampling 'continue' data from {continue_count} down to {success_failure_count}") # Corrected
        continue_data = random.sample(continue_data, success_failure_count)
    elif success_failure_count == 0 and continue_count > 0:
        logger.warning("No 'success' or 'failure' samples. Keeping all 'continue' samples.")
    else:
         logger.info(f"'Continue' count ({continue_count}) not greater than Success+Failure count ({success_failure_count}) or SF count is 0. Keeping all 'continue' samples.") # Corrected

    balanced_data = success_failure_data + continue_data
    random.shuffle(balanced_data)
    total_lines_to_write = len(balanced_data)
    logger.info(f'Total items after balancing and shuffling: {total_lines_to_write}')
    # --- End Balancing Logic ---

    try:
        with open(temp_aggregated_file_path, 'w', encoding='utf-8') as f_agg:
            for item in balanced_data:
                try:
                    keys_to_keep = {"solution", "images", "messages"}
                    item_for_dumping = {key: item[key] for key in keys_to_keep if key in item}
                    f_agg.write(json.dumps(item_for_dumping, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.error(f'Error writing item to {temp_aggregated_file_path}: {e}. Item: {str(item)[:100]}', exc_info=True)
        
        if os.path.exists(temp_aggregated_file_path):
            if total_lines_to_write > 0:
                 shutil.move(temp_aggregated_file_path, aggregated_file_path)
                 logger.info(f'Aggregation and balancing complete. Processed {found_files_count} source file(s).')
                 logger.info(f'Total lines written: {total_lines_to_write} to {aggregated_file_path}')
                
                 if perform_analysis:
                     analysis_output_dir = os.path.join(main_save_dir, 'analysis_aggregated_balanced')
                     logger.info(f'Performing analysis on aggregated balanced data. Output to: {analysis_output_dir}')
                     try:
                         analyze_confusion_matrix(balanced_data, prefix="aggregated_balanced_", output_dir=analysis_output_dir)
                         logger.info(f'Analysis of aggregated data saved to {analysis_output_dir}')
                     except Exception as e:
                         logger.error(f'Error during final analysis of aggregated data: {e}', exc_info=True)
            else:
                logger.warning("No data to write after balancing. Aggregated file will not be created/updated.")
                if os.path.exists(temp_aggregated_file_path):
                    os.remove(temp_aggregated_file_path)
        else:
            logger.warning(f'Temporary aggregation file was not created. No data aggregated.')

    except IOError as e:
        logger.error(f'Error opening/writing aggregated file {temp_aggregated_file_path}: {e}', exc_info=True)
    except Exception as e:
        logger.error(f'An unexpected error occurred during aggregation: {e}', exc_info=True)
    finally:
        if os.path.exists(temp_aggregated_file_path):
            try:
                os.remove(temp_aggregated_file_path)
                logger.info(f'Cleaned up temporary file: {temp_aggregated_file_path}')
            except OSError as e:
                logger.error(f'Error removing temporary file {temp_aggregated_file_path}: {e}', exc_info=True) 