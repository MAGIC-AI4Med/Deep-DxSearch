import json
import numpy as np
import re
from collections import defaultdict

def extract_textbf_content(answer_text):
    """Extract content within \textbf{} tags from the answer text."""
    pattern = r'\\textbf\{([^}]+)\}'
    matches = re.findall(pattern, answer_text)
    return [match.strip() for match in matches]

def has_intersection(predicted_diseases, ground_truth_diseases):
    """Check if there is any intersection between predicted diseases and ground truth."""
    # Convert to lowercase for case-insensitive matching
    pred_lower = [d.lower() for d in predicted_diseases]
    truth_lower = [d.lower() for d in ground_truth_diseases]
    
    # Check for direct matches
    for pred in pred_lower:
        if pred in truth_lower:
            return True
    
    # Check for substring matches (if a prediction is part of a ground truth or vice versa)
    for pred in pred_lower:
        for truth in truth_lower:
            if pred in truth or truth in pred:
                return True
    
    return False

def get_data_source_group(data_source):
    """Determine which group a data source belongs to."""
    data_source_upper = data_source.upper()
    
    if 'RAMEDIS' in data_source_upper:
        return 'RAMEDIS'
    elif 'LIRICAL' in data_source_upper:
        return 'LIRICAL'
    elif 'HMS' in data_source_upper:
        return 'HMS'
    elif 'MME' in data_source_upper:
        return 'MME'
    elif 'RAREARENA' in data_source_upper:
        return 'RareArena'
    else:
        # Return the original data source if no match found
        return data_source

def compute_metrics(jsonl_path):
    scores_dict = defaultdict(list)
    data_source_scores = defaultdict(lambda: defaultdict(list))
    
    # For tracking textbf matches - using defaultdict for automatic data source detection
    textbf_match_counts = defaultdict(lambda: {'matched': 0, 'total': 0})
    textbf_match_counts['overall'] = {'matched': 0, 'total': 0}
    
    # Read and process the jsonl file
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Determine which group this data source belongs to
            data_source = data.get('data_source', 'unknown')
            group = get_data_source_group(data_source)
            
            # Collect all score types
            for score_type in ['score', 'refer_score', 'search_score', 'answer_score']:
                # Convert None or missing values to 0
                score_value = data.get(score_type, None)
                score_value = float(0.0 if score_value is None else score_value)
                scores_dict[score_type].append(score_value)
                
                # Add to the appropriate data source group
                data_source_scores[group][score_type].append(score_value)
            
            # Extract assistant's answer and check for textbf matches
            sequences_str = data.get('sequences_str', '')
            ground_truth = data.get('ground_truth', [])
            
            # First, extract the assistant's response section
            assistant_match = re.search(r'<\|im_start\|>assistant(.*?)<\|im_end\|>', sequences_str, re.DOTALL)
            if assistant_match and ground_truth:
                assistant_text = assistant_match.group(1).strip()
                
                # Then find the answer section within the assistant's response
                answer_match = re.search(r'<answer>(.*?)</answer>', assistant_text, re.DOTALL)
                if answer_match:
                    answer_text = answer_match.group(1).strip()
                    
                    # Extract diseases from textbf tags
                    predicted_diseases = extract_textbf_content(answer_text)
                    
                    # Check if there's an intersection between predicted diseases and ground truth
                    has_match = has_intersection(predicted_diseases, ground_truth)
                    
                    # Update counts
                    textbf_match_counts['overall']['total'] += 1
                    if has_match:
                        textbf_match_counts['overall']['matched'] += 1
                    
                    # Update group-specific counts
                    textbf_match_counts[group]['total'] += 1
                    if has_match:
                        textbf_match_counts[group]['matched'] += 1

    # Calculate overall means
    overall_means = {
        score_type: np.mean(scores) if scores else 0.0
        for score_type, scores in scores_dict.items()
    }
    
    # Calculate means per data source group
    data_source_means = {}
    for source, source_scores in data_source_scores.items():
        if any(source_scores.values()):  # Only include if there are scores
            data_source_means[source] = {
                score_type: np.mean(scores) if scores else 0.0
                for score_type, scores in source_scores.items()
            }
    
    # Calculate match rates
    textbf_match_rates = {}
    for group, counts in textbf_match_counts.items():
        if counts['total'] > 0:
            textbf_match_rates[group] = counts['matched'] / counts['total']
        else:
            textbf_match_rates[group] = 0.0
    
    return overall_means, data_source_means, textbf_match_rates, textbf_match_counts

def print_results(val_num, overall_means, data_source_means, textbf_match_rates, textbf_match_counts):
    print(f"\n{'='*50}")
    print(f"Results for val_{val_num}")
    print(f"{'='*50}")
    
    print("\nOverall Mean Scores:")
    print("-" * 40)
    for score_type, mean in overall_means.items():
        print(f"{score_type:15s}: {mean:.3f}")
    
    print("\nMean Scores by Data Source Group:")
    print("-" * 40)
    for source, scores in data_source_means.items():
        print(f"\n{source}:")
        for score_type, mean in scores.items():
            print(f"  {score_type:15s}: {mean:.3f}")
    
    print("\nDisease Match Rates (\\textbf match with ground truth):")
    print("-" * 40)
    for group, rate in textbf_match_rates.items():
        counts = textbf_match_counts[group]
        print(f"{group:15s}: {rate:.3f} ({counts['matched']}/{counts['total']})")

def print_combined_match_rate_table(all_results):
    """Print a combined table of disease match rates across all val numbers."""
    
    print(f"\n{'='*80}")
    print("COMBINED DISEASE MATCH RATE TABLE")
    print(f"{'='*80}")
    
    # Get all unique data sources
    all_sources = set()
    val_nums = []
    for val_num, (_, _, match_rates, _) in all_results.items():
        val_nums.append(val_num)
        all_sources.update(match_rates.keys())
    
    val_nums.sort()
    all_sources = sorted(all_sources)
    
    # Print header
    header = f"{'Data Source':<15}"
    for val_num in val_nums:
        header += f"{'val_' + str(val_num):>10}"
    print(header)
    print("-" * len(header))
    
    # Print data for each source
    for source in all_sources:
        if source == 'overall':
            continue  # Skip overall for now, print it at the end
            
        row = f"{source:<15}"
        for val_num in val_nums:
            match_rates = all_results[val_num][2]
            if source in match_rates:
                row += f"{match_rates[source]:>10.3f}"
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    # Print overall at the end
    if 'overall' in all_sources:
        print("-" * len(header))
        row = f"{'Overall':<15}"
        for val_num in val_nums:
            match_rates = all_results[val_num][2]
            if 'overall' in match_rates:
                row += f"{match_rates['overall']:>10.3f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

if __name__ == "__main__":
    val_nums = [...]
    all_results = {}
    
    for val in val_nums:
        jsonl_path = f".../path/to/outputs/val_{val}.jsonl"
        
        try:
            overall_means, data_source_means, textbf_match_rates, textbf_match_counts = compute_metrics(jsonl_path)
            all_results[val] = (overall_means, data_source_means, textbf_match_rates, textbf_match_counts)
            print_results(val, overall_means, data_source_means, textbf_match_rates, textbf_match_counts)
        except FileNotFoundError:
            print(f"Warning: File not found for val_{val}")
            continue
    
    # Print the combined disease match rate table
    if all_results:
        print_combined_match_rate_table(all_results)