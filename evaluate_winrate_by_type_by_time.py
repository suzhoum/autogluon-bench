import pandas as pd
import numpy as np

# Problem types for each task
problem_types = {
  'nyc-albert': 'binary', 'nyc-dilbert': 'binary', 'nyc-dionis': 'binary', 'nyc-mlg-ulb-creditcardfraud': 'binary', 'nyc-p53mutants': 'binary', 'nyc-record2vec_plugs': 'binary', 'nyc-robert': 'binary', 'op100-168338': 'binary', 'large-codes-5gb': 'multiclass', 'large-criteo-kaggle-5gb': 'binary', 'openml-3020': 'binary', 'aa-alster-books-nov21': 'multiclass', 'aa-record2vec-plugs': 'multiclass', 'nyc-wendykan-lending-club-loan-data': 'multiclass', 'op100-146825': 'multiclass', 'openml-167121': 'multiclass', 'openml-167130': 'multiclass', 'openml-3573': 'multiclass', 'fmnist-5gb': 'multiclass', 'aa-burakhmmtgl-energy-molecule': 'regression', 'aa-leaderboard278': 'regression', 'aa-leaderboard343': 'regression', 'large-big-claims-severity-5': 'regression', 'large-reddit-score-5gb': 'regression', 'comp-mercari-price-suggestion-challenge': 'regression', 'big-deloitte-ml-5': 'regression', 'large-taxifare-5gb': 'regression', 'large-lendingc-5gb': 'binary'
}

time_limits_files = {
    '15m': "results_automlbenchmark_48c15m_canvas_20240311_processed.csv",
    '30m': "results_automlbenchmark_48c30m_canvas_20240311_processed.csv",
    '45m': "results_automlbenchmark_48c45m_canvas_20240311_processed.csv",
    '60m': "results_automlbenchmark_48c60m_canvas_20240311_processed.csv",
    '90m': "results_automlbenchmark_48c90m_canvas_20240311_processed.csv",
    '120m': "results_automlbenchmark_48c120m_canvas_20240311_processed.csv",
}

def determine_outcome(framework_result, baseline_result):
    if pd.isna(framework_result) and not pd.isna(baseline_result):
        return 'loss'
    elif not pd.isna(framework_result) and framework_result > 0 and (pd.isna(baseline_result) or framework_result > baseline_result):
        return 'win'
    elif (pd.isna(framework_result) and pd.isna(baseline_result)) or (not pd.isna(framework_result) and not pd.isna(baseline_result) and framework_result == baseline_result) or (not pd.isna(framework_result) and pd.isna(baseline_result) and framework_result <= 0):
        return 'tie'
    else:
        return 'loss'


interested_frameworks = ['AutoGluon_hq_ds', 'AutoGluon_bq_ds']  # Example frameworks
baseline_df = pd.read_csv("canvas_hpo_results.csv")

# Initialize DataFrame to store win rate results
win_rate_results = pd.DataFrame(columns=['Time Limit', 'Framework', 'Problem Type', 'Win Rate'])

dfs = []

for time_limit, file_path in time_limits_files.items():
    results_df = pd.read_csv(file_path)
    
    # Assuming 'task' column exists in baseline_df for merging
    results_df = results_df.merge(baseline_df[['task', 'canvas_hpo_result']], on='task', how='left')
    
    # Map tasks to their problem types
    results_df['Problem Type'] = results_df['task'].map(problem_types)
    
    for framework in interested_frameworks:
        framework_result_col = f"{framework}_result"
        # Filter out rows where the specific framework's result column does not exist
        if framework_result_col in results_df.columns:
            for problem_type in results_df['Problem Type'].unique():
                filtered_df = results_df[results_df['Problem Type'] == problem_type]
                win_count = 0
                total_count = 0

                for _, row in filtered_df.iterrows():
                    framework_result = row[framework_result_col]
                    baseline_result = row['canvas_hpo_result']
                    
                    outcome = determine_outcome(framework_result, baseline_result)
                    
                    if outcome == 'win':
                        win_count += 1
                    if outcome in ['win', 'loss']:  # Count valid comparisons
                        total_count += 1

                # Calculate win rate for the current time limit, framework, and problem type
                win_rate = win_count / total_count if total_count > 0 else np.nan
                
                # Append results to the list as a DataFrame
                dfs.append(pd.DataFrame({
                    'Time Limit': [time_limit],
                    'Framework': [framework],
                    'Problem Type': [problem_type],
                    'Win Rate': [win_rate]
                }))

# Concatenate all DataFrames in the list to form the final win_rate_results DataFrame
win_rate_results = pd.concat(dfs, ignore_index=True)

# Print or save win_rate_results as needed
print(win_rate_results)
