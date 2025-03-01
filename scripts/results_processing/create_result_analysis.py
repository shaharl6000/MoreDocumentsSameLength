
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import seaborn as sns
from openai import OpenAI
import tiktoken
import os
import json


def concatenate_columns(df):
    # Select columns that don't include 'f1' or 'manual'
    columns_to_concatenate = [col for col in df.columns if 'f1' not in col and 'manual' not in col]
    
    # Concatenate the selected columns into a single series
    concatenated_series = df[columns_to_concatenate].astype(str).agg(' '.join, axis=1)
    
    return concatenated_series


# Function to calculate statistics for numeric columns
def calculate_numeric_stats(column, metric):
    return {
        f'{metric}_mean': column.mean(),
        f'{metric}_median': column.median(),
        f'{metric}_variance': column.var(),
        f'{metric}_25th_percentile': column.quantile(0.25),
        f'{metric}_75th_percentile': column.quantile(0.75),
        f'{metric}_mode_value': column.mode().iloc[0] if not column.mode().empty else np.nan,
        f'{metric}_range': column.max() - column.min(),
        f'{metric}_skewness': column.skew(),
        f'{metric}_kurtosis': column.kurt(),
        f'{metric}_interquartile_range': column.quantile(0.75) - column.quantile(0.25)   ,     
        f'{metric}_number_of_exact_ones': (column == 1).sum(),
        f'{metric}_number_above_threshold_0.75': (column > 0.75).sum(),
        f'{metric}_number_below_threshold_0.25': (column < 0.25).sum()
        }

# Function to calculate statistics for text columns
def calculate_string_stats(column):
    lengths = column.dropna().apply(len)
    word_counts = column.dropna().apply(lambda x: len(x.split()))
    return {
        'pred_total_length': lengths.sum(),
        'pred_mean_length': lengths.mean(),
        'pred_std_length': lengths.std(),
        'pred_25th_percentile_length': lengths.quantile(0.25),
        'pred_median_length': lengths.median(),
        'pred_75th_percentile_length': lengths.quantile(0.75),
        'pred_mean_word_count': word_counts.mean()
    }

def column_statistics(df):
    excluded_columns = ["id", "Unnamed: 0", "highlevel_group", "prefix_group", "postfix_group"]
    stats = {}

    columns_to_process = [col for col in df.columns if col not in excluded_columns]

    for column in columns_to_process:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats[column] = calculate_numeric_stats(df[column].dropna(),"f1")
        else:
            stats[column] = calculate_string_stats(df[column].dropna())

    return stats


def save_stats_to_csv(stats, filepath):
    # Create a DataFrame from the stats dictionary
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(filepath, index=False, index_label='Statistic')
    

def break_to_question_types(df):
    print(df.columns)
    df["highlevel_group"] = df["Unnamed: 0"].str.split("_").str[0]
    
    df["prefix_group"] = df["Unnamed: 0"].str.split("_").str[0].apply(lambda x: x.split("hop")[0])
    
    df["postfix_group"] = df["Unnamed: 0"].str.split("_").str[0].apply(lambda x: x.split("hop")[1] if "hop" in x else "missing")

    return df


def check_and_plot_correlation(stats_df, directory, threshold=0.85):
    stats_df = extract_filtred_df(stats_df)
    stats_df = stats_df.T 

    # Ensure 'mean_f1' is in the DataFrame
    if 'f1_mean' not in stats_df.columns:
        print("f1_mean not found in DataFrame columns")
        return

    # Compute correlations with 'mean_f1'
    correlation_vector = stats_df.corr("pearson")['f1_mean'].drop('f1_mean')

    # Convert to DataFrame for plotting
    correlation_df = correlation_vector.to_frame().T

    # Plot the heatmap
    plt.figure(figsize=(5, 10))
    sns.heatmap(correlation_df.T, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black', cbar=False)
    plt.title('Correlation with f1_mean')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'correlation_with_f1_mean.png'))
    plt.close()
    print("Heatmap saved in directory")


def save_string_column_plots(stats_df, directory, phrase = "rephrased"):
    filtered_df = extract_filtred_df(stats_df, phrase)
    for index, row in filtered_df.iterrows():
        ax = row.plot(kind='barh', legend=False, figsize=(10, 5))
        plt.title(f'Statistics for {index}')
        plt.xlabel('Value')
        plt.ylabel('Column')
        plt.tight_layout()
        # Annotate bars with the values
        # for container in ax.containers:
        #     ax.bar_label(container)
        plt.savefig(os.path.join(directory, f'{index}_{phrase}_statistics.png'))
        plt.close()

def extract_filtred_df(stats_df, phrase = "Gemma"):
    gemma_columns = stats_df.columns[stats_df.columns.str.contains(phrase)]
    index_col = stats_df.columns[stats_df.columns.str.startswith("index")]
    
    # Concatenate gemma columns and index column
    filtered_df = pd.concat([stats_df[gemma_columns], stats_df[index_col]], axis=1)
    
    # Set index to 'index' column if exists
    if 'index' in filtered_df.columns:
        filtered_df.set_index('index', inplace=True)
    return filtered_df


# Function to calculate statistics for each column
def column_statistics_group(group):
    excluded_columns = ["id", "Unnamed: 0", "highlevel_group", "prefix_group", "postfix_group"]
    stats = {}

    columns_to_process = [col for col in group.columns if col not in excluded_columns]

    for column in columns_to_process:
        if pd.api.types.is_numeric_dtype(group[column]):
            stats[column] = calculate_numeric_stats(group[column].dropna(), "f1")
        else:
            stats[column] = calculate_string_stats(group[column])


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# Assuming you have uploaded the CSV file, replace the path below with the correct file path
csv_file_path = "/mnt/data/sample_f1_data.csv"  # Example path; change this to your actual file path


from collections import defaultdict
def calculate_grouped_numeric_means(df, groupby_columns, exclude_columns=['Unnamed: 0']):
    # Ensure the columns to group by are not included in the columns to calculate the mean
    exclude_columns = set(exclude_columns + groupby_columns + [col for col in df.columns if "group" in col])

    # Group the DataFrame by the specified columns
    df_grouped = df.groupby(groupby_columns)

    # Select columns to calculate the mean, excluding the specified ones
    mean_columns = [col for col in df.columns if col not in exclude_columns]
    print(f"Creating means for {mean_columns}")
    
    # Calculate the mean for the selected columns
    grouped_means = df_grouped[mean_columns].mean()
    print(f"Grouped means:\n{grouped_means}")

    # Rename the columns to indicate they are means
    grouped_means.columns = [f"{col}_mean" for col in grouped_means.columns]

    # Calculate the mean F1 score across all columns for each group and add it as a new column
    grouped_means['mean_f1_group'] = grouped_means.mean(axis=1)

    # Calculate the mean F1 score for the entire model (mean of all columns)
    mean_model_score = grouped_means.mean(axis=0)

    # Add the mean model score as a new row to the grouped means DataFrame
    grouped_means.loc['mean_model_score'] = mean_model_score

    # Return the updated grouped means DataFrame
    return grouped_means

# def calculate_grouped_numeric_means(df, groupby_columns, exclude_columns=['Unnamed: 0']):
#     # Ensure the columns to group by are not included in the columns to calculate the mean
#     exclude_columns = set(exclude_columns + groupby_columns + [col for col in df.columns if "group" in col])

#     # Group the DataFrame by the specified columns
#     df_grouped = df.groupby(groupby_columns)

#     # Select columns to calculate the mean, excluding the specified ones
#     mean_columns = [col for col in df.columns if col not in exclude_columns]
#     print(f"Creating means for {mean_columns}")
    
#     # Calculate the mean for the selected columns
#     grouped_means = df_grouped[mean_columns].mean()
#     print(f"Grouped means:\n{grouped_means}")

#     # Rename the columns to indicate they are means
#     grouped_means.columns = [f"{col}_mean" for col in grouped_means.columns]

#     # Calculate the mean of the grouped means (column mean across all groups)
#     df['mean_f1_group'] = df[mean_columns].mean(axis=1)

#     # Calculate the mean F1 score for the entire model (mean of all columns)
#     mean_model_score = df[mean_columns].mean(axis=0)

#     # Add the mean model score as a new row to the DataFrame
#     df.loc['mean_model_score'] = mean_model_score
#     df.at['mean_model_score', 'mean_f1_group'] = mean_model_score.mean()

#     # Display the resulting DataFrame
#     return df

def save_string_column_plots_grouped(stats_df, directory, phrase="rephrased"):
    # Filter the DataFrame to include only relevant columns
    filtered_df = extract_filtred_df(stats_df, "")
    
    for col in filtered_df.columns:
        # Extract corresponding columns for original and rephrased questions
        rephrased_col = f"{col}"
        original_col = f"{col}".replace("rephrasedQuestions", "")
        print(original_col, "###", rephrased_col)
        if original_col in stats_df.columns and rephrased_col in stats_df.columns:
            # Create a DataFrame with both columns for plotting
            plot_data = pd.DataFrame({
                'Original': stats_df[original_col],
                'Rephrased': stats_df[rephrased_col]
            })

            # Plot the data
            ax = plot_data.plot(kind='barh', figsize=(12, 6), width=0.8)
            plt.title(f'Statistics for {col}')
            plt.xlabel('Value')
            plt.ylabel('Statistic')
            plt.tight_layout()

            # Annotate bars with the values
            for container in ax.containers:
                ax.bar_label(container)

            # Save the plot
            plt.savefig(os.path.join(directory, f'{col}_{phrase}_statistics_comparison.png'))
            plt.close()

PATH_F1 = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/combined_scores.csv"
PATH_PREDICTIONS = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/csv_pred/predictions_of_seam.csv"

# def create_model_heatmaps_v3(df_results, model_name, measured_feature_name, saving_directory_path):
#     # Filter columns based on the model name
#     relevant_columns = [col for col in df_results.columns if model_name.lower() in col.lower()]
    
#     # Filter the DataFrame to include only relevant columns
#     plotted_df = df_results[relevant_columns]
    
#     # Identify rephrased and non-rephrased columns
#     rephrased_columns = [col for col in plotted_df.columns if 'rephrasedQuestions' in col]
#     non_rephrased_columns = [col for col in plotted_df.columns if col not in rephrased_columns and 
#                              col not in ["id", "Unnamed: 0"] and "group" not in col]

#     # Replace specific words from the column names
#     df_rephrased = plotted_df[rephrased_columns].rename(columns=lambda x: x.replace('f1_mean', '').replace('rephrasedQuestions', '').replace(model_name.lower(), '').strip('_'))
#     df_non_rephrased = plotted_df[non_rephrased_columns].rename(columns=lambda x: x.replace('f1_mean', '').replace('rephrasedQuestions', '').replace(model_name.lower(), '').strip('_'))
#     print(f"rephrased has {df_rephrased.columns} columns and non rephrased has {df_non_rephrased.columns} ")
#     # Transpose the dataframes to have models as rows and metrics as columns
#     df_rephrased_transposed = df_rephrased.T
#     df_non_rephrased_transposed = df_non_rephrased.T

#     # Setting up the plot area
#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))

#     # Rephrased heatmap
#     sns.heatmap(df_rephrased_transposed, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, ax=ax1)
#     ax1.set_title(f"{measured_feature_name.capitalize()} Scores for Rephrased {model_name.capitalize()} Models")
#     ax1.set_xlabel(f"{measured_feature_name}")
#     ax1.set_ylabel("Rephrased Models")

#     # Non-Rephrased heatmap
#     sns.heatmap(df_non_rephrased_transposed, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, ax=ax2)
#     ax2.set_title(f"{measured_feature_name.capitalize()} Scores for Non-Rephrased {model_name.capitalize()} Models")
#     ax2.set_xlabel(f"{measured_feature_name}")
#     ax2.set_ylabel("Non-Rephrased Models")
#     saving_path = f"{saving_directory_path}/dual_heatmap_{model_name}_{measured_feature_name}.png"
    
#     plt.tight_layout()
#     plt.savefig(saving_path)
def create_model_heatmaps_v3(df_results, model_name, measured_feature_name, saving_directory_path):
    # Filter columns based on the model name
    if measured_feature_name == 'Statistics':
        extra_col = ["index"]
    else :
        extra_col = []
    
    # Set the index column
    relevant_columns = extra_col +[col for col in df_results.columns if model_name.lower() in col.lower()] 
    
    # Filter the DataFrame to include only relevant columns
    plotted_df = df_results.loc[:, relevant_columns]
    # if "index" in plotted_df.columns:
    #     plotted_df = plotted_df.set_index("index")

    # Identify rephrased and non-rephrased columns
    rephrased_columns = [col for col in plotted_df.columns if 'rephrasedQuestions' in col]
    non_rephrased_columns = [col for col in plotted_df.columns if col not in rephrased_columns and 
                             col not in ["id", "Unnamed: 0"] and "group" not in col] 
    
    # Replace specific words from the column names
    df_rephrased = plotted_df[rephrased_columns].rename(columns=lambda x: x.replace('f1_mean', '').replace('rephrasedQuestions', '').replace(model_name.lower(), '').strip('_'))
    df_non_rephrased = plotted_df[non_rephrased_columns].rename(columns=lambda x: x.replace('f1_mean', '').replace('rephrasedQuestions', '').replace(model_name.lower(), '').strip('_'))
    
    # Check if there are any columns in the rephrased and non-rephrased dataframes
    if len(df_rephrased.columns) == 0 and len(df_non_rephrased.columns) == 0:
        print(f"No relevant columns found for model '{model_name}' and measured feature '{measured_feature_name}'.")
        return
        
    # Transpose the dataframes to have models as rows and metrics as columns
    if len(df_rephrased.columns) > 0:
        df_rephrased_transposed = df_rephrased.T
        # if "index" in df_rephrased.columns:
        #     df_rephrased_transposed = df_rephrased_transposed.reset_index()
        df_rephrased_transposed = df_rephrased_transposed.fillna(0)
    else:
        df_rephrased_transposed = pd.DataFrame()
    
    if len(df_non_rephrased.columns) > 0:
        df_non_rephrased_transposed = df_non_rephrased.T
        # if "index" in df_non_rephrased.columns:
        #     df_non_rephrased_transposed = df_non_rephrased_transposed.reset_index()
        df_non_rephrased_transposed = df_non_rephrased_transposed.fillna(0)
    else:
        df_non_rephrased_transposed = pd.DataFrame()

    # Setting up the plot area
    if measured_feature_name == "Statistics":
        x_axis_namez = plotted_df.loc[:, "index"]
    else:
        x_axis_namez = df_rephrased_transposed.columns
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    
    # Rephrased heatmap
    if len(df_rephrased_transposed.columns) > 0:
        sns.heatmap(df_rephrased_transposed, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, ax=ax1)
        ax1.set_title(f"{measured_feature_name.capitalize()} Scores for Rephrased {model_name.capitalize()} Models")
        ax1.set_xlabel(f"{measured_feature_name}")
        ax1.set_ylabel("Rephrased Models")
        ax1.set_xticklabels(x_axis_namez, rotation=90, fontsize=8)
        ax1.set_yticklabels(df_rephrased_transposed.index, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No rephrased models found.", ha="center", va="center", transform=ax1.transAxes)
    
    # Non-Rephrased heatmap
    if len(df_non_rephrased_transposed.columns) > 0:
        sns.heatmap(df_non_rephrased_transposed, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, ax=ax2)
        ax2.set_title(f"{measured_feature_name.capitalize()} Scores for Non-Rephrased {model_name.capitalize()} Models")
        ax2.set_xlabel(f"{measured_feature_name}")
        ax2.set_ylabel("Non-Rephrased Models")
        ax2.set_xticklabels(x_axis_namez, rotation=90, fontsize=8)
        ax2.set_yticklabels(df_non_rephrased_transposed.index, fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No non-rephrased models found.", ha="center", va="center", transform=ax2.transAxes)
    
    saving_path = f"{saving_directory_path}/dual_heatmap_{model_name}_{measured_feature_name}.png"
    
    plt.tight_layout()
    plt.savefig(saving_path)
    print("$$ SAVED $$")

if __name__ == "__main__":
    df_f1 = pd.read_csv(PATH_F1)
    df_results = pd.read_csv(PATH_PREDICTIONS)

    df_f1_stats = column_statistics(df_f1)
    df_results_stats = column_statistics(df_results)

    save_stats_to_csv(df_f1_stats, "df_f1_stats.csv")
    save_stats_to_csv(df_results_stats, "df_results_stats.csv")
    
    df_f1_stats = pd.DataFrame(df_f1_stats)
    df_f1_stats.columns = df_f1_stats.columns.str.replace('_f1', '')
    df_results_stats = pd.DataFrame(df_results_stats)

    common_columns = df_f1_stats.columns.intersection(df_results_stats.columns)
    
    # Concatenate the two DataFrames along the columns
    result_df = pd.concat([df_f1_stats, df_results_stats], ignore_index=False)

    # Reset index to have statistics names as a column
    result_df.reset_index(inplace=True)
    result_df = result_df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

    ##### RESTOREEEE

    # Example usage
    result_df.to_csv('combined_results.csv', index=False)
    # string_columns = [col for col in result_df.columns if col not in ["id", "Unnamed", "highlevel_group", "prefix_group", "postfix_group"]]
    # save_string_column_plots(result_df, r"/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/plots/bar_plots/")
    # save_string_column_plots_grouped(result_df, r"/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/plots/grouped_bars/")
    # check_and_plot_correlation(result_df, r"/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/plots/correlation/")

    df = break_to_question_types(df_f1)
    print(df.columns)
    grouped_stats = calculate_grouped_numeric_means(df, [
        'highlevel_group'] ) #,"prefix_group", "postfix_group"])
    
    grouped_stats.to_csv('/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/grouped_f1_stats.csv', index=True, index_label=grouped_stats.index.name)


    # Apply the function to the `df_results` DataFrame for each model in models
    
    models = ["gemma", "llama", "mistral"]
    print(f"{df_f1_stats.columns} columns of f1 stats")

    print(f"{grouped_stats.columns} columns of grouped")
    for model in models:
        create_model_heatmaps_v3(grouped_stats, model, 'highlevel_group',"/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/plots/heatmap_pairs")
        index_values_to_select = ["f1_mean", "f1_variance", "f1_number_above_threshold_0.75", "f1_number_below_threshold_0.25", "f1_number_of_exact_ones"]

    # Use .loc to filter the rows and columns based on index and column values
    selected_index_features = result_df["index"].isin(index_values_to_select)
    filtered_stats_df = result_df.loc[selected_index_features, :]

    create_model_heatmaps_v3(filtered_stats_df, model, 'Statistics', "/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/plots/heatmap_pairs/")