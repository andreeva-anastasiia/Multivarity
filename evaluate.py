import argparse
import data_generation.generators as generators
import datasets as ds
from data_evaluation.metrics import autogluon_performance, chronos_performance, statistical_similarity
import pandas as pd
import numpy as np
import os 

os.environ["HF_TOKEN"] = "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"


def parse_args():
    parser = argparse.ArgumentParser(description="Script to evaluate the Data Augmentation/Synthesis Method")
    parser.add_argument('--data_generation_method', type=str, required=True, help='Name of the method to evaluate')
    parser.add_argument('--is_synthesis', action='store_true', help='Flag to indicate if the method is a Synthesis Method')
    parser.add_argument('--data_path', type=str, default='ddrg/time-series-datasets', help='Name of Huggingface DS to use as input for DataAugmentation Methods')
    parser.add_argument('--output', type=str, default='output', help='Output file to save the results')
    return parser.parse_args()



def preprocess_test_data(test_data):
    """
    Extract and preprocess the 'value' field from test_data to align with generated_data structure.

    Args:
        test_data (pd.Series): Original test data with nested structure.

    Returns:
        pd.DataFrame: A flattened DataFrame similar to generated_data.
    """
    # Extract the 'value' field (which is a list of lists)
    nested_values = test_data['value']  # Assuming 'value' contains the numerical data

    # Flatten the nested list into a single list
    flattened_values = [item for sublist in nested_values for item in sublist]

    # Create a DataFrame with a similar structure to generated_data
    processed_data = pd.DataFrame({
        0: flattened_values,  # Values column
        'time': range(len(flattened_values))  # Time index
    })

    return processed_data




if __name__ == "__main__":
    args = parse_args()


    # TODO Hardcode Test Data to keep it constant btwn Experiments to make resutls comparable
    real_world_data = ds.load_dataset(args.data_path, 'UNIVARIATE', trust_remote_code=True)
    test_set = real_world_data["train"][2:4] # TODO Remove 2:4 for final eval (when more data becomes available)
    input_data = real_world_data["train"].to_pandas()[:2] # TODO Remove 0:2 for final eval (when more data becomes available)

    # EDIT
    input_data = preprocess_test_data(input_data)
    # EDIT

    # Load Method via name from File
    generator = getattr(generators, args.data_generation_method)() # TODO Pass Keyword arguments if necessary e.g. loading them from json and then **config
    # real_world_data["train"][0]#["value"][0]

    # # print(input_data)
    print(type(real_world_data))
    print(real_world_data)
    #
    # # Check the structure of the dataset (especially the 'train' set)
    # print('Check the structure of the dataset')
    # print(input_data['value'])


    generated_time_series = []
    intermediate_results = []

    for idx, time_series in input_data.iterrows():

        # Extract numerical values by converting 'value' column to DataFrame
        # time_series_data = pd.DataFrame({'value': time_series['value']})

        # Evaluate the Method
        # generated_data = generator(time_series_data)
        generated_data = generator(time_series)

        # Add timestamp column
        generated_data['time'] = range(len(generated_data))

        # Save the generated data to CSV
        output_file_path = os.path.join(args.output, f"generated_time_series.csv")
        generated_data.to_csv(output_file_path, index=False)  # Save to CSV without the index column


        if not args.is_synthesis:
            # Analyze the Generated Data  on one to one metrics
            stat_sim_scores = statistical_similarity(generated_data=generated_data, test_data=time_series)
            intermediate_results.append(stat_sim_scores)

    # aggregate intermediate results TODO Check if this makes sense (Goal: Average each metric over all generated time series)
    results = {key: sum([r[key] for r in intermediate_results] / len(intermediate_results)) for key in intermediate_results[0].keys()}





    # Add Gloabl Metrics (Metrics that can generalize over several Time Series)
    train_set = pd.concat([input_data, generated_data], axis=0)

    # Analyze Generated Data 
    # Does it make AutoGluon better
    ag_score = autogluon_performance(train_set, test_set)

    # Does it make Chronos better
    # TODO (In December): chronos_score = chronos_performance(train_set, test_set)
    chronos_score = 0 # Placeholder





    # TODO Save Results to File (Method_name, ag_score, chronos_score, input_data_path, test_data_path)



    # Save the results (global metrics) to a CSV file
    results_file_path = os.path.join(args.output, "evaluation_results.csv")
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_file_path, index=False)

    print(f"Generated time series saved in: {args.output}")
    print(f"Global evaluation results saved in: {results_file_path}")