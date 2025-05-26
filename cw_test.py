import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime # For timestamping output files/dirs

def generate_projection_matrix(original_dim, projection_dim, device):
    """
    Generates a single Gaussian random projection matrix.

    Args:
        original_dim (int): The original dimension of the data.
        projection_dim (int): The dimension to project the data to.
        device (torch.device or str): The device to create the matrix on.

    Returns:
        torch.Tensor: The projection matrix of shape (original_dim, projection_dim),
                      dtype torch.float32, on the specified device.
    """
    scale = 1.0 / np.sqrt(projection_dim)
    # Sample from N(0,1)
    matrix = torch.randn(original_dim, projection_dim) 
    # Scale
    matrix = matrix * scale
    return matrix.to(device=device, dtype=torch.float32)


def d_cw_squared_X_N0I_vectorized(X_std, D, gamma_val, D_approx_threshold):
    """
    Placeholder for d_cw_squared_X_N0I_vectorized.
    This will be properly implemented in a subsequent task.
    """
    # For now, let's return a dummy value, e.g., the mean of squared values
    # to have some computation.
    if X_std.ndim == 1:
        X_std = X_std.unsqueeze(0) # ensure 2D
    # Ensure D is used in some way, even if trivially for the placeholder
    return torch.mean(X_std[:, :D]**2).item() 


def test_multivariate_normality_cw(
    X_observed: torch.Tensor, 
    n_mc_samples: int = 1000, 
    D_approx_threshold: int = 20, 
    projection_dim: int = None, 
    device: str = 'cpu', 
    random_seed: int = None
):
    """
    Performs a Cramér-von Mises test for multivariate normality,
    with an optional random projection step.

    Args:
        X_observed (torch.Tensor): The observed data (n_samples, D_features).
        n_mc_samples (int): Number of Monte Carlo samples for the null distribution.
        D_approx_threshold (int): Threshold for dimension D to switch between exact and approx psi_D.
                                  (Used in d_cw_squared_X_N0I_vectorized)
        projection_dim (int, optional): If specified and valid (0 < projection_dim < D_obs_original),
                                        data will be projected to this dimension. Defaults to None.
        device (str): Device to run computations on ('cpu' or 'cuda').
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        tuple: (cw_stat_observed, p_value, null_distribution_mc)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed) # numpy is used by generate_projection_matrix

    X_observed = X_observed.to(device=device, dtype=torch.float32)
    n_samples_obs, D_obs = X_observed.shape

    # Standardize observed data
    mean_obs = torch.mean(X_observed, dim=0)
    std_obs = torch.std(X_observed, dim=0)
    std_obs[std_obs == 0] = 1.0 # Avoid division by zero for dimensions with zero variance
    X_std_observed = (X_observed - mean_obs) / std_obs

    # --- Projection Step ---
    D_obs_original = D_obs
    projection_matrix = None

    # Check if projection_dim is valid for projection
    if projection_dim is not None and projection_dim > 0 and projection_dim < D_obs_original:
        print(f"Projection active: original_dim={D_obs_original}, projection_dim={projection_dim}, device={device}")
        projection_matrix = generate_projection_matrix(D_obs_original, projection_dim, device=device)
        X_std_observed = torch.matmul(X_std_observed, projection_matrix)
        D_obs = projection_dim # Update D_obs to the new, projected dimension
    else:
        if projection_dim is not None:
            print(f"Projection_dim ({projection_dim}) is not valid for original_dim={D_obs_original}. Projection skipped.")
        # D_obs remains D_obs_original
        pass


    # Placeholder for gamma_val calculation.
    # The actual calculation for gamma_val (e.g. from psi_D_mc) is expected to be
    # part of d_cw_squared_X_N0I_vectorized or a helper it calls.
    # For this function's scope, we pass it along.
    # Let's assume gamma_val is determined within d_cw_squared_X_N0I_vectorized or a fixed constant for now.
    # Based on the next subtask, gamma_val is related to psi_D_mc which is complex.
    # The subtask asks to *pass* D_approx_threshold, not calculate gamma here.
    # Let's assume gamma_val should be determined within d_cw_squared_X_N0I_vectorized
    # or it's a parameter that should be passed to this test function if it's pre-calculated.
    # For now, as per instructions for d_cw_squared_X_N0I_vectorized, it takes gamma_val.
    # We will set a placeholder value here.
    gamma_val = 1.0 # Placeholder, as its calculation is not part of this subtask.

    # Calculate observed statistic using the (potentially projected) data and (potentially updated) D_obs
    cw_stat_observed = d_cw_squared_X_N0I_vectorized(X_std_observed, D_obs, gamma_val, D_approx_threshold)

    # Monte Carlo simulation for null distribution
    null_distribution_mc = torch.empty(n_mc_samples, device=device, dtype=torch.float32)

    for i in range(n_mc_samples):
        # Generate data from N(0, I) using the *original* dimension
        X_null_single_dataset = torch.randn(n_samples_obs, D_obs_original, device=device, dtype=torch.float32)
        
        # Standardize the null data (conceptually, already N(0,I), but finite samples will have non-exact mean/std)
        mean_null = torch.mean(X_null_single_dataset, dim=0)
        std_null = torch.std(X_null_single_dataset, dim=0)
        std_null[std_null == 0] = 1.0 
        X_null_std = (X_null_single_dataset - mean_null) / std_null
        
        # Apply the *same* projection if it was used for the observed data
        if projection_matrix is not None:
            X_null_std = torch.matmul(X_null_std, projection_matrix)
        
        # Calculate null statistic using the (potentially projected) null data and (potentially updated) D_obs
        null_stat = d_cw_squared_X_N0I_vectorized(X_null_std, D_obs, gamma_val, D_approx_threshold)
        null_distribution_mc[i] = null_stat

    # Calculate p-value
    p_value = torch.mean((null_distribution_mc >= cw_stat_observed).float()).item()

    return cw_stat_observed, p_value, null_distribution_mc


def run_simulation_experiment(
    n_samples_list: list, 
    d_features_list: list, 
    n_reps: int,
    n_mc_samples_test: int = 1000, 
    D_approx_threshold_test: int = 20, 
    projection_dim_config: bool = False, 
    device: str = 'cpu', 
    random_seed: int = None
):
    """
    Runs a simulation experiment for the multivariate normality test.

    Args:
        n_samples_list (list): List of sample sizes to test.
        d_features_list (list): List of feature dimensions to test.
        n_reps (int): Number of repetitions for each configuration.
        n_mc_samples_test (int): Number of MC samples for the normality test itself.
        D_approx_threshold_test (int): D_approx_threshold for the normality test.
        projection_dim_config (bool): If True, dynamic projection dimension is calculated and used.
        device (str): Device for computations ('cpu', 'cuda').
        random_seed (int, optional): Global random seed for the experiment.

    Returns:
        list: A list of dictionaries, each containing results for a configuration.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    results = []

    for n_samples in n_samples_list:
        for d_features in d_features_list:
            
            d_proj_to_pass = None
            log_msg_proj_val = "" # For logging the projection dimension value

            if projection_dim_config:
                if d_features <= 0: 
                    # This case should ideally not be hit if d_features_list contains valid dimensions.
                    d_proj_to_pass = None 
                elif d_features == 1:
                    d_proj_to_pass = 1
                else: # d_features > 1
                    # Formula: min(D_current, max(10, int(D_current * 0.1) + 5))
                    # Ensure calc_proj_dim inside max is at least 1
                    calc_proj_dim_inner = int(d_features * 0.1) + 5
                    d_proj_to_pass = min(d_features, max(10, calc_proj_dim_inner))
                    
                    # Final check to ensure d_proj_to_pass is at least 1 (already handled by logic for d_features >= 1)
                    if d_proj_to_pass < 1: # Defensive, should not be strictly necessary if d_features >= 1
                        d_proj_to_pass = 1
                
                if d_proj_to_pass is not None:
                    log_msg_proj_val = f", Proj. d={d_proj_to_pass}"

            print(f"Starting simulation: n={n_samples}, d={d_features}{log_msg_proj_val}, reps={n_reps}, device={device}")
            
            p_values_current_config = []
            for rep_idx in range(n_reps):
                # Generate sample data (standard normal for H0 true)
                X_data = torch.randn(n_samples, d_features, device=device, dtype=torch.float32)

                current_iter_seed = None # Allow test to run with its own internal randomness unless specific seed per rep is needed
                if random_seed is not None:
                    # If a global seed is set, we might want to ensure each rep is still pseudo-random
                    # but different from others. Or fix it if that's the goal.
                    # For now, test_multivariate_normality_cw handles its own seed for MC samples if its random_seed arg is set.
                    # Here, we are controlling the seed for X_data generation.
                    # If test_multivariate_normality_cw needs a seed for its MC part that varies per rep:
                    # current_iter_seed = random_seed + rep_idx 
                    pass


                _, p_value, _ = test_multivariate_normality_cw(
                    X_observed=X_data,
                    n_mc_samples=n_mc_samples_test,
                    D_approx_threshold=D_approx_threshold_test,
                    projection_dim=d_proj_to_pass, 
                    device=device,
                    random_seed=current_iter_seed # Pass None or a derived seed
                )
                p_values_current_config.append(p_value)
            
            alpha = 0.05 # Significance level
            type_I_error_rate = np.mean(np.array(p_values_current_config) < alpha)
            
            results.append({
                'n_samples': n_samples,
                'd_features': d_features,
                'projection_active_config': projection_dim_config, # whether calculation was enabled
                'd_proj_passed': d_proj_to_pass, # actual dimension passed
                'p_values': p_values_current_config, # Storing all p-values can be memory intensive
                'type_I_error_rate': type_I_error_rate
            })
            print(f"Finished: n={n_samples}, d={d_features}{log_msg_proj_val} -> Type I Error Rate: {type_I_error_rate:.4f}")
            
    return results


def compare_dimensions(
    n_samples_list: list, 
    d_features_list: list, 
    n_reps: int,
    tasks_to_run_comparison: list, # e.g., list of strings identifying simulation variants
    output_dir_plot_path_str: str, # Directory for saving plots and Excel from plot_results
    n_mc_samples_test: int = 1000,
    D_approx_threshold_test: int = 20,
    use_projection: bool = False, 
    device: str = 'cpu', 
    random_seed: int = None
):
    """
    Compares different dimension settings or tasks using run_simulation_experiment.
    Controls whether projection is enabled for these experiments based on use_projection.

    Args:
        n_samples_list (list): List of sample sizes.
        d_features_list (list): List of feature dimensions.
        n_reps (int): Number of repetitions for each configuration.
        tasks_to_run_comparison (list): Identifiers for different tasks/scenarios.
                                        (Currently, this primarily serves as a loop iterator
                                         if data generation is fixed within run_simulation_experiment).
        n_mc_samples_test (int): MC samples for the underlying normality test.
        D_approx_threshold_test (int): D_approx_threshold for the underlying test.
        use_projection (bool): If True, projection_dim_config in run_simulation_experiment is True.
        device (str): Computation device.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A dictionary where keys are task_identifiers and values are their experiment results.
    """
    if random_seed is not None:
        # Seeding at this top level helps ensure that if tasks are run sequentially,
        # they get the same sequence of random numbers if desired.
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    projection_status_msg = "Projection Enabled" if use_projection else "Projection Disabled"
    # Example: "Starting dimension comparison for Covariance Model {cov_model} on {device} (Projection Enabled)."
    # Since cov_model is not directly a parameter here, we make a general statement.
    # If tasks_to_run_comparison implies different models, that would be printed per task.
    print(f"--- Starting Dimension Comparison (Overall Status: {projection_status_msg}) on {device} ---")

    all_task_results = {}

    for task_idx, task_identifier in enumerate(tasks_to_run_comparison):
        # If random_seed is set, ensure each task gets a different (but predictable) sequence
        # or the same if that's intended. For true independence of tasks, might derive seed:
        task_specific_seed = None
        if random_seed is not None:
            task_specific_seed = random_seed + task_idx # Simple way to vary seed per task

        print(f"\nRunning Experiments for Task: '{task_identifier}' ({projection_status_msg})")

        experiment_results_for_task = run_simulation_experiment(
            n_samples_list=n_samples_list,
            d_features_list=d_features_list,
            n_reps=n_reps,
            n_mc_samples_test=n_mc_samples_test,
            D_approx_threshold_test=D_approx_threshold_test,
            projection_dim_config=use_projection, # Key change: pass use_projection
            device=device,
            random_seed=task_specific_seed # Pass the derived or original seed
        )
        
        all_task_results[task_identifier] = experiment_results_for_task
        
        print(f"Summary for Task: '{task_identifier}'")
        for res_item in experiment_results_for_task:
            # Adding a bit more detail from res_item if needed
            proj_active_conf = res_item.get('projection_active_config', 'N/A')
            d_proj_passed = res_item.get('d_proj_passed', 'N/A')
            t1_error = res_item.get('type_I_error_rate', float('nan'))
            print(f"  n={res_item['n_samples']}, d={res_item['d_features']}, "
                  f"Config Proj={proj_active_conf}, d_proj={d_proj_passed}, "
                  f"T1Err={t1_error:.4f}")

    print("\n--- Dimension Comparison Finished ---")
    return all_task_results


def run_paper_experiments_pytorch(
    # Experiment parameters (examples, can be customized)
    n_samples_model1_list: list = [100, 200],
    d_features_model1_list: list = [10, 20, 50],
    n_reps_model1: int = 100, # Reps for model 1 comparison
    
    n_samples_detailed_exp_val: int = 150, # Fixed N for detailed analysis
    dimensions_for_detailed_analysis_exp: list = [5, 25, 75], # D values for detailed analysis
    covariance_models_to_analyze_exp: list = ["Identity", "AR1_Mod", "EquiCorr_Mod"], 
    n_reps_detailed_exp: int = 100, # Reps for detailed analysis
    
    n_mc_samples_test: int = 1000, # MC samples for the underlying test
    D_approx_threshold_test: int = 20, # D_approx_threshold for the test

    apply_projection: bool = False, 
    device: str = 'cpu', 
    random_seed: int = None
):
    """
    Runs a series of paper experiments using PyTorch, with control over projection.

    Args:
        n_samples_model1_list (list): N values for Model 1 comparison.
        d_features_model1_list (list): D values for Model 1 comparison.
        n_reps_model1 (int): Repetitions for Model 1 comparison.
        n_samples_detailed_exp_val (int): Fixed N for detailed analysis.
        dimensions_for_detailed_analysis_exp (list): D values for detailed analysis.
        covariance_models_to_analyze_exp (list): Names of covariance models for detailed analysis.
        n_reps_detailed_exp (int): Repetitions for detailed analysis.
        n_mc_samples_test (int): MC samples for the normality test.
        D_approx_threshold_test (int): D_approx_threshold for the normality test.
        apply_projection (bool): Global switch to enable/disable projection feature.
        device (str): Computation device.
        random_seed (int, optional): Main random seed for reproducibility.

    Returns:
        dict: Contains results from all parts of the experiment.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    projection_status_main_msg = "Projection Enabled" if apply_projection else "Projection Disabled"
    print(f"===== Running Paper Experiments (PyTorch) - Overall Status: {projection_status_main_msg} =====")
    print(f"Device: {device}, Main Random Seed: {random_seed}")

    all_results = {}

    # --- Part 1: Model 1 Dimension Comparison ---
    print(f"\n--- Part 1: Model 1 Dimension Comparison ({projection_status_main_msg}) ---")
    # For this example, "Model1_DefaultGaussian" is a placeholder task name.
    # compare_dimensions iterates through tasks_to_run_comparison. If Model 1 implies specific data,
    # that logic would be tied to this task name or handled inside compare_dimensions/run_simulation_experiment.
    tasks_model1 = ["Model1_DefaultGaussian"] # Example task
    
    # Create a unique directory for this experiment run using a timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiment_results_{timestamp_str}"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Log experiment parameters
    params_log_dict_exp = {
        "Run Timestamp": timestamp_str,
        "Device": device,
        "Global Random Seed": str(random_seed) if random_seed is not None else "N/A",
        "Projection Status": str(apply_projection),
        "N MC Samples (Test)": str(n_mc_samples_test),
        "D Approx Threshold (Test)": str(D_approx_threshold_test),
        # Parameters for Model 1 comparison
        "Model1 N Samples": str(n_samples_model1_list),
        "Model1 D Features": str(d_features_model1_list),
        "Model1 N Reps": str(n_reps_model1),
        # Parameters for Detailed Analysis
        "Detailed N Samples": str(n_samples_detailed_exp_val),
        "Detailed D Features": str(dimensions_for_detailed_analysis_exp),
        "Detailed Covariance Models": str(covariance_models_to_analyze_exp),
        "Detailed N Reps": str(n_reps_detailed_exp),
    }
    if apply_projection:
        params_log_dict_exp["Projection Dimension Logic"] = "Dynamic (min(D, max(10, int(D*0.1)+5)))"

    try:
        params_df = pd.DataFrame(list(params_log_dict_exp.items()), columns=['Parameter', 'Value'])
        params_log_path = os.path.join(base_output_dir, "pytorch_experiment_parameters.csv")
        params_df.to_csv(params_log_path, index=False)
        print(f"Saved experiment parameters to {params_log_path}")
    except Exception as e:
        print(f"Error saving experiment parameters: {e}")

    output_dir_part1 = os.path.join(base_output_dir, "part1_model_comparison")
    if not os.path.exists(output_dir_part1):
        os.makedirs(output_dir_part1)
        
    seed_part1 = random_seed + 1 if random_seed is not None else None
    
    model1_dimension_comparison_results = compare_dimensions(
        n_samples_list=n_samples_model1_list,
        d_features_list=d_features_model1_list,
        n_reps=n_reps_model1,
        tasks_to_run_comparison=tasks_model1,
        output_dir_plot_path_str=output_dir_part1, # Pass the path for part 1 results
        n_mc_samples_test=n_mc_samples_test,
        D_approx_threshold_test=D_approx_threshold_test,
        use_projection=apply_projection, # Pass apply_projection to compare_dimensions
        device=device,
        random_seed=seed_part1
    )
    all_results["model1_dimension_comparison"] = model1_dimension_comparison_results

    # --- Part 2: Detailed Analysis for Fixed n, d across Covariance Models ---
    print(f"\n--- Part 2: Detailed Analysis for Fixed N={n_samples_detailed_exp_val} ({projection_status_main_msg}) ---")
    
    output_dir_part2 = os.path.join(base_output_dir, "part2_detailed_analysis")
    if not os.path.exists(output_dir_part2):
        os.makedirs(output_dir_part2)

    detailed_analysis_results = {}
    seed_part2_base = random_seed + 2 if random_seed is not None else None

    for d_idx, d_val_detailed_exp in enumerate(dimensions_for_detailed_analysis_exp):
        print(f"\n  Analyzing for D = {d_val_detailed_exp}")
        detailed_analysis_results_for_d = {}
        for cov_model_idx, model_name_detail_exp in enumerate(covariance_models_to_analyze_exp):
            
            print(f"    Covariance Model: {model_name_detail_exp}")

            projection_dim_config_detailed = apply_projection 

            current_model_seed = None
            if seed_part2_base is not None:
                 current_model_seed = seed_part2_base + d_idx * len(covariance_models_to_analyze_exp) + cov_model_idx
            
            results_list_from_sim_exp = run_simulation_experiment(
                n_samples_list=[n_samples_detailed_exp_val], 
                d_features_list=[d_val_detailed_exp],    
                n_reps=n_reps_detailed_exp,
                n_mc_samples_test=n_mc_samples_test,
                D_approx_threshold_test=D_approx_threshold_test,
                projection_dim_config=projection_dim_config_detailed, 
                device=device,
                random_seed=current_model_seed 
            )
            
            # Extract d_proj_passed from the results (it's a list with one item)
            d_proj_val_for_saving = None
            if results_list_from_sim_exp and isinstance(results_list_from_sim_exp, list) and len(results_list_from_sim_exp) > 0:
                d_proj_val_for_saving = results_list_from_sim_exp[0].get('d_proj_passed')

            # Call save_detailed_model_results
            save_detailed_model_results(
                model_name_str=model_name_detail_exp,
                fixed_n_detailed_val=n_samples_detailed_exp_val,
                d_features_val_detail=d_val_detailed_exp,
                results_detailed_list=results_list_from_sim_exp, # Pass the full list
                output_dir_path_str=output_dir_part2,
                projection_dim_used=d_proj_val_for_saving
            )
            detailed_analysis_results_for_d[model_name_detail_exp] = results_list_from_sim_exp
        detailed_analysis_results[f"D_{d_val_detailed_exp}"] = detailed_analysis_results_for_d
        
    all_results["detailed_analysis"] = detailed_analysis_results
    
    print(f"\n===== Paper Experiments Finished. Results saved in {base_output_dir} =====")
    return all_results


# Helper function for saving detailed results (Part 2 of paper experiments)
def save_detailed_model_results(
    model_name_str: str,
    fixed_n_detailed_val: int,
    d_features_val_detail: int,
    results_detailed_list: list, # This will be the list from run_simulation_experiment (usually one item)
    output_dir_path_str: str,
    projection_dim_used = None # Actual projection dimension value passed
):
    """
    Saves detailed results (p-values, stats) for a specific N, D, CovModel configuration
    and plots a histogram of p-values.
    """
    if not os.path.exists(output_dir_path_str):
        os.makedirs(output_dir_path_str)

    model_name_cleaned_detail = model_name_str.replace(" ", "_").replace("(", "").replace(")", "")
    
    # results_detailed_list is what run_simulation_experiment returns: a list of dicts.
    # For this specific call in run_paper_experiments_pytorch, it's for a single (N,D) pair,
    # so results_detailed_list will contain one dictionary.
    if not results_detailed_list:
        print(f"No results to save for {model_name_str}, N={fixed_n_detailed_val}, D={d_features_val_detail}")
        return

    res_data = results_detailed_list[0] # Get the first (and only) item
    p_values = res_data.get('p_values', [])
    # test_stats = res_data.get('test_statistics', []) # Assuming test_multivariate_normality_cw might return this

    # Prepare DataFrame
    num_reps = len(p_values)
    df_data = {
        'Repetition': list(range(1, num_reps + 1)),
        'P_Value': p_values
    }
    
    # Add Test Statistic if available (currently not explicitly returned by test_multivariate_normality_cw)
    # if test_stats and len(test_stats) == num_reps:
    #     df_data['Test_Statistic'] = test_stats

    # Handle Projection Dimension column
    proj_dim_to_report = 'N/A'
    if projection_dim_used is not None:
        if projection_dim_used == d_features_val_detail:
            proj_dim_to_report = f"{d_features_val_detail} (Original)"
        else:
            proj_dim_to_report = str(projection_dim_used)
    else: # projection_dim_used is None (config was off)
         proj_dim_to_report = f"{d_features_val_detail} (Original)"


    df_data['Projection_Dimension'] = [proj_dim_to_report] * num_reps
    df_detailed_out = pd.DataFrame(df_data)

    filename_csv = f"detailed_results_{model_name_cleaned_detail}_n{fixed_n_detailed_val}_d{d_features_val_detail}.csv"
    full_path_csv = os.path.join(output_dir_path_str, filename_csv)
    df_detailed_out.to_csv(full_path_csv, index=False)
    print(f"Saved detailed results to {full_path_csv}")

    # Plotting p-value histogram
    plt.figure(figsize=(10, 6))
    plt.hist(p_values, bins=20, edgecolor='k', alpha=0.7)
    plt.title(
        f'P-value Distribution for {model_name_str}\n'
        f'(N={fixed_n_detailed_val}, Original D={d_features_val_detail}, Reported D={proj_dim_to_report})'
    )
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    filename_plot = f"pvalue_hist_{model_name_cleaned_detail}_n{fixed_n_detailed_val}_d{d_features_val_detail}.png"
    full_path_plot = os.path.join(output_dir_path_str, filename_plot)
    plt.savefig(full_path_plot)
    plt.close()
    print(f"Saved p-value histogram to {full_path_plot}")


# Helper function for plotting comparison results (Part 1 of paper experiments)
def plot_results(
    results_data_plot: list, # This is experiment_results_for_task from compare_dimensions
    model_name_prefix_str_plot: str,
    output_dir_plot_path_str: str,
    use_projection_for_title: bool = False # Passed from compare_dimensions
):
    """
    Plots Type I error rates vs. N and D, and saves data to Excel.
    results_data_plot is a list of dictionaries, each from run_simulation_experiment.
    """
    if not os.path.exists(output_dir_plot_path_str):
        os.makedirs(output_dir_plot_path_str)

    if not results_data_plot:
        print(f"No results to plot for {model_name_prefix_str_plot}")
        return

    df_table_out = pd.DataFrame(results_data_plot)
    
    # Ensure correct column names, especially for 'd_proj_passed'
    # df_table_out might look like:
    # [{'n_samples': N, 'd_features': D, ..., 'd_proj_passed': VAL}, ...]
    # We want columns: N_Samples, D_Features, Type_I_Error_Rate, Projection_Dimension_Used
    
    # Renaming for clarity in the table
    df_table_out = df_table_out.rename(columns={
        'n_samples': 'N_Samples',
        'd_features': 'D_Features',
        'type_I_error_rate': 'Type_I_Error_Rate',
        'd_proj_passed': 'Projection_Dimension_Used' 
    })

    # Formatting Projection_Dimension_Used for display
    def format_proj_dim(row):
        proj_dim = row['Projection_Dimension_Used']
        orig_dim = row['D_Features']
        if proj_dim is None or pd.isna(proj_dim):
            return 'N/A (Original)'
        if proj_dim == orig_dim:
            return f"{orig_dim} (Original)"
        return str(proj_dim)

    if 'Projection_Dimension_Used' in df_table_out.columns and 'D_Features' in df_table_out.columns:
         df_table_out['Projection_Dimension_Used'] = df_table_out.apply(format_proj_dim, axis=1)
    elif 'D_Features' in df_table_out.columns : # Proj dim not available, mark all as original
        df_table_out['Projection_Dimension_Used'] = df_table_out['D_Features'].astype(str) + " (Original)"
    else: # Fallback if D_Features is also missing for some reason
        df_table_out['Projection_Dimension_Used'] = "N/A"


    # Select relevant columns for Excel
    excel_cols = ['N_Samples', 'D_Features', 'Type_I_Error_Rate', 'Projection_Dimension_Used']
    df_excel = df_table_out[excel_cols]

    excel_filename = f"{model_name_prefix_str_plot}_dimension_comparison_data.xlsx"
    full_path_excel = os.path.join(output_dir_plot_path_str, excel_filename)
    try:
        df_excel.to_excel(full_path_excel, index=False, sheet_name='SimResults')
        print(f"Saved comparison data to {full_path_excel}")
    except Exception as e:
        print(f"Error saving to Excel {full_path_excel}: {e}. Ensure 'openpyxl' is installed.")


    # Plotting (Type I error vs. N for different D, and Type I error vs. D for different N)
    # This requires pivoting the data or iterating through unique values of N and D.
    
    title_suffix = " (Projection Enabled, Dynamic Dim)" if use_projection_for_title else ""
    main_plot_title = f"{model_name_prefix_str_plot}{title_suffix}"

    # Plot 1: Type I Error Rate vs. N_Samples (lines for each D_Features)
    plt.figure(figsize=(12, 7))
    unique_dims = sorted(df_table_out['D_Features'].unique())
    for d_feat in unique_dims:
        subset = df_table_out[df_table_out['D_Features'] == d_feat]
        plt.plot(subset['N_Samples'], subset['Type_I_Error_Rate'], marker='o', label=f'D = {d_feat}')
    
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Type I Error Rate (alpha=0.05)")
    plt.title(f"Type I Error Rate vs. N_Samples\n{main_plot_title}")
    plt.legend(title="Dimension (D)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(0.05, color='r', linestyle='--', label='Nominal alpha=0.05')
    plt.legend() # Show legend again to includehline
    
    plot_filename_vs_n = f"{model_name_prefix_str_plot}_error_vs_N.png"
    full_path_plot_vs_n = os.path.join(output_dir_plot_path_str, plot_filename_vs_n)
    plt.savefig(full_path_plot_vs_n)
    plt.close()
    print(f"Saved plot to {full_path_plot_vs_n}")

    # Plot 2: Type I Error Rate vs. D_Features (lines for each N_Samples)
    plt.figure(figsize=(12, 7))
    unique_samples = sorted(df_table_out['N_Samples'].unique())
    for n_samp in unique_samples:
        subset = df_table_out[df_table_out['N_Samples'] == n_samp]
        plt.plot(subset['D_Features'], subset['Type_I_Error_Rate'], marker='o', label=f'N = {n_samp}')

    plt.xlabel("Dimension (D)")
    plt.ylabel("Type I Error Rate (alpha=0.05)")
    plt.title(f"Type I Error Rate vs. Dimension (D)\n{main_plot_title}")
    plt.legend(title="N_Samples")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(0.05, color='r', linestyle='--', label='Nominal alpha=0.05')
    plt.legend()

    plot_filename_vs_d = f"{model_name_prefix_str_plot}_error_vs_D.png"
    full_path_plot_vs_d = os.path.join(output_dir_plot_path_str, plot_filename_vs_d)
    plt.savefig(full_path_plot_vs_d)
    plt.close()
    print(f"Saved plot to {full_path_plot_vs_d}")
