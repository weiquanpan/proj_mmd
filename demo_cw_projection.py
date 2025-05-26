# This script demonstrates how to use the modified cw_test.py
# to run experiments with and without the random projection feature.

import torch
from cw_test import run_paper_experiments_pytorch

if __name__ == "__main__":
    # Demo script execution will go here
    print("Starting demo: Running experiments WITHOUT projection...")
    print("This will use the default parameters set within run_paper_experiments_pytorch in cw_test.py.")
    print("For a quicker demo, consider modifying parameters like n_reps_model1, n_reps_detailed_exp,")
    print("n_mc_samples_test, and the N/D lists directly in cw_test.py for faster execution.")
    
    # Determine device
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {selected_device}")

    results_no_projection = run_paper_experiments_pytorch(
        apply_projection=False,
        device=selected_device,
        # For a truly minimal demo, you might want to also pass very small lists for experiment scope,
        # if run_paper_experiments_pytorch is modified to accept them.
        # e.g., n_samples_model1_list=[50], d_features_model1_list=[10], n_reps_model1=5,
        # n_samples_detailed_exp_val=50, dimensions_for_detailed_analysis_exp=[10],
        # covariance_models_to_analyze_exp=["Identity"], n_reps_detailed_exp=5,
        # n_mc_samples_test=100 # This is for the normality test's MC samples
    )
    # The function run_paper_experiments_pytorch currently returns a dict of results,
    # but the print statement implies it returns the base directory.
    # Let's assume it returns the base_output_dir or we can infer it.
    # From cw_test.py, run_paper_experiments_pytorch returns 'all_results' dictionary.
    # The base_output_dir is created within it but not returned.
    # For the demo, we will just print that the execution is complete.
    print("Demo run (without projection) complete.")
    if results_no_projection: # Check if results were returned
         print("Results dictionary (no projection):")
         # Printing full results can be verbose, maybe just keys or a summary.
         # For now, let's print a marker that it finished and returned.
         print(f"  Top-level keys in results: {list(results_no_projection.keys())}")
         # The actual directory name is constructed inside run_paper_experiments_pytorch
         # and not returned. We can't print its exact name here without modifying that function.
         print("  (Refer to console output from cw_test.py for specific results directory name)")

    print("\n\n--- DEMO: Running experiments WITH projection ---")
    print("This will use the default parameters set within run_paper_experiments_pytorch.")
    print("Projection dimensions will be dynamically calculated based on original data dimensions.")
    # selected_device is already defined from the "without projection" part.
    print(f"Running on device: {selected_device}") 

    results_with_projection = run_paper_experiments_pytorch(
        apply_projection=True,
        device=selected_device
        # Similar to the above, for a truly minimal demo, parameters could be overridden here
        # if run_paper_experiments_pytorch is modified to accept them.
    )
    print("Demo run (with projection) complete.")
    if results_with_projection: # Check if results were returned
         print("Results dictionary (with projection):")
         print(f"  Top-level keys in results: {list(results_with_projection.keys())}")
         print("  (Refer to console output from cw_test.py for specific results directory name)")

    print("\n---")
    print("Demo script finished.")
    print("Each call to run_paper_experiments_pytorch creates a new timestamped directory.")
    print("Inside each of these directories, you can find:")
    print("  - 'pytorch_experiment_parameters.csv': Check 'Projection Status' and 'Projection Dimension Logic'.")
    print("  - CSV and Excel files for detailed analysis and dimension comparison.")
    print("  - Plots corresponding to the analyses.")
    print("Compare the contents of the directories from the 'WITH projection' and 'WITHOUT projection' runs.")
