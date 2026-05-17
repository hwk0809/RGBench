#!/usr/bin/env python3
"""
Environment comparison visualization script.
Uses the single cloth green_tshirt and draws per-action metric comparisons.
Shows mean +/- standard deviation with a vibrant palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

import warnings
import matplotlib
warnings.filterwarnings('ignore')

# Font and style
plt.rcParams['font.family'] = 'serif'

def select_mode():
    """Prompt the user to choose a simulation mode."""
    print("\nPlease choose a simulation mode:")
    print("1. fixed_point")
    print("2. robot")

    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == "1":
            return "fixed_point"
        elif choice == "2":
            return "robot"
        else:
            print("Invalid choice. Please enter 1 or 2")

class ClothVisualizationAnalyzer:
    def __init__(self, outputs_dir: str = "outputs", mode: str = "fixed_point"):
        self.outputs_dir = Path(outputs_dir)
        self.mode = mode
        # Image save directory
        self.save_dir = Path(outputs_dir).parent / "outputs/analysis/chart_action"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.all_metrics_columns = [
            'chamfer_l1_sim_to_real',
            'chamfer_l1_real_to_sim',
            'one_sided_hausdorff_sim_to_real',
            'one_sided_hausdorff_real_to_sim',
            'z_mean_error'
        ]

        # Metric label aliases
        self.metric_labels = {
            'chamfer_l1_sim_to_real': r'CD$_{s2r}$',
            'chamfer_l1_real_to_sim': r'CD$_{r2s}$',
            'one_sided_hausdorff_sim_to_real': r'HD$_{s2r}$',
            'one_sided_hausdorff_real_to_sim': r'HD$_{r2s}$',
            'z_mean_error': r'ZD'
        }

        # Environments, actions, target cloth - garment_dynamics placed first
        self.environments = ["garment_dynamics", "pybullet", "isaacsim"]
        self.actions = ["grasp", "fold", "fling"]  # swap order of grasp and fling
        self.target_cloth = "green_tshirt"

        # Vibrant palette
        self.env_colors = {
            'pybullet': '#FF6B9D',         # pink - vibrant
            'garment_dynamics': '#45B7D1', # sky blue - fresh and bright
            'isaacsim': '#96CEB4'          # mint green - fresh and natural
        }

    def get_metrics_for_action(self, action):
        """Return the metric list for the given action."""
        # Exclude sim_stability_score
        base_metrics = [
            'chamfer_l1_sim_to_real',
            'chamfer_l1_real_to_sim',
            'one_sided_hausdorff_sim_to_real',
            'one_sided_hausdorff_real_to_sim'
        ]

        # Only fold includes z_mean_error
        if action == 'fold':
            return base_metrics + ['z_mean_error']
        else:
            return base_metrics

    def collect_target_cloth_results(self) -> pd.DataFrame:
        """Collect experiment results for the target cloth."""
        all_results = []

        # Recursively search every metrics.csv file
        metrics_files = list(self.outputs_dir.glob("**/metrics.csv"))
        print(f"Found {len(metrics_files)} result files")

        # Dict that stores every timestamp version per experiment
        experiment_versions = {}

        for metrics_file in metrics_files:
            try:
                # Parse experiment info from the path
                path_parts = metrics_file.parts
                if len(path_parts) >= 8:
                    cloth = path_parts[-8]
                    action = path_parts[-7]
                    environment = path_parts[-6]
                    mode = path_parts[-5]
                    robot = path_parts[-4]
                    sample = path_parts[-3]
                    timestamp = path_parts[-2]

                    # Keep only the target cloth and mode
                    if cloth != self.target_cloth:
                        continue
                    if mode != self.mode:
                        continue

                    # Build experiment key
                    experiment_key = (cloth, action, environment, sample)

                    # Store every version
                    if experiment_key not in experiment_versions:
                        experiment_versions[experiment_key] = []

                    experiment_versions[experiment_key].append({
                        'timestamp': timestamp,
                        'file_path': metrics_file,
                        'mode': mode,
                        'robot': robot
                    })

            except Exception as e:
                print(f"Failed to parse file path {metrics_file}: {e}")
                continue

        print(f"Found {len(experiment_versions)} experiment combinations for {self.target_cloth} (mode: {self.mode})")

        # Pick the latest timestamp version per experiment
        for experiment_key, versions in experiment_versions.items():
            cloth, action, environment, sample = experiment_key

            # Sort by timestamp and pick the latest
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            latest_version = versions[0]

            try:
                # Read metrics from the latest version
                df = pd.read_csv(latest_version['file_path'])

                if len(df) < 2:
                    continue

                # Second-to-last row is mean; last row is std (already a std, not variance)
                mean_row = df.iloc[-2]
                std_row = df.iloc[-1]

                # Build the result row
                result_row = {
                    'cloth': cloth,
                    'action': action,
                    'environment': environment,
                    'mode': latest_version['mode'],
                    'sample': sample,
                    'timestamp': latest_version['timestamp']
                }

                # Pull mean and std for every metric
                for metric in self.all_metrics_columns:
                    if metric in df.columns:
                        mean_val = mean_row[metric] if pd.notna(mean_row[metric]) else np.nan
                        std_val = std_row[metric] if pd.notna(std_row[metric]) else np.nan

                        result_row[f'{metric}_mean'] = mean_val
                        # std column already holds the std (no need to sqrt)
                        result_row[f'{metric}_std'] = std_val
                    else:
                        result_row[f'{metric}_mean'] = np.nan
                        result_row[f'{metric}_std'] = np.nan

                all_results.append(result_row)

            except Exception as e:
                print(f"Processing failed for {latest_version['file_path']}: {e}")
                continue

        if not all_results:
            raise ValueError(f"No valid metrics files found for {self.target_cloth} in mode '{self.mode}'!")

        result_df = pd.DataFrame(all_results)
        print(f"Total collected: {len(result_df)} experiments for {self.target_cloth}")

        return result_df

    def aggregate_data_for_plotting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data for plotting."""
        plot_data = []

        for action in self.actions:
            # Metric list for the current action
            action_metrics = self.get_metrics_for_action(action)

            for env in self.environments:
                # Filter to current action and environment
                action_env_data = df[
                    (df['action'] == action) &
                    (df['environment'] == env)
                ]

                if len(action_env_data) > 0:
                    # Mean and std of every metric across all samples
                    for metric in action_metrics:
                        mean_col = f'{metric}_mean'
                        std_col = f'{metric}_std'

                        if mean_col in action_env_data.columns:
                            # Mean across samples
                            mean_values = action_env_data[mean_col].dropna()
                            std_values = action_env_data[std_col].dropna()

                            if len(mean_values) > 0:
                                overall_mean = mean_values.mean()
                                # Average of per-sample stds
                                overall_std = std_values.mean() if len(std_values) > 0 else 0

                                plot_data.append({
                                    'action': action,
                                    'environment': env,
                                    'metric': metric,
                                    'metric_label': self.metric_labels[metric],
                                    'mean': overall_mean,
                                    'std': overall_std,
                                    'sample_count': len(mean_values)
                                })

        return pd.DataFrame(plot_data)


    def create_action_comparison_plots(self, plot_df: pd.DataFrame, save_path: str = None):
        """Create the per-action metric comparison plots with mean +/- std error bars."""
        # Figure size and layout
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.patch.set_facecolor('white')
        # No overall title; only per-action titles

        # One subplot per action
        for i, action in enumerate(self.actions):
            ax = axes[i]

            # Filter data for this action
            action_data = plot_df[plot_df['action'] == action]

            if len(action_data) == 0:
                ax.text(0.5, 0.5, f'No data for {action}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#E74C3C')
                ax.set_title(f'{action.title()}', fontsize=16, fontweight='bold', color='#2C3E50')
                continue

            # Metric list for the current action
            action_metrics = self.get_metrics_for_action(action)

            # Build plotting data
            x_pos = np.arange(len(action_metrics))
            bar_width = 0.25

            # Collect data points to set y-axis range
            all_means = []
            all_stds = []

            # One bar per environment
            for j, env in enumerate(self.environments):
                env_data = action_data[action_data['environment'] == env]

                means = []
                stds = []

                # Iterate metrics in order
                for metric in action_metrics:
                    metric_data = env_data[env_data['metric'] == metric]
                    if len(metric_data) > 0:
                        mean_val = metric_data['mean'].iloc[0]
                        std_val = metric_data['std'].iloc[0]
                        means.append(mean_val)
                        stds.append(std_val)
                        # Collect for y-range computation
                        if mean_val > 0:
                            all_means.append(mean_val)
                            all_stds.append(std_val)
                    else:
                        means.append(0)
                        stds.append(0)

                # Draw bars
                positions = x_pos + j * bar_width
                bars = ax.bar(positions, means, bar_width,
                            label=env, color=self.env_colors[env],
                            alpha=0.85, edgecolor='white', linewidth=1.5)

                # Add error bars (mean +/- std)
                ax.errorbar(positions, means, yerr=stds,
                        fmt='none', ecolor='black', capsize=4,
                        capthick=1.5, elinewidth=1.5, alpha=0.8)

                # Give bars depth
                for bar in bars:
                    bar.set_zorder(3)

            # Subplot style
            ax.set_title(f'{action.title()}', fontsize=22, fontweight='bold',
                        color='#2C3E50', pad=20)
            # Subplot label (a) (b) (c)
            ax.text(0.5, -0.22, f"({chr(ord('a') + i)})", fontsize=20, fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Value', fontsize=18, fontweight='bold', color='#34495E')
            ax.set_xticks(x_pos + bar_width)
            ax.set_xticklabels([self.metric_labels[m] for m in action_metrics],
                            rotation=45, ha='right', fontsize=18, color='#2C3E50')

            # Grid styling
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8, color='#BDC3C7')
            ax.set_axisbelow(True)

            # Legend styling
            legend = ax.legend(loc='upper left', frameon=True, fancybox=False,
                            shadow=False, fontsize=15,
                            markerfirst=True, numpoints=1, markerscale=1.0)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.4)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1)


            # y-axis range - make sure error bars are fully visible
            if len(all_means) > 0 and len(all_stds) > 0:
                # Top of the error bars
                max_with_error = max([m + s for m, s in zip(all_means, all_stds)])
                # Give grasp extra headroom
                if action == 'grasp':
                    ax.set_ylim(bottom=0, top=max_with_error * 1.25)  # 25% headroom
                else:
                    ax.set_ylim(bottom=0, top=max_with_error * 1.15)  # 15% headroom
            elif len(all_means) > 0:
                max_val = max(all_means)
                ax.set_ylim(bottom=0, top=max_val * 1.2)

            # Axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#BDC3C7')
            ax.spines['bottom'].set_color('#BDC3C7')

            # Tick colors
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='both', colors='#2C3E50')

        # Adjust layout
        plt.tight_layout()

        # Save image (png and pdf)
        if save_path is None:
            save_path = self.save_dir / f"{self.target_cloth}_metrics_comparison_{self.mode}.png"
        else:
            save_path = self.save_dir / Path(save_path).name

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.2)
        pdf_path = str(save_path).rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.2)
        print(f"Chart saved: {save_path} and {pdf_path}")

        # Show figure
        plt.show()

        return fig


    def print_data_summary(self, plot_df: pd.DataFrame):
        """Print a data summary."""
        print("\n" + "="*80)
        print(f"{self.target_cloth.replace('_', ' ').title()} visualization summary (mode: {self.mode})")
        print("="*80)

        print(f"\nBasic info:")
        print(f"  Target cloth: {self.target_cloth}")
        print(f"  Simulation mode: {self.mode}")
        print(f"  Action types: {len(self.actions)} ({', '.join(self.actions)})")
        print(f"  Simulation environments: {len(self.environments)} ({', '.join(self.environments)})")
        print(f"  Total data points: {len(plot_df)}")

        print(f"\nMetric configuration:")
        for action in self.actions:
            metrics = self.get_metrics_for_action(action)
            print(f"  {action}: {len(metrics)} metrics ({', '.join([self.metric_labels[m] for m in metrics])})")

        print(f"\nData completeness:")
        for action in self.actions:
            action_data = plot_df[plot_df['action'] == action]
            print(f"  {action}: {len(action_data)} data points")
            for env in self.environments:
                env_count = len(action_data[action_data['environment'] == env])
                expected_count = len(self.get_metrics_for_action(action))
                print(f"    {env}: {env_count}/{expected_count} metrics")

        print(f"\nAverage performance per environment:")
        env_performance = plot_df.groupby('environment')['mean'].mean().sort_values()
        for i, (env, avg_val) in enumerate(env_performance.items(), 1):
            print(f"  {i}. {env}: {avg_val:.6f}")

        print("\n" + "="*80)


def main():
    """Main entrypoint."""
    print("Starting Green T-shirt visualization analysis")

    # Prompt for mode
    selected_mode = select_mode()

    # Locate project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, "outputs")

    analyzer = ClothVisualizationAnalyzer(outputs_dir=outputs_dir, mode=selected_mode)

    try:
        print(f"\nCollecting experiment data for {analyzer.target_cloth}...")
        raw_df = analyzer.collect_target_cloth_results()

        print("Aggregating data for plotting...")
        plot_df = analyzer.aggregate_data_for_plotting(raw_df)

        print("Creating visualization chart...")

        # Main comparison chart
        main_fig = analyzer.create_action_comparison_plots(plot_df)

        # Print data summary
        analyzer.print_data_summary(plot_df)

        print(f"\nVisualization analysis complete!")
        print(f"Generated chart:")
        print(f"   1. {analyzer.target_cloth}_metrics_comparison_{selected_mode}.png")
        print(f"   2. {analyzer.target_cloth}_summary_statistics_{selected_mode}.png")

        print(f"\nChart notes:")
        print(f"   Main comparison: 3 subplots for 3 actions (grasp, fold, fling)")
        print(f"   Environment order: garment_dynamics, pybullet, isaacsim")
        print(f"   Excluded metric: sim_stability_score")
        print(f"   Special metric: only fold includes z_mean_error")
        print(f"   All metrics are 'smaller is better'")
        print(f"   Using vibrant palette: sky blue (garment_dynamics), pink (pybullet), mint green (isaacsim)")

    except Exception as e:
        print(f"Visualization analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
