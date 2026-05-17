#!/usr/bin/env python3
"""
Cloth-type comparison visualization script.
Groups by cloth type for the grasp action and the CD(R2S) metric.
Shows fixed_point and robot mode side-by-side as subplots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set font and style
plt.rcParams['font.family'] = 'serif'

class ClothTypeComparisonAnalyzer:
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.target_action = "grasp"
        self.target_metric = "chamfer_l1_real_to_sim"
        self.metric_label = "CD(R2S)"
        self.modes = ["fixed_point", "robot"]

        # Environment list (same order as before)
        self.environments = ["garment_dynamics", "pybullet", "isaacsim"]

        # Color palette (kept identical to prior version)
        self.env_colors = {
            'pybullet': '#FF6B9D',         # pink - vibrant
            'garment_dynamics': '#45B7D1', # sky blue - fresh and bright
            'isaacsim': '#96CEB4'          # mint green - fresh and natural
        }

        # Cloth type list (populated dynamically from data)
        self.cloth_types = []
        self.cloth_display_names = []

        # Excluded cloth types
        self.excluded_cloth_types = ["beige_hoodie"]
        # Image save directory
        self.save_dir = Path(outputs_dir).parent / "outputs/analysis/chart_simmode"
        self.save_dir.mkdir(parents=True, exist_ok=True)


    def process_cloth_name(self, cloth_name):
        """Strip the color prefix from a cloth name."""
        # Common color prefixes
        color_prefixes = [
            'beige_', 'black_', 'blue_', 'brown_', 'gray_', 'grey_',
            'green_', 'orange_', 'pink_', 'purple_', 'red_', 'white_', 'yellow_'
        ]

        # Strip the color prefix
        for color in color_prefixes:
            if cloth_name.startswith(color):
                return cloth_name[len(color):]

        return cloth_name

    def get_environments_for_mode(self, mode):
        """Return the list of environments for the given mode."""
        if mode == "robot":
            # Exclude pybullet in robot mode
            return [env for env in self.environments if env != "pybullet"]
        else:
            # Other modes use all environments
            return self.environments

    def collect_grasp_cd_results(self) -> pd.DataFrame:
        """Collect CD(R2S) results for the grasp action."""
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

                    # Process only the grasp action and the requested modes
                    if action != self.target_action:
                        continue
                    if mode not in self.modes:
                        continue

                    # Drop excluded cloth types
                    if cloth in self.excluded_cloth_types:
                        continue

                    # Exclude pybullet in robot mode
                    if mode == "robot" and environment == "pybullet":
                        continue

                    # Build experiment key
                    experiment_key = (cloth, action, environment, mode, sample)

                    # Store every version
                    if experiment_key not in experiment_versions:
                        experiment_versions[experiment_key] = []

                    experiment_versions[experiment_key].append({
                        'timestamp': timestamp,
                        'file_path': metrics_file,
                        'robot': robot
                    })

            except Exception as e:
                print(f"Failed to parse file path {metrics_file}: {e}")
                continue

        print(f"Found {len(experiment_versions)} experiment combinations for action '{self.target_action}'")

        # Pick the latest timestamp version per experiment
        for experiment_key, versions in experiment_versions.items():
            cloth, action, environment, mode, sample = experiment_key

            # Sort by timestamp and pick the latest
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            latest_version = versions[0]

            try:
                # Read metrics from the latest version
                df = pd.read_csv(latest_version['file_path'])

                if len(df) < 2:
                    continue

                # Skip if the target metric column is missing
                if self.target_metric not in df.columns:
                    continue

                # Second-to-last row is mean; last row is std
                mean_row = df.iloc[-2]
                std_row = df.iloc[-1]

                # Pull mean and std for the target metric
                mean_val = mean_row[self.target_metric] if pd.notna(mean_row[self.target_metric]) else np.nan
                std_val = std_row[self.target_metric] if pd.notna(std_row[self.target_metric]) else np.nan

                if pd.notna(mean_val):  # keep only valid data
                    all_results.append({
                        'cloth_type': cloth,
                        'action': action,
                        'environment': environment,
                        'mode': mode,
                        'sample': sample,
                        'timestamp': latest_version['timestamp'],
                        'mean': mean_val,
                        'std': std_val if pd.notna(std_val) else 0
                    })

            except Exception as e:
                print(f"Processing failed for {latest_version['file_path']}: {e}")
                continue

        if not all_results:
            raise ValueError(f"No valid metrics files found for {self.target_action} action and {self.target_metric} metric!")

        result_df = pd.DataFrame(all_results)

        # Build the cloth type list (excluded types are removed)
        all_cloth_types = result_df['cloth_type'].unique()
        filtered_cloth_types = [ct for ct in all_cloth_types if ct not in self.excluded_cloth_types]

        # Strip color prefixes from names; sort by processed display name
        processed_cloth_types = []
        for cloth_type in filtered_cloth_types:
            processed_name = self.process_cloth_name(cloth_type)
            processed_cloth_types.append((cloth_type, processed_name))

        # Sort by display name; keep originals for data lookup
        processed_cloth_types.sort(key=lambda x: x[1])
        self.cloth_types = [ct[0] for ct in processed_cloth_types]  # original name
        self.cloth_display_names = [ct[1] for ct in processed_cloth_types]  # display name

        print(f"Total collected: {len(result_df)} experiments for action '{self.target_action}'")
        print(f"Found {len(self.cloth_types)} cloth types: {', '.join(self.cloth_display_names)}")
        if self.excluded_cloth_types:
            print(f"Excluded cloth types: {', '.join(self.excluded_cloth_types)}")

        return result_df

    def aggregate_data_for_plotting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data for plotting."""
        plot_data = []

        for mode in self.modes:
            # Environments available for this mode
            mode_environments = self.get_environments_for_mode(mode)

            for cloth_type in self.cloth_types:
                for env in mode_environments:
                    # Filter to current combination
                    filtered_data = df[
                        (df['mode'] == mode) &
                        (df['cloth_type'] == cloth_type) &
                        (df['environment'] == env)
                    ]

                    if len(filtered_data) > 0:
                        # Mean and std across all samples
                        mean_values = filtered_data['mean'].dropna()
                        std_values = filtered_data['std'].dropna()

                        if len(mean_values) > 0:
                            overall_mean = mean_values.mean()
                            overall_std = std_values.mean() if len(std_values) > 0 else 0

                            plot_data.append({
                                'mode': mode,
                                'cloth_type': cloth_type,
                                'environment': env,
                                'mean': overall_mean,
                                'std': overall_std,
                                'sample_count': len(mean_values)
                            })

        return pd.DataFrame(plot_data)

    def create_cloth_type_comparison_plot(self, plot_df: pd.DataFrame, save_path: str = None):
        """Create the cloth-type comparison plot with subplots for fixed_point and robot."""
        # Figure size and layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('white')

        # One subplot per mode
        for i, mode in enumerate(self.modes):
            ax = axes[i]

            # Filter to the current mode
            mode_data = plot_df[plot_df['mode'] == mode]

            if len(mode_data) == 0:
                ax.text(0.5, 0.5, f'No data for {mode}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#E74C3C')
                ax.set_title(f'{mode.replace("_", " ").title()}',
                           fontsize=24, fontweight='bold', color='#2C3E50')
                continue

            # Environments available for this mode
            mode_environments = self.get_environments_for_mode(mode)

            # Build plotting data
            x_pos = np.arange(len(self.cloth_types))
            bar_width = 0.25 if len(mode_environments) == 3 else 0.35  # adjust bar width

            # Collect all data points to set the y-axis range
            all_means = []
            all_stds = []

            # Draw a bar per environment
            for j, env in enumerate(mode_environments):
                env_data = mode_data[mode_data['environment'] == env]

                means = []
                stds = []

                # Iterate cloth types in display order
                for cloth_type in self.cloth_types:
                    cloth_data = env_data[env_data['cloth_type'] == cloth_type]
                    if len(cloth_data) > 0:
                        mean_val = cloth_data['mean'].iloc[0]
                        std_val = cloth_data['std'].iloc[0]
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
            ax.set_title(f'{mode.replace("_", " ").title()}',
                        fontsize=26, fontweight='bold', color='#2C3E50', pad=20)
            # Subplot label (a) (b)
            ax.text(0.5, -0.29, f"({chr(ord('a') + i)})", fontsize=22, fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(f'{self.metric_label} Value', fontsize=24, fontweight='bold', color='#34495E')

            # x-axis tick positions (depends on environment count)
            if len(mode_environments) == 3:
                ax.set_xticks(x_pos + bar_width)
            else:
                ax.set_xticks(x_pos + bar_width / 2)

            # Use processed display names
            ax.set_xticklabels([name.replace('_', ' ').title() for name in self.cloth_display_names],
                            rotation=45, ha='right', fontsize=22, color='#2C3E50')

            # Grid styling
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8, color='#BDC3C7')
            ax.set_axisbelow(True)

            # Legend styling
            legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                            shadow=False, fontsize=15,
                            markerfirst=True, numpoints=1, markerscale=1.0)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.4)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1)

            # y-axis range - make sure error bars are fully visible
            if len(all_means) > 0 and len(all_stds) > 0:
                max_with_error = max([m + s for m, s in zip(all_means, all_stds)])
                ax.set_ylim(bottom=0, top=max_with_error * 1.15)
            elif len(all_means) > 0:
                max_val = max(all_means)
                ax.set_ylim(bottom=0, top=max_val * 1.2)

            # Axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#BDC3C7')
            ax.spines['bottom'].set_color('#BDC3C7')

            # Tick colors
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='both', colors='#2C3E50')

        # Adjust layout
        plt.tight_layout()

        # Save image
        if save_path is None:
            save_path = self.save_dir / f"cloth_types_{self.target_action}_{self.target_metric}_comparison.png"
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
        print(f"Cloth-type comparison summary (action: {self.target_action}, metric: {self.metric_label})")
        print("="*80)

        print(f"\nBasic info:")
        print(f"  Target action: {self.target_action}")
        print(f"  Target metric: {self.metric_label} ({self.target_metric})")
        print(f"  Simulation modes: {len(self.modes)} ({', '.join(self.modes)})")
        print(f"  Cloth types: {len(self.cloth_types)} ({', '.join(self.cloth_display_names)})")
        print(f"  Total data points: {len(plot_df)}")

        print(f"\nEnvironment configuration:")
        for mode in self.modes:
            mode_environments = self.get_environments_for_mode(mode)
            print(f"  {mode}: {len(mode_environments)} environments ({', '.join(mode_environments)})")

        print(f"\nData completeness:")
        for mode in self.modes:
            mode_data = plot_df[plot_df['mode'] == mode]
            mode_environments = self.get_environments_for_mode(mode)
            print(f"  {mode}: {len(mode_data)} data points")
            for env in mode_environments:
                env_count = len(mode_data[mode_data['environment'] == env])
                print(f"    {env}: {env_count}/{len(self.cloth_types)} cloth types")

        print(f"\nAverage performance per environment:")
        for mode in self.modes:
            print(f"  {mode}:")
            mode_data = plot_df[plot_df['mode'] == mode]
            if len(mode_data) > 0:
                env_performance = mode_data.groupby('environment')['mean'].mean().sort_values()
                for i, (env, avg_val) in enumerate(env_performance.items(), 1):
                    print(f"    {i}. {env}: {avg_val:.6f}")
            else:
                print(f"    no data")

        print("\n" + "="*80)


def main():
    """Main entrypoint."""
    print("Starting cloth-type comparison analysis")

    # Locate project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, "outputs")

    analyzer = ClothTypeComparisonAnalyzer(outputs_dir=outputs_dir)

    try:
        print(f"\nCollecting {analyzer.metric_label} data for action '{analyzer.target_action}'...")
        raw_df = analyzer.collect_grasp_cd_results()

        print("Aggregating data for plotting...")
        plot_df = analyzer.aggregate_data_for_plotting(raw_df)

        print("Creating visualization chart...")

        # Create the cloth-type comparison plot
        main_fig = analyzer.create_cloth_type_comparison_plot(plot_df)

        # Print data summary
        analyzer.print_data_summary(plot_df)

        print(f"\nVisualization analysis complete!")
        print(f"Generated chart: cloth_types_{analyzer.target_action}_{analyzer.target_metric}_comparison.png")

        print(f"\nChart notes:")
        print(f"   Left and right subplots correspond to fixed_point and robot modes")
        print(f"   X-axis: cloth type (color prefix removed)")
        print(f"   Y-axis: {analyzer.metric_label} value")
        print(f"   Bar color corresponds to simulation environment")
        print(f"   Error bars show standard deviation")
        print(f"   Metric is 'smaller is better'")
        print(f"   robot mode excludes pybullet; beige_hoodie is excluded")

    except Exception as e:
        print(f"Visualization analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
