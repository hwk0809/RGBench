#!/usr/bin/env python3
"""
High-quality metric visualization script (full-label variant) - keeps full X-axis labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ConferenceQualityVisualizerComplete:
    def __init__(self, excel_file: str = "cloth_metrics_simple.xlsx"):
        self.excel_file = excel_file
        self.metrics_columns = [
            'chamfer_l1_sim_to_real',
            'chamfer_l2_sim_to_real', 
            'chamfer_l1_real_to_sim',
            'one_sided_hausdorff_sim_to_real',
            'one_sided_hausdorff_real_to_sim',
            'sim_stability_score',
            'z_mean_error'
        ]
        
        # Full metric display names
        self.metrics_labels = [
            'Chamfer L1\n(Sim→Real)',
            'Chamfer L2\n(Sim→Real)', 
            'Chamfer L1\n(Real→Sim)',
            'Hausdorff\n(Sim→Real)',
            'Hausdorff\n(Real→Sim)',
            'Stability\nScore',
            'Z-axis\nError'
        ]
        
        # Configure publication-quality style
        self._setup_publication_style()
        
        # Define a professional color palette
        self.cloth_colors = {
            'blue_dress': '#2E86AB',      # professional blue
            'green_tshirt': '#A23B72',    # deep magenta
            'grey_pleat_skirt': '#F18F01', # orange
            'grey_sunwear': '#C73E1D',    # deep red
            'khaki_blazer': '#6A994E',    # olive green
            'white_cakeskirt': '#7209B7', # purple
            'white_shirt': '#264653',     # deep teal
        }
        
        # Backup colors
        self.backup_colors = ['#E76F51', '#F4A261', '#E9C46A', '#2A9D8F', '#457B9D']
        
    def _setup_publication_style(self):
        """Configure publication-quality plot style."""
        plt.rcParams.update({
            # Font settings
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 9,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14,
            
            # Lines and markers
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'patch.linewidth': 0.5,
            
            # Figure quality
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,  # add margin
            
            # Colors and style
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.axisbelow': True,
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            
            # Remove top and right spines
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Tick settings
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            
            # Text rendering
            'text.usetex': False,
            'mathtext.fontset': 'stix',
        })
        
    def load_data(self) -> pd.DataFrame:
        """Load Excel data."""
        try:
            df = pd.read_excel(self.excel_file, sheet_name='metric_stats')
            print(f"Loaded {len(df)} rows of experiment data")
            return df
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None
    
    def prepare_visualization_data(self, df: pd.DataFrame) -> dict:
        """Prepare data for visualization."""
        action_data = {}
        
        for action in sorted(df['action'].unique()):
            action_df = df[df['action'] == action]
            cloth_metrics = {}
            
            for cloth in sorted(action_df['cloth'].unique()):
                cloth_df = action_df[action_df['cloth'] == cloth]
                metrics_means = []
                
                for metric in self.metrics_columns:
                    mean_col = f'{metric}_mean'
                    if mean_col in cloth_df.columns:
                        overall_mean = cloth_df[mean_col].mean()
                        metrics_means.append(overall_mean)
                    else:
                        metrics_means.append(np.nan)
                
                cloth_metrics[cloth] = metrics_means
            
            action_data[action] = cloth_metrics
            print(f"{action.title()} action: {len(cloth_metrics)} cloth types")
        
        return action_data
    
    def get_cloth_color(self, cloth_name: str) -> str:
        """Return the color assigned to the given cloth."""
        if cloth_name in self.cloth_colors:
            return self.cloth_colors[cloth_name]
        else:
            hash_idx = hash(cloth_name) % len(self.backup_colors)
            return self.backup_colors[hash_idx]
    
    def create_single_action_plot(self, action: str, cloth_metrics: dict, output_dir: Path):
        """Create a high-quality plot for a single action while keeping full labels."""
        # Use a larger canvas so full labels fit
        fig_width = 10.0  # extra width
        fig_height = 6.0   # extra height
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        x_positions = np.arange(len(self.metrics_columns))
        
        # Plot a line per cloth
        for cloth, metrics in cloth_metrics.items():
            valid_indices = ~np.isnan(metrics)
            valid_x = x_positions[valid_indices]
            valid_y = np.array(metrics)[valid_indices]
            
            if len(valid_y) > 0:
                color = self.get_cloth_color(cloth)
                cloth_label = cloth.replace('_', ' ').title()
                
                ax.plot(valid_x, valid_y, 
                       marker='o', markersize=5, linewidth=2,
                       label=cloth_label,
                       color=color,
                       alpha=0.9,
                       markerfacecolor=color,
                       markeredgecolor='white',
                       markeredgewidth=0.8)
        
        # Configure chart attributes
        ax.set_title(f'{action.title()} Action Performance', 
                    fontweight='bold', pad=25)  # extra title padding
        ax.set_xlabel('Evaluation Metrics', fontweight='bold', labelpad=15)
        ax.set_ylabel('Mean Values (lower is better)', fontweight='bold', labelpad=10)
        
        # X-axis labels - keep full labels with a better layout
        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.metrics_labels, 
                          rotation=30,  # moderate rotation
                          ha='right', 
                          fontsize=8,
                          rotation_mode='anchor')
        
        # Configure Y-axis
        ax.set_yscale('log')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend - placed outside on the right
        legend = ax.legend(bbox_to_anchor=(1.02, 1), 
                          loc='upper left',
                          frameon=True, 
                          fancybox=False,
                          shadow=False,
                          framealpha=0.9,
                          edgecolor='gray')
        legend.get_frame().set_linewidth(0.5)
        
        # Adjust layout to leave room for full labels
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, right=0.75)  # extra bottom margin
        
        # Save chart
        output_base = output_dir / f'{action}_action_performance_complete'
        
        plt.savefig(f'{output_base}.pdf', 
                   format='pdf', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
        
        plt.savefig(f'{output_base}.png', 
                   format='png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
        
        plt.close()
        
        print(f"Saved full-label chart: {output_base}.pdf/.png")
        return output_base
    
    def create_combined_comparison_plot(self, action_data: dict, output_dir: Path):
        """Create a three-action comparison plot while keeping full labels."""
        # Use a larger canvas
        fig_width = 18.0  # much wider
        fig_height = 7.0   # taller
        
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        # Configure the main title
        fig.suptitle('Cloth Performance Comparison Across Different Actions', 
                     fontsize=16, fontweight='bold', y=0.92)
        
        x_positions = np.arange(len(self.metrics_columns))
        
        # Collect all cloths
        all_cloths = set()
        for cloth_metrics in action_data.values():
            all_cloths.update(cloth_metrics.keys())
        all_cloths = sorted(list(all_cloths))
        
        for idx, (action, cloth_metrics) in enumerate(action_data.items()):
            ax = axes[idx]
            
            # Plot a line per cloth
            for cloth in all_cloths:
                if cloth in cloth_metrics:
                    metrics = cloth_metrics[cloth]
                    valid_indices = ~np.isnan(metrics)
                    valid_x = x_positions[valid_indices]
                    valid_y = np.array(metrics)[valid_indices]
                    
                    if len(valid_y) > 0:
                        color = self.get_cloth_color(cloth)
                        cloth_label = cloth.replace('_', ' ').title()
                        
                        ax.plot(valid_x, valid_y, 
                               marker='o', markersize=4, linewidth=1.5,
                               label=cloth_label if idx == 0 else "",
                               color=color,
                               alpha=0.9,
                               markerfacecolor=color,
                               markeredgecolor='white',
                               markeredgewidth=0.5)
            
            # Configure subplot attributes
            ax.set_title(f'{action.title()}', fontsize=12, fontweight='bold', pad=1)
            ax.set_xlabel('Metrics', fontsize=10, labelpad=10)
            if idx == 0:
                ax.set_ylabel('Mean Values (log scale)', fontsize=10, labelpad=10)
            
            ax.set_xticks(x_positions)
            # Keep full labels; tweak font size and rotation
            ax.set_xticklabels(self.metrics_labels, 
                              rotation=25,  # reduced rotation
                              ha='right', 
                              fontsize=7,   # slightly smaller font
                              rotation_mode='anchor')
            
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
            ax.set_axisbelow(True)
        
        # Increase subplot spacing
        plt.subplots_adjust(wspace=0.4)
        
        # Create a shared legend - placed at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        
        # Split the legend across rows
        if len(labels) > 4:
            # Two-row layout
            mid_point = len(labels) // 2
            
            # First row
            fig.legend(handles[:mid_point], labels[:mid_point], 
                      loc='lower center', 
                      bbox_to_anchor=(0.5, 0.05),
                      ncol=mid_point,
                      frameon=False,
                      fontsize=8,
                      columnspacing=1.5)
            
            # Second row
            fig.legend(handles[mid_point:], labels[mid_point:], 
                      loc='lower center', 
                      bbox_to_anchor=(0.5, 0.00),
                      ncol=len(handles[mid_point:]),
                      frameon=False,
                      fontsize=8,
                      columnspacing=1.5)
            
            bottom_margin = 0.15
        else:
            fig.legend(handles, labels, 
                      loc='lower center', 
                      bbox_to_anchor=(0.5, -0.05),
                      ncol=len(labels),
                      frameon=False,
                      fontsize=8,
                      columnspacing=2.0)
            bottom_margin = 0.25
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_margin, top=0.85)
        
        # Save chart
        output_base = output_dir / 'all_actions_comparison_complete'
        
        plt.savefig(f'{output_base}.pdf', 
                   format='pdf', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.savefig(f'{output_base}.png', 
                   format='png', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.close()
        
        print(f"Saved full-label combined chart: {output_base}.pdf/.png")
        return output_base
    
    def create_ultra_wide_comparison_plot(self, action_data: dict, output_dir: Path):
        """Create an ultra-wide comparison plot to maximize label space."""
        # Use an ultra-wide canvas
        fig_width = 20.0  # very wide
        fig_height = 6.0
        
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        fig.suptitle('Cloth Performance Comparison Across Different Actions', 
                     fontsize=18, fontweight='bold', y=0.92)
        
        x_positions = np.arange(len(self.metrics_columns))
        
        # Collect all cloths
        all_cloths = set()
        for cloth_metrics in action_data.values():
            all_cloths.update(cloth_metrics.keys())
        all_cloths = sorted(list(all_cloths))
        
        for idx, (action, cloth_metrics) in enumerate(action_data.items()):
            ax = axes[idx]
            
            # Plot a line per cloth
            for cloth in all_cloths:
                if cloth in cloth_metrics:
                    metrics = cloth_metrics[cloth]
                    valid_indices = ~np.isnan(metrics)
                    valid_x = x_positions[valid_indices]
                    valid_y = np.array(metrics)[valid_indices]
                    
                    if len(valid_y) > 0:
                        color = self.get_cloth_color(cloth)
                        cloth_label = cloth.replace('_', ' ').title()
                        
                        ax.plot(valid_x, valid_y, 
                               marker='o', markersize=5, linewidth=2,
                               label=cloth_label if idx == 0 else "",
                               color=color,
                               alpha=0.9,
                               markerfacecolor=color,
                               markeredgecolor='white',
                               markeredgewidth=0.8)
            
            # Configure subplot attributes
            ax.set_title(f'{action.title()}', fontsize=14, fontweight='bold', pad=1)
            ax.set_xlabel('Evaluation Metrics', fontsize=11, labelpad=12)
            if idx == 0:
                ax.set_ylabel('Mean Values (log scale)', fontsize=11, labelpad=10)
            
            ax.set_xticks(x_positions)
            # Full labels with larger font and slight rotation
            ax.set_xticklabels(self.metrics_labels, 
                              rotation=15,  # slight rotation
                              ha='right', 
                              fontsize=9,   # larger font
                              rotation_mode='anchor')
            
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
        
        # Increase subplot spacing
        plt.subplots_adjust(wspace=0.5)
        
        # Legend on the right
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, 
                  loc='center left', 
                  bbox_to_anchor=(0.85, 0.5),
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, bottom=0.2, top=0.85)
        
        # Save chart
        output_base = output_dir / 'all_actions_ultra_wide_complete'
        
        plt.savefig(f'{output_base}.pdf', 
                   format='pdf', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.savefig(f'{output_base}.png', 
                   format='png', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.close()
        
        print(f"Saved ultra-wide full-label chart: {output_base}.pdf/.png")
        return output_base
    
    def create_vertical_comparison_plot(self, action_data: dict, output_dir: Path):
        """Create a vertical comparison plot - full-label variant."""
        # Vertical layout gives more room for X-axis labels
        fig_width = 10.0
        fig_height = 12.0
        
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
        
        fig.suptitle('Cloth Performance Comparison Across Different Actions', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        x_positions = np.arange(len(self.metrics_columns))
        
        # Collect all cloths
        all_cloths = set()
        for cloth_metrics in action_data.values():
            all_cloths.update(cloth_metrics.keys())
        all_cloths = sorted(list(all_cloths))
        
        for idx, (action, cloth_metrics) in enumerate(action_data.items()):
            ax = axes[idx]
            
            # Plot a line per cloth
            for cloth in all_cloths:
                if cloth in cloth_metrics:
                    metrics = cloth_metrics[cloth]
                    valid_indices = ~np.isnan(metrics)
                    valid_x = x_positions[valid_indices]
                    valid_y = np.array(metrics)[valid_indices]
                    
                    if len(valid_y) > 0:
                        color = self.get_cloth_color(cloth)
                        cloth_label = cloth.replace('_', ' ').title()
                        
                        ax.plot(valid_x, valid_y, 
                               marker='o', markersize=5, linewidth=2,
                               label=cloth_label,
                               color=color,
                               alpha=0.9,
                               markerfacecolor=color,
                               markeredgecolor='white',
                               markeredgewidth=0.8)
            
            # Configure subplot attributes
            ax.set_title(f'{action.title()} Action', fontsize=14, fontweight='bold', pad=1)
            
            # Only the last subplot displays full X-axis labels
            if idx == 2:
                ax.set_xlabel('Evaluation Metrics', fontsize=12, labelpad=15)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(self.metrics_labels, 
                                  rotation=25, 
                                  ha='right', 
                                  fontsize=9)
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])  # hide labels
            
            ax.set_ylabel('Mean Values (log scale)', fontsize=11)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Show legend on the first subplot
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.8, bottom=0.15, top=0.93)
        
        # Save chart
        output_base = output_dir / 'all_actions_vertical_complete'
        
        plt.savefig(f'{output_base}.pdf', 
                   format='pdf', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.savefig(f'{output_base}.png', 
                   format='png', 
                   dpi=300, 
                   bbox_inches='tight')
        
        plt.close()
        
        print(f"Saved vertical-layout full-label chart: {output_base}.pdf/.png")
        return output_base
    
    def create_heatmap_by_action(self, action: str, cloth_metrics: dict, output_dir: Path):
        """Create a heatmap for a single action."""
        # Build the data matrix
        cloths = sorted(cloth_metrics.keys())
        data_matrix = []
        
        for cloth in cloths:
            metrics = cloth_metrics[cloth]
            # Replace NaN values with the metric's max to keep them visible
            processed_metrics = []
            for i, val in enumerate(metrics):
                if np.isnan(val):
                    # Find the maximum across cloths for this metric
                    max_val = max([cloth_metrics[c][i] for c in cloths if not np.isnan(cloth_metrics[c][i])], default=0)
                    processed_metrics.append(max_val * 1.2)  # slightly above the max
                else:
                    processed_metrics.append(val)
            data_matrix.append(processed_metrics)
        
        data_matrix = np.array(data_matrix)
        
        # Create the figure
        fig_width = 10.0
        fig_height = 6.0
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Use a log transform so values span better
        log_data = np.log10(data_matrix + 1e-10)  # avoid log(0)
        
        # Build a custom colormap: yellow to red
        colors = ['#FFF3CD', '#FFE69C', '#FFD43B', '#FFC107', '#FF8F00', '#F57C00', '#E65100', '#BF360C', '#8B0000']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list('yellow_to_red', colors, N=n_bins)
        
        # Draw the heatmap
        im = ax.imshow(log_data, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Configure axis labels
        cloth_labels = [cloth.replace('_', ' ').title() for cloth in cloths]
        ax.set_xticks(np.arange(len(self.metrics_labels)))
        ax.set_yticks(np.arange(len(cloth_labels)))
        ax.set_xticklabels(self.metrics_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(cloth_labels, fontsize=9)
        
        # Configure title and axis labels
        ax.set_title(f'{action.title()} Action - Performance Heatmap\n(Log Scale, Yellow=Low, Red=High)', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Evaluation Metrics', fontsize=11, labelpad=10)
        ax.set_ylabel('Cloth Types', fontsize=11, labelpad=10)
        
        # Add value annotations
        for i in range(len(cloths)):
            for j in range(len(self.metrics_labels)):
                if not np.isnan(data_matrix[i, j]):
                    # Pick text color based on background brightness
                    bg_intensity = (log_data[i, j] - log_data.min()) / (log_data.max() - log_data.min())
                    text_color = 'white' if bg_intensity > 0.6 else 'black'
                    
                    # Show the original (non-log) value
                    value = data_matrix[i, j]
                    if value < 0.01:
                        text = f'{value:.3f}'
                    elif value < 1:
                        text = f'{value:.2f}'
                    else:
                        text = f'{value:.1f}'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=7, fontweight='bold')
        
        # Add a color bar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Log₁₀(Metric Values)', rotation=270, labelpad=20, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, right=0.95)
        
        # Save chart
        output_base = output_dir / f'{action}_heatmap_complete'
        plt.savefig(f'{output_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {output_base}.pdf/.png")
        return output_base
    
    def create_combined_heatmap(self, action_data: dict, output_dir: Path):
        """Create a combined heatmap across all actions."""
        # Collect all cloths and prepare data
        all_cloths = set()
        for cloth_metrics in action_data.values():
            all_cloths.update(cloth_metrics.keys())
        all_cloths = sorted(list(all_cloths))
        
        # Create the figure - three subplots
        fig_width = 20.0
        fig_height = 8.0
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        fig.suptitle('Performance Heatmaps Across Different Actions\n(Log Scale: Yellow=Low Performance, Red=High Performance)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Create the colormap
        colors = ['#FFF8DC', '#FFE4B5', '#FFD700', '#FFA500', '#FF8C00', '#FF6347', '#DC143C', '#B22222', '#8B0000']
        cmap = mcolors.LinearSegmentedColormap.from_list('yellow_to_red', colors, N=256)
        
        # Create one heatmap per action
        vmin, vmax = float('inf'), float('-inf')
        log_data_list = []
        
        # First compute global min/max for consistent color scale
        for action, cloth_metrics in action_data.items():
            data_matrix = []
            for cloth in all_cloths:
                if cloth in cloth_metrics:
                    metrics = cloth_metrics[cloth]
                    processed_metrics = []
                    for val in metrics:
                        if np.isnan(val):
                            processed_metrics.append(1e-6)  # replace NaN with a small value
                        else:
                            processed_metrics.append(max(val, 1e-6))
                    data_matrix.append(processed_metrics)
                else:
                    data_matrix.append([1e-6] * len(self.metrics_columns))
            
            data_matrix = np.array(data_matrix)
            log_data = np.log10(data_matrix)
            log_data_list.append(log_data)
            vmin = min(vmin, log_data.min())
            vmax = max(vmax, log_data.max())
        
        # Draw each subplot
        for idx, (action, cloth_metrics) in enumerate(action_data.items()):
            ax = axes[idx]
            log_data = log_data_list[idx]
            
            # Draw the heatmap
            im = ax.imshow(log_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            
            # Configure labels
            cloth_labels = [cloth.replace('_', ' ').title() for cloth in all_cloths]
            
            ax.set_title(f'{action.title()}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(np.arange(len(self.metrics_labels)))
            ax.set_xticklabels(self.metrics_labels, rotation=45, ha='right', fontsize=8)
            
            if idx == 0:  # only the first subplot shows y-axis labels
                ax.set_yticks(np.arange(len(cloth_labels)))
                ax.set_yticklabels(cloth_labels, fontsize=8)
                ax.set_ylabel('Cloth Types', fontsize=11, labelpad=10)
            else:
                ax.set_yticks(np.arange(len(cloth_labels)))
                ax.set_yticklabels([])
            
            # Add value annotations (only large values to avoid clutter)
            data_matrix = np.power(10, log_data)
            for i in range(len(all_cloths)):
                for j in range(len(self.metrics_labels)):
                    cloth = all_cloths[i]
                    if cloth in cloth_metrics and not np.isnan(cloth_metrics[cloth][j]):
                        value = data_matrix[i, j]
                        if value > 1e-5:  # only show non-trivial values
                            bg_intensity = (log_data[i, j] - vmin) / (vmax - vmin)
                            text_color = 'white' if bg_intensity > 0.6 else 'black'
                            
                            if value < 0.01:
                                text = f'{value:.3f}'
                            elif value < 1:
                                text = f'{value:.2f}'
                            else:
                                text = f'{value:.1f}'
                            
                            ax.text(j, i, text, ha='center', va='center', 
                                   color=text_color, fontsize=6, fontweight='bold')
        
        # Add a shared color bar
        # cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30, pad=0.02)
        cbar = fig.colorbar(im, ax=axes, 
                    shrink=0.8, 
                    aspect=40,
                    pad=0.1,
                    orientation='horizontal',  # horizontal orientation
                    location='bottom')         # placed at the bottom
        cbar.set_label('Log₁₀(Metric Values)', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, top=0.85, right=0.92)
        
        # Save chart
        output_base = output_dir / 'all_actions_heatmap_complete'
        plt.savefig(f'{output_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined heatmap: {output_base}.pdf/.png")
        return output_base
    
    def create_ultra_wide_heatmap(self, action_data: dict, output_dir: Path):
        """Create an ultra-wide heatmap with better label display."""
        # Collect all cloths
        all_cloths = set()
        for cloth_metrics in action_data.values():
            all_cloths.update(cloth_metrics.keys())
        all_cloths = sorted(list(all_cloths))
        
        # Create the ultra-wide figure
        fig_width = 24.0
        fig_height = 8.0
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        fig.suptitle('Comprehensive Performance Heatmap Analysis\n(Logarithmic Scale: Yellow indicates Lower Values, Red indicates Higher Values)', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # High-quality colormap
        colors = ['#FFFACD', '#FFFF99', '#FFD700', '#FFBF00', '#FF8C00', '#FF6347', '#FF4500', '#DC143C', '#8B0000']
        cmap = mcolors.LinearSegmentedColormap.from_list('academic_yellow_red', colors, N=512)
        
        # Compute the global range
        all_log_data = []
        for action, cloth_metrics in action_data.items():
            data_matrix = []
            for cloth in all_cloths:
                if cloth in cloth_metrics:
                    metrics = cloth_metrics[cloth]
                    processed_metrics = [max(val, 1e-8) if not np.isnan(val) else 1e-8 for val in metrics]
                    data_matrix.append(processed_metrics)
                else:
                    data_matrix.append([1e-8] * len(self.metrics_columns))
            
            log_data = np.log10(np.array(data_matrix))
            all_log_data.append(log_data)
        
        vmin = min(data.min() for data in all_log_data)
        vmax = max(data.max() for data in all_log_data)
        
        # Draw subplots
        for idx, (action, cloth_metrics) in enumerate(action_data.items()):
            ax = axes[idx]
            log_data = all_log_data[idx]
            
            im = ax.imshow(log_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            
            # Configure full labels
            cloth_labels = [cloth.replace('_', ' ').title() for cloth in all_cloths]
            
            ax.set_title(f'{action.title()} Action', fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(np.arange(len(self.metrics_labels)))
            ax.set_xticklabels(self.metrics_labels, rotation=35, ha='right', fontsize=10)
            ax.set_xlabel('Evaluation Metrics', fontsize=12, labelpad=15)
            
            if idx == 0:
                ax.set_yticks(np.arange(len(cloth_labels)))
                ax.set_yticklabels(cloth_labels, fontsize=10)
                ax.set_ylabel('Cloth Types', fontsize=12, labelpad=15)
            else:
                ax.set_yticks(np.arange(len(cloth_labels)))
                ax.set_yticklabels([])
            
            # Grid lines
            ax.set_xticks(np.arange(len(self.metrics_labels)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(cloth_labels)) - 0.5, minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.7)
        
        # High-quality color bar
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=40, pad=0.02)
        cbar.set_label('Log₁₀ (Metric Values)', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)
        
        # Add caption text
        fig.text(0.02, 0.02, 'Note: Lower values indicate better performance for all metrics', 
                fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85, right=0.90, left=0.08)
        
        # Save
        output_base = output_dir / 'all_actions_ultra_wide_heatmap'
        plt.savefig(f'{output_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ultra-wide heatmap: {output_base}.pdf/.png")
        return output_base

    def generate_all_visualizations(self):
        """Generate every full-label visualization."""
        print("Generating conference-grade full-label visualizations...")
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare visualization data
        action_data = self.prepare_visualization_data(df)
        
        # Create output directory
        output_dir = Path("conference_quality_charts_complete")
        output_dir.mkdir(exist_ok=True)
        
        # Generate single-action charts
        print("Generating full-label single-action performance charts...")
        for action, cloth_metrics in action_data.items():
            self.create_single_action_plot(action, cloth_metrics, output_dir)
        
        # Generate full-label combined comparison plot
        print("Generating full-label combined comparison chart...")
        self.create_combined_comparison_plot(action_data, output_dir)
        
        # Generate ultra-wide version
        print("Generating ultra-wide full-label chart...")
        self.create_ultra_wide_comparison_plot(action_data, output_dir)
        
        # Generate vertical-layout comparison plot
        print("Generating vertical-layout full-label chart...")
        self.create_vertical_comparison_plot(action_data, output_dir)
        
        # New: generate heatmaps
        print("Generating single-action heatmaps...")
        for action, cloth_metrics in action_data.items():
            self.create_heatmap_by_action(action, cloth_metrics, output_dir)
        
        print("Generating combined heatmap...")
        self.create_combined_heatmap(action_data, output_dir)
        
        print("Generating ultra-wide heatmap...")
        self.create_ultra_wide_heatmap(action_data, output_dir)
        
        print(f"\nFull-label visualization complete.")
        print(f"Charts saved to: {output_dir}")
        print(f"Full-label highlights:")
        print(f"   Preserves every original full label")
        print(f"   Significantly larger canvas")
        print(f"   Tuned font size and rotation")
        print(f"   Larger margins and spacing")
        print(f"   Multiple layout options provided")
        print(f"   Conference-grade quality")
        print(f"   New: heatmap visualizations")
        
        return output_dir
    
    # def generate_all_visualizations(self):
    #     """Generate every full-label visualization."""
    #     print("Generating conference-grade full-label visualizations...")
        
    #     # Load data
    #     df = self.load_data()
    #     if df is None:
    #         return
        
    #     # Prepare visualization data
    #     action_data = self.prepare_visualization_data(df)
        
    #     # Create output directory
    #     output_dir = Path("conference_quality_charts_complete")
    #     output_dir.mkdir(exist_ok=True)
        
    #     # Generate single-action charts
    #     print("Generating full-label single-action performance charts...")
    #     for action, cloth_metrics in action_data.items():
    #         self.create_single_action_plot(action, cloth_metrics, output_dir)
        
    #     # Generate full-label combined comparison plot
    #     print("Generating full-label combined comparison chart...")
    #     self.create_combined_comparison_plot(action_data, output_dir)
        
    #     # Generate ultra-wide version
    #     print("Generating ultra-wide full-label chart...")
    #     self.create_ultra_wide_comparison_plot(action_data, output_dir)
        
    #     # Generate vertical-layout comparison plot
    #     print("Generating vertical-layout full-label chart...")
    #     self.create_vertical_comparison_plot(action_data, output_dir)
        
    #     print(f"\nFull-label visualization complete.")
    #     print(f"Charts saved to: {output_dir}")
    #     print(f"Full-label highlights:")
    #     print(f"   Preserves every original full label")
    #     print(f"   Significantly larger canvas")
    #     print(f"   Tuned font size and rotation")
    #     print(f"   Larger margins and spacing")
    #     print(f"   Multiple layout options provided")
    #     print(f"   Conference-grade quality")
        
    #     return output_dir


def main():
    """Main entrypoint."""
    visualizer = ConferenceQualityVisualizerComplete()
    
    # Make sure the Excel file exists
    if not Path(visualizer.excel_file).exists():
        print(f"Excel file does not exist: {visualizer.excel_file}")
        print("Please run simple_metrics_stats.py first to generate the data file")
        return
    
    try:
        # Generate every visualization
        output_dir = visualizer.generate_all_visualizations()
        
        print(f"\nFull-label output files:")
        print(f"   Line chart - combined: all_actions_comparison_complete.pdf/.png")
        print(f"   Line chart - ultra wide: all_actions_ultra_wide_complete.pdf/.png")
        print(f"   Line chart - vertical: all_actions_vertical_complete.pdf/.png")
        print(f"   Heatmap - combined: all_actions_heatmap_complete.pdf/.png")
        print(f"   Heatmap - ultra wide: all_actions_ultra_wide_heatmap.pdf/.png")
        # print(f"   Heatmap - single action: {{action}}_heatmap_complete.pdf/.png")
        
        print(f"\nRecommended use:")
        print(f"   - paper figures: ultra-wide heatmap or line chart")
        print(f"   - slides: combined heatmap")
        print(f"   - detailed analysis: single-action heatmaps")
        print(f"   - data comparison: heatmaps reveal value differences best")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()