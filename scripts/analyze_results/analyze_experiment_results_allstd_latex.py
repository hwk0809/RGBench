import pandas as pd
import os
from pathlib import Path

def generate_final_latex_table(comparison_df, cloths, actions, environments, env_display=None, ours_env="garment_dynamics"):
    # Keep only these four metrics, in the same order as the figure
    metrics_columns = [
        'chamfer_l1_sim_to_real',    # CD(S2R)
        'chamfer_l1_real_to_sim',    # CD(R2S)
        'one_sided_hausdorff_sim_to_real', # HD(S2R)
        'one_sided_hausdorff_real_to_sim'  # HD(R2S)
    ]
    metric_display = {
        'chamfer_l1_sim_to_real': r'CD$_{s2r}$',
        'chamfer_l1_real_to_sim': r'CD$_{r2s}$',
        'one_sided_hausdorff_sim_to_real': r'HD$_{s2r}$',
        'one_sided_hausdorff_real_to_sim': r'HD$_{r2s}$'
    }
    if env_display is None:
        env_display = {e: e for e in environments}

    def env_order(envs, ours):
        envs = [e for e in envs if e != ours]
        return envs + [ours]

    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{3.5pt}")
    latex.append(r"\begin{tabular}{l l | *{3}{c} | *{3}{c} | *{3}{c} | *{3}{c}}")
    latex.append(r"\toprule")
    # First-level header
    header1 = [r"\multirow{2}{*}{Cloth}", r"\multirow{2}{*}{Action}"]
    for metric in metrics_columns:
        header1.append(r"\multicolumn{3}{c|}{\textbf{" + metric_display[metric] + "}}")
    latex.append(" & ".join(header1) + r" \\")
    # Second-level header
    header2 = ["", ""]
    for metric in metrics_columns:
        envs = env_order(environments, ours_env)
        header2.extend([env_display[e] if e != ours_env else "ours" for e in envs])
    latex.append(r"\noalign{\vspace{1.5pt}}")
    latex.append(r"\cline{3-14}")
    latex.append(r"\noalign{\vspace{2pt}}")
    latex.append(" & ".join(header2) + r" \\")
    latex.append(r"\midrule")

    for cloth in cloths:
        for action in actions:
            row = [cloth.replace('_', ' ').title(), action.title()]
            for metric in metrics_columns:
                envs = env_order(environments, ours_env)
                means = []
                values = []
                for env in envs:
                    df_row = comparison_df[
                        (comparison_df['cloth'] == cloth) &
                        (comparison_df['action'] == action) &
                        (comparison_df['environment'] == env)
                    ]
                    mean_col = f"{metric}_mean"
                    if len(df_row) > 0 and pd.notna(df_row.iloc[0][mean_col]):
                        mean = df_row.iloc[0][mean_col]
                        val_str = f"{mean:.4f}"
                        means.append(mean)
                        values.append(val_str)
                    else:
                        values.append("-")
                        means.append(float('inf'))
                # Bold the best value
                if any(v != "-" for v in values):
                    min_idx = int(pd.Series(means).idxmin())
                    values[min_idx] = r"\textbf{" + values[min_idx] + "}"
                row.extend(values)
            latex.append(" & ".join(row) + r" \\")
        latex.append(r"\midrule")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{`ours` is moved to the last column of each metric; the minimum value is bolded.}")
    latex.append(r"\label{tab:ours_last_column}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)

if __name__ == "__main__":
    from analyze_experiment_results_allstd import VerticalEnvironmentComparisonAnalyzer
    import os

    # Locate the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, "outputs")

    # Important: pass outputs_dir
    analyzer = VerticalEnvironmentComparisonAnalyzer(outputs_dir=outputs_dir, mode="fixed_point")
    _, comparison_df, _, _, _ = analyzer.create_excel_report()

    cloths = analyzer.cloths
    actions = analyzer.actions
    environments = ["pybullet", "isaacsim", "garment_dynamics"]
    env_display = {
        "pybullet": "pybullet",
        "isaacsim": "isaacsim",
        "garment_dynamics": "ours"
    }

    latex_code = generate_final_latex_table(
        comparison_df, cloths, actions, environments, env_display, ours_env="garment_dynamics"
    )

    save_dir = Path(__file__).parent.parent / "outputs/analysis/latex"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "env_comparison_table_flat.tex"

    with open(save_path, "w") as f:
        f.write(latex_code)
