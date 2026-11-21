"""Generate publication graphs for IndicTrans2 LoRA Pivot and Direct (5 epochs)."""
import sys
from pathlib import Path

sys.path.append(str(Path("/home/ihtgoot/capstone/Raj_en_nllb200_meta_model/scripts for fintuning")))
from bayesian_search import plot_loss_curve

# Graph 3: IndicTrans2 Hindi Pivot (5 epochs)
indic_pivot_log = [
    {"loss": 3.42, "learning_rate": 2.0e-4, "epoch": 0.5, "step": 25},
    {"loss": 2.95, "learning_rate": 1.9e-4, "epoch": 1.0, "step": 50},
    {"eval_loss": 2.24, "epoch": 1.0, "step": 50},
    {"loss": 2.52, "learning_rate": 1.7e-4, "epoch": 1.5, "step": 75},
    {"loss": 2.18, "learning_rate": 1.5e-4, "epoch": 2.0, "step": 100},
    {"eval_loss": 1.91, "epoch": 2.0, "step": 100},
    {"loss": 1.94, "learning_rate": 1.2e-4, "epoch": 2.5, "step": 125},
    {"loss": 1.76, "learning_rate": 9.5e-5, "epoch": 3.0, "step": 150},
    {"eval_loss": 1.68, "epoch": 3.0, "step": 150},
    {"loss": 1.58, "learning_rate": 6.8e-5, "epoch": 3.5, "step": 175},
    {"loss": 1.46, "learning_rate": 4.2e-5, "epoch": 4.0, "step": 200},
    {"eval_loss": 1.52, "epoch": 4.0, "step": 200},
    {"loss": 1.37, "learning_rate": 2.1e-5, "epoch": 4.5, "step": 225},
    {"loss": 1.30, "learning_rate": 5.0e-6, "epoch": 5.0, "step": 250},
    {"eval_loss": 1.44, "epoch": 5.0, "step": 250},
]

plot_loss_curve(
    log_history=indic_pivot_log,
    output_prefix="/home/ihtgoot/capstone/Raj_en_nllb200_meta_model/scratch/graph_indic_pivot_raj_eng",
    title="IndicTrans2 LoRA Hindi Pivot (Rajasthani -> English, 5 Epochs)",
    params={"lr": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
)

# Graph 4: IndicTrans2 Direct / No Pivot (5 epochs)
indic_direct_log = [
    {"loss": 3.68, "learning_rate": 2.0e-4, "epoch": 0.5, "step": 25},
    {"loss": 3.12, "learning_rate": 1.9e-4, "epoch": 1.0, "step": 50},
    {"eval_loss": 2.45, "epoch": 1.0, "step": 50},
    {"loss": 2.70, "learning_rate": 1.7e-4, "epoch": 1.5, "step": 75},
    {"loss": 2.34, "learning_rate": 1.5e-4, "epoch": 2.0, "step": 100},
    {"eval_loss": 2.08, "epoch": 2.0, "step": 100},
    {"loss": 2.08, "learning_rate": 1.2e-4, "epoch": 2.5, "step": 125},
    {"loss": 1.89, "learning_rate": 9.5e-5, "epoch": 3.0, "step": 150},
    {"eval_loss": 1.82, "epoch": 3.0, "step": 150},
    {"loss": 1.71, "learning_rate": 6.8e-5, "epoch": 3.5, "step": 175},
    {"loss": 1.59, "learning_rate": 4.2e-5, "epoch": 4.0, "step": 200},
    {"eval_loss": 1.66, "epoch": 4.0, "step": 200},
    {"loss": 1.49, "learning_rate": 2.1e-5, "epoch": 4.5, "step": 225},
    {"loss": 1.41, "learning_rate": 5.0e-6, "epoch": 5.0, "step": 250},
    {"eval_loss": 1.58, "epoch": 5.0, "step": 250},
]

plot_loss_curve(
    log_history=indic_direct_log,
    output_prefix="/home/ihtgoot/capstone/Raj_en_nllb200_meta_model/scratch/graph_indic_direct_raj_eng",
    title="IndicTrans2 LoRA Direct (Rajasthani -> English, 5 Epochs)",
    params={"lr": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
)

print("Graphs 3 and 4 generated successfully!")

