"""Run reproducible experiments and save report artifacts without Streamlit."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import trainer

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)
experiments = [
    {"name":"Baseline large model", "hidden":256, "dropout":0.0, "weight_decay":0.0, "lr":0.001, "epochs":160, "patience":0, "scale":True, "seed":0},
    {"name":"Dropout and weight decay", "hidden":256, "dropout":0.3, "weight_decay":0.05, "lr":0.001, "epochs":160, "patience":0, "scale":True, "seed":0},
    {"name":"Early stopping", "hidden":256, "dropout":0.3, "weight_decay":0.05, "lr":0.001, "epochs":160, "patience":20, "scale":True, "seed":0},
    {"name":"Smaller model with regularisation", "hidden":16, "dropout":0.3, "weight_decay":0.05, "lr":0.001, "epochs":160, "patience":20, "scale":True, "seed":0},
]
rows, histories = [], []
data = trainer.load_data(scale=True)
for cfg in experiments:
    name = cfg.pop("name")
    model, hist, summary = trainer.train(cfg, data)
    result = trainer.evaluate(model, data[4], data[5])
    rows.append({"experiment":name, **cfg, **summary,
                 "test_accuracy":result["accuracy"],
                 "cancers_found":result["cancers_found"],
                 "false_alarms":result["false_alarms"]})
    histories.append((name, hist))
    cfg["name"] = name

pd.DataFrame(rows).to_csv(OUT/"experiments.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, hist) in zip(axes.flat, histories):
    ax.plot(hist["train"], label="Training loss")
    ax.plot(hist["val"], label="Validation loss")
    ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.legend(); ax.grid(alpha=.2)
fig.suptitle("City Hospital: training and validation curves", fontsize=15)
fig.tight_layout()
fig.savefig(OUT/"training_curves.png", dpi=160)
with open(OUT/"REPORT.md","w",encoding="utf-8") as f:
    f.write("# Experiment report\\n\\n")
    f.write(pd.DataFrame(rows).to_markdown(index=False))
    f.write("\\n\\n## Recommendation\\nUse validation loss and early stopping to select a model. This is a course demonstration, not a clinically validated screening system.\\n")
print(pd.DataFrame(rows).to_string(index=False))
print("Saved artifacts/experiments.csv, training_curves.png and REPORT.md")
