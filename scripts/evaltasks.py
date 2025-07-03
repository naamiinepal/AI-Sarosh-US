import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json
import logging
from get_best import get_latest_run_dir, get_best_checkpoint

tasks=['us_lie','us_plac']
logging.basicConfig(
    filename='/mnt/Enterprise2/AI-Sarosh/code-repo/logs/eval_q.log',            # log file name
    level=logging.DEBUG,                 # log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # log format
)
# --- Utility Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_entropy(p):
    return - (p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))

# --- Load CSV ---


for task in tasks:
    op_dir = f"/mnt/Enterprise2/AI-Sarosh/code-repo/outputs/mvit-trained-on-turbo-quality-{task}/eval/runs"
    latest_dir = get_latest_run_dir(op_dir)
    csv_path = f"{latest_dir}/predictions/predictions.csv"

    logging.info(f"..................{csv_path}.......................")

    df = pd.read_csv(csv_path)
    # Ensure logits are float and compute probabilities
    df["logit"] = df["logit"].astype(float)
    df["prob"] = df["output"]
    # --- Storage for patient-level results ---
    results = []
    # --- Group by study (patient) ---
    for study_id, group in df.groupby("study_id"):
        logits = group["logit"].to_numpy()
        probs = sigmoid(logits)
        gt = group["gt"].iloc[0]
        # Strategy 1: Uniform average
        uniform_avg = probs.mean()
        # Strategy 2: Confidence-weighted average (|prob - 0.5|)
        confidences = np.abs(probs - 0.5)
        weights_conf = confidences / (confidences.sum() + 1e-9)
        conf_weighted = np.average(probs, weights=weights_conf)
        # Strategy 3: Entropy-weighted
        entropies = binary_entropy(probs)
        inv_entropy = 1 / (entropies + 1e-6)
        weights_entropy = inv_entropy / inv_entropy.sum()
        entropy_weighted = np.average(probs, weights=weights_entropy)
        # Strategy 4: Max-confidence prediction
        max_conf_idx = confidences.argmax()
        max_conf = probs[max_conf_idx]
        # Strategy 5: Top-3 confidence average
        top_k = min(3, len(probs))
        topk_idx = np.argsort(confidences)[-top_k:]
        top3_avg = probs[topk_idx].mean()
        # Strategy 6: Logit-magnitude weighted average
        abs_logits = np.abs(logits)
        logit_weights = abs_logits / (abs_logits.sum() + 1e-9)
        logit_weighted = np.average(probs, weights=logit_weights)
        # Final predictions (binary)
        preds = {
            "pred_uniform": int(uniform_avg >= 0.5),
            "pred_conf_weighted": int(conf_weighted >= 0.5),
            "pred_entropy_weighted": int(entropy_weighted >= 0.5),
            "pred_max_conf": int(max_conf >= 0.5),
            "pred_top3_avg": int(top3_avg >= 0.5),
            "pred_logit_weighted": int(logit_weighted >= 0.5)
        }
        results.append({
            "study_id": study_id,
            "gt": gt,
            "logits_all": logits.tolist(),
            "uniform_avg": uniform_avg,
            "conf_weighted": conf_weighted,
            "entropy_weighted": entropy_weighted,
            "max_conf": max_conf,
            "top3_avg": top3_avg,
            "logit_weighted": logit_weighted,
            **preds
        })
    # --- Serialize embed vectors ---
    for r in results:
        for key in r:
            if key.endswith("_embed"):
                r[key] = json.dumps(r[key].tolist())
    # Create dataframe and save
    output_df = pd.DataFrame(results)
    output_df.to_csv("notebooks/patient_level.csv", index=False)
    # --- Evaluate Each Strategy ---
    logging.info("\n\n📊 Evaluation Metrics per Strategy (Patient-Level)\n")
    strategies = [
        "pred_uniform",
        "pred_conf_weighted",
        "pred_entropy_weighted",
        "pred_max_conf",
        "pred_top3_avg",
        "pred_logit_weighted"
    ]
    gt = output_df["gt"].values
    for strategy in strategies:
        preds = output_df[strategy].values
        logging.info(f"🔹 Strategy: {strategy}")
        logging.info("Classification Report:")
        logging.info(classification_report(gt, preds, digits=4))
        logging.info("Confusion Matrix:")
        logging.info(confusion_matrix(gt, preds))
        logging.info("-" * 60)
    # --- Total (Per-frame) Metrics (No Grouping) ---
    logging.info("\n\n📊 Overall Per-frame Evaluation (No Grouping)\n")
    # Threshold sigmoid probabilities to get binary predictions
    df["frame_pred"] = (df["logit"] >=0).astype(int)
    frame_gt = df["gt"].values
    frame_preds = df["frame_pred"].values
    # Compute metrics
    logging.info("Classification Report (Per-frame):")
    logging.info(classification_report(frame_gt, frame_preds, digits=4))
    logging.info("Confusion Matrix (Per-frame):")
    logging.info(confusion_matrix(frame_gt, frame_preds))
    # logging.info total accuracy and F1
    total_accuracy = accuracy_score(frame_gt, frame_preds)
    total_f1 = f1_score(frame_gt, frame_preds)
    logging.info(f"✅ Total Accuracy: {total_accuracy:.4f}")
    logging.info(f"✅ Total F1-score: {total_f1:.4f}")