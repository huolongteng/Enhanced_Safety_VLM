"""
A step-by-step evaluation script for comparing model predictions against the
provided test dataset labels. This file intentionally avoids function and class
definitions so each step can be read and modified directly.
"""

# step-1: import standard libraries and set the paths for ground-truth labels
# and model predictions. Update ``MODEL_OUTPUT_PATH`` to point to either your
# student or teacher model's JSON output before running the script.
import json
from pathlib import Path

TEST_JSON_PATH = Path("data/test_dataset.json")
MODEL_OUTPUT_PATH = Path("data/model_outputs/model_outputs.json")  # replace with your model's output path


# step-2: load the labeled test dataset and extract the "rating" and "category"
# fields for each entry. Labels are normalized to lower-case strings so that
# trivial casing differences do not affect matching.
with TEST_JSON_PATH.open("r", encoding="utf-8") as test_file:
    test_entries = json.load(test_file)

truth_by_id = {}
for entry in test_entries:
    raw_output = entry.get("output", {})
    parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    rating_label = str(parsed_output.get("rating", "")).strip().lower()
    category_label = str(parsed_output.get("category", "")).strip().lower()
    truth_by_id[entry.get("id")] = {
        "label": (rating_label, category_label),
        "raw": parsed_output,
    }


# step-3: load the model predictions. The script expects each prediction entry
# to include an identifier matching the test dataset and an "output" payload
# containing "rating" and "category" in the same JSON format as the labels.
with MODEL_OUTPUT_PATH.open("r", encoding="utf-8") as prediction_file:
    prediction_entries = json.load(prediction_file)

pred_by_id = {}
for record in prediction_entries:
    raw_prediction = record.get("output") or record.get("model_output", {})
    parsed_prediction = json.loads(raw_prediction) if isinstance(raw_prediction, str) else raw_prediction
    rating_prediction = str(parsed_prediction.get("rating", "")).strip().lower()
    category_prediction = str(parsed_prediction.get("category", "")).strip().lower()
    pred_by_id[record.get("id")] = {
        "label": (rating_prediction, category_prediction),
        "raw": parsed_prediction,
    }


# step-4: align predictions with ground-truth labels by ``id`` and accumulate
# counts needed for accuracy, precision, recall, F1, and F2 calculations.
correct_predictions = 0
considered_examples = 0
missing_predictions = []
per_label_counts = {}

total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0

for sample_id, truth in truth_by_id.items():
    prediction = pred_by_id.get(sample_id)
    if prediction is None:
        missing_predictions.append(sample_id)
        continue

    considered_examples += 1
    truth_label = truth["label"]
    predicted_label = prediction["label"]

    for label in (truth_label, predicted_label):
        if label not in per_label_counts:
            per_label_counts[label] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}

    per_label_counts[truth_label]["support"] += 1

    if predicted_label == truth_label:
        correct_predictions += 1
        total_true_positives += 1
        per_label_counts[truth_label]["tp"] += 1
    else:
        total_false_positives += 1
        total_false_negatives += 1
        per_label_counts[predicted_label]["fp"] += 1
        per_label_counts[truth_label]["fn"] += 1


# step-5: compute aggregate metrics (micro-averaged over all labels) with
# safeguards against division by zero.
accuracy = correct_predictions / considered_examples if considered_examples else 0.0
precision = (
    total_true_positives / (total_true_positives + total_false_positives)
    if (total_true_positives + total_false_positives) > 0
    else 0.0
)
recall = (
    total_true_positives / (total_true_positives + total_false_negatives)
    if (total_true_positives + total_false_negatives) > 0
    else 0.0
)

f1_denominator = precision + recall
f1_score = (2 * precision * recall / f1_denominator) if f1_denominator else 0.0

beta = 2
f2_denominator = (beta ** 2 * precision) + recall
f2_score = ((1 + beta ** 2) * precision * recall / f2_denominator) if f2_denominator else 0.0


# step-6: print the evaluation summary to the terminal, including any ids that
# were missing predictions to help with troubleshooting.
print("===== Evaluation Summary =====")
print(f"Evaluated examples: {considered_examples} / {len(truth_by_id)}")
print(f"Missing predictions: {len(missing_predictions)}")
if missing_predictions:
    print("IDs with missing predictions:")
    for sample_id in missing_predictions:
        print(f" - {sample_id}")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1_score:.4f}")
print(f"F2 Score : {f2_score:.4f}")

print("\nPer-label support and errors (tp/fp/fn):")
for label, counts in sorted(per_label_counts.items()):
    rating_value, category_value = label
    print(
        f"[{rating_value} | {category_value}] -> support={counts['support']}, "
        f"tp={counts['tp']}, fp={counts['fp']}, fn={counts['fn']}"
    )
