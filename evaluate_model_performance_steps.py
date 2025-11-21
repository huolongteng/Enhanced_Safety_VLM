"""
A step-by-step evaluation script for comparing model predictions against the
provided test dataset labels. This file intentionally avoids function and class
definitions so each step can be read and modified directly.
"""

# step-1: import standard libraries and set the paths for ground-truth labels
# and model predictions. Update ``MODEL_OUTPUT_PATH`` to point to either your
# student or teacher model's JSON output before running the script.
import csv
import json
from pathlib import Path

TEST_CSV_PATH = Path("data/test.csv")
MODEL_OUTPUT_PATH = Path("data/model_outputs/student_model_outputs.json")  # replace with your model's output path


# step-2: load the labeled test dataset from ``data/test.csv`` and extract the
# "rating" and "category" fields for each entry. Labels are normalized to
# lower-case strings so that trivial casing differences do not affect matching.
truth_by_id = {}
with TEST_CSV_PATH.open("r", encoding="utf-8", newline="") as test_file:
    reader = csv.DictReader(test_file)
    for row in reader:
        sample_id = row.get("id")
        if not sample_id:
            continue
        rating_label = str(row.get("rating", "")).strip().lower()
        category_label = str(row.get("category", "")).strip().lower()
        truth_by_id[sample_id] = {
            "label": (rating_label, category_label),
            "raw": {"rating": rating_label, "category": category_label},
        }


# step-3: load the model predictions. The script expects each prediction entry
# to include an identifier matching the test dataset and an "output" payload
# containing "rating" and "category" in the same JSON format as the labels.
with MODEL_OUTPUT_PATH.open("r", encoding="utf-8") as prediction_file:
    prediction_entries = json.load(prediction_file)


def _extract_assistant_json(raw_prediction: str):
    """Return the assistant portion of a conversation-style prediction string.

    Saved outputs contain a user prompt followed by the assistant reply. The
    reply always includes a JSON object (``{...}``), so we extract the first
    balanced block starting from the initial ``{`` to ensure we parse the
    model's structured output even if other text follows. If the extraction
    fails, an empty dictionary is returned.
    """

    assistant_segment = raw_prediction
    if "assistant" in raw_prediction:
        assistant_segment = raw_prediction.split("assistant", 1)[1]

    start = assistant_segment.find("{")
    if start == -1:
        try:
            return json.loads(raw_prediction)
        except json.JSONDecodeError:
            return {}

    depth = 0
    end = None
    for idx in range(start, len(assistant_segment)):
        char = assistant_segment[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is not None:
        try:
            return json.loads(assistant_segment[start : end + 1])
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(raw_prediction)
    except json.JSONDecodeError:
        return {}


pred_by_id = {}
rating_distribution = {}
category_distribution = {}
rating_members = {}
category_members = {}
for record in prediction_entries:
    raw_prediction = record.get("output") or record.get("model_output", {})
    if isinstance(raw_prediction, str):
        parsed_prediction = _extract_assistant_json(raw_prediction)
    else:
        parsed_prediction = raw_prediction

    rating_prediction = str(parsed_prediction.get("rating", "")).strip().lower()
    category_prediction = str(parsed_prediction.get("category", "")).strip().lower()
    pred_by_id[record.get("id")] = {
        "label": (rating_prediction, category_prediction),
        "raw": parsed_prediction,
    }

    if rating_prediction:
        rating_distribution[rating_prediction] = rating_distribution.get(
            rating_prediction, 0
        ) + 1
        rating_members.setdefault(rating_prediction, []).append(record.get("id"))

    if category_prediction:
        category_distribution[category_prediction] = category_distribution.get(
            category_prediction, 0
        ) + 1
        category_members.setdefault(category_prediction, []).append(record.get("id"))


# step-4: align predictions with ground-truth labels by ``id`` and accumulate
# counts needed for accuracy, precision, recall, F1, and F2 calculations.
correct_predictions = 0
considered_examples = 0
missing_predictions = []
per_label_counts = {}

rating_correct = 0
category_correct = 0

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

    if truth_label[0] == predicted_label[0]:
        rating_correct += 1
    if truth_label[1] == predicted_label[1]:
        category_correct += 1

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

# macro-averaged metrics across all observed labels
macro_precisions = []
macro_recalls = []
macro_f1s = []
for counts in per_label_counts.values():
    label_precision_den = counts["tp"] + counts["fp"]
    label_recall_den = counts["tp"] + counts["fn"]
    label_precision = counts["tp"] / label_precision_den if label_precision_den else 0.0
    label_recall = counts["tp"] / label_recall_den if label_recall_den else 0.0
    label_f1_den = label_precision + label_recall
    label_f1 = (2 * label_precision * label_recall / label_f1_den) if label_f1_den else 0.0

    # only include labels that were either predicted or present in the truth set
    if label_precision_den or label_recall_den:
        macro_precisions.append(label_precision)
        macro_recalls.append(label_recall)
        macro_f1s.append(label_f1)

macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

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
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall   : {macro_recall:.4f}")
print(f"Macro F1 Score : {macro_f1:.4f}")
print(
    f"Rating accuracy: {rating_correct / considered_examples:.4f}" if considered_examples else "Rating accuracy: 0.0000"
)
print(
    f"Category accuracy: {category_correct / considered_examples:.4f}" if considered_examples else "Category accuracy: 0.0000"
)

print("\nPrediction rating distribution (normalized to lowercase):")
for rating_value, count in sorted(rating_distribution.items()):
    ids = ", ".join(str(i) for i in rating_members.get(rating_value, []))
    print(f" - rating='{rating_value}': {count} -> ids: [{ids}]")

print("\nPrediction category distribution (normalized to lowercase):")
for category_value, count in sorted(category_distribution.items()):
    ids = ", ".join(str(i) for i in category_members.get(category_value, []))
    print(f" - category='{category_value}': {count} -> ids: [{ids}]")

print("\nPer-label support and errors (tp/fp/fn):")
for label, counts in sorted(per_label_counts.items()):
    rating_value, category_value = label
    print(
        f"[{rating_value} | {category_value}] -> support={counts['support']}, "
        f"tp={counts['tp']}, fp={counts['fp']}, fn={counts['fn']}"
    )
