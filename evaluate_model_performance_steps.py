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

from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support

TEST_CSV_PATH = Path("data/test.csv")
MODEL_OUTPUT_PATH = Path("data/model_outputs/teacher_model_outputs.json")  # replace with your model's output path


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


def _extract_rating_and_category(raw_text: str):
    """Parse rating/category JSON that appears after the "assistant" keyword."""

    if not isinstance(raw_text, str):
        return None

    lower_bound = raw_text.find("assistant")
    if lower_bound == -1:
        return None

    assistant_tail = raw_text[lower_bound + len("assistant") :]
    start = assistant_tail.find("{")
    if start == -1:
        return None

    depth = 0
    end = None
    for idx, char in enumerate(assistant_tail[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is None:
        return None

    try:
        parsed = json.loads(assistant_tail[start : end + 1])
    except json.JSONDecodeError:
        return None

    rating = str(parsed.get("rating", "")).strip().lower()
    category = str(parsed.get("category", "")).strip().lower()
    if rating and category:
        return rating, category

    return None


pred_by_id = {}
for record in prediction_entries:
    raw_prediction = record.get("output") or record.get("model_output", {})
    parsed_prediction = _extract_rating_and_category(raw_prediction)
    if parsed_prediction is None:
        pred_by_id[record.get("id")] = None
    else:
        rating_prediction, category_prediction = parsed_prediction
        pred_by_id[record.get("id")] = {
            "label": (rating_prediction, category_prediction),
            "raw": {"rating": rating_prediction, "category": category_prediction},
        }


# step-4: align predictions with ground-truth labels by ``id`` and accumulate
# counts needed for accuracy, precision, recall, F1, and F2 calculations.
considered_examples = 0
missing_predictions = []
y_true_rating = []
y_pred_rating = []
y_true_category = []
y_pred_category = []

for sample_id, truth in truth_by_id.items():
    prediction = pred_by_id.get(sample_id)
    if not prediction:
        missing_predictions.append(sample_id)
        continue

    truth_rating, truth_category = truth["label"]
    predicted_rating, predicted_category = prediction["label"]

    if not (truth_rating and truth_category and predicted_rating and predicted_category):
        missing_predictions.append(sample_id)
        continue

    considered_examples += 1
    y_true_rating.append(truth_rating)
    y_pred_rating.append(predicted_rating)
    y_true_category.append(truth_category)
    y_pred_category.append(predicted_category)


# step-5: compute aggregate metrics per field with macro averaging
if considered_examples:
    rating_accuracy = accuracy_score(y_true_rating, y_pred_rating)
    category_accuracy = accuracy_score(y_true_category, y_pred_category)

    rating_precision, rating_recall, rating_f1, _ = precision_recall_fscore_support(
        y_true_rating, y_pred_rating, average="macro", zero_division=0
    )
    category_precision, category_recall, category_f1, _ = precision_recall_fscore_support(
        y_true_category, y_pred_category, average="macro", zero_division=0
    )

    rating_f2 = fbeta_score(
        y_true_rating, y_pred_rating, beta=2, average="macro", zero_division=0
    )
    category_f2 = fbeta_score(
        y_true_category, y_pred_category, beta=2, average="macro", zero_division=0
    )
else:
    rating_accuracy = rating_precision = rating_recall = rating_f1 = rating_f2 = 0.0
    category_accuracy = (
        category_precision
    ) = category_recall = category_f1 = category_f2 = 0.0


# step-6: print the evaluation summary to the terminal, including any ids that
# were missing predictions to help with troubleshooting.
print("===== Evaluation Summary =====")
print(f"Evaluated examples: {considered_examples} / {len(truth_by_id)}")
print(f"Missing predictions: {len(missing_predictions)}")
if missing_predictions:
    print("IDs with missing predictions:")
    for sample_id in missing_predictions:
        print(f" - {sample_id}")

print("\nRating metrics")
print(f"Accuracy : {rating_accuracy:.4f}")
print(f"Precision: {rating_precision:.4f}")
print(f"Recall   : {rating_recall:.4f}")
print(f"F1 Score : {rating_f1:.4f}")
print(f"F2 Score : {rating_f2:.4f}")

print("\nCategory metrics")
print(f"Accuracy : {category_accuracy:.4f}")
print(f"Precision: {category_precision:.4f}")
print(f"Recall   : {category_recall:.4f}")
print(f"F1 Score : {category_f1:.4f}")
print(f"F2 Score : {category_f2:.4f}")
