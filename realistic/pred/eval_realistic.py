import re
import json


def extract_answer(text):
    """Extract the boxed answer (\\boxed{...}) from the generated text."""
    match = re.search(r'\\boxed{(\d+)}', text)
    return int(match.group(1)) if match else None


def check_incorrect_questions(unprocessed_dataset):
    incorrect_questions = []

    for example in unprocessed_dataset:
        question_id = example["id"]
        op = example["op"]  # Include operation type
        correct_answer = int(
            re.search(r"Answer: (\d+)", example["solution"]).group(1))

        incorrect = False  # Track whether this question is incorrect

        for reply in example["replies"]:
            generated_answer = extract_answer(reply)
            if generated_answer is None or generated_answer != correct_answer:
                incorrect = True  # Mark as incorrect
                break

        if incorrect:
            incorrect_questions.append((op, question_id))

    return incorrect_questions

if __name__ == '__main__':
    # Load dataset
    dir_name = "datasets"
    file_path = f"{dir_name}/ops_test-llama-3.1-8b-instruct_0"

    with open(file_path, 'r') as f:
        unprocessed_dataset = json.load(f)

    incorrect_questions = check_incorrect_questions(unprocessed_dataset)

    # Save results
    result_file = "results/ops_test-llama-3.1-8b-instruct_0-incorrect-ids.txt"
    with open(result_file, "w") as f:
        for op, question_id in incorrect_questions:
            f.write(f"op: {op}, question ID: {question_id}\n")

    print(f"Incorrect question IDs with ops saved to {result_file}")
