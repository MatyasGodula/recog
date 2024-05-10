def read_dsv_file(filename) -> dict:
    dictionary = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Properly strip whitespace and assign back to line
            if line:  # Check if line is not empty
                key, value = line.split(':')
                dictionary[key] = value.strip()  # Strip any extra spaces around the value
    return dictionary

def find_similarities_and_precision(truth_dict, classification_dict):
    ok = 0
    not_ok = 0
    for fname_classification, class_value in classification_dict.items():
        if fname_classification in truth_dict:
            truth_value = truth_dict[fname_classification]
            if class_value != truth_value:
                not_ok += 1
                print(f"Mismatch: {fname_classification}, Truth: {truth_value}, Classified: {class_value}")
            else:
                ok += 1

    total = ok + not_ok
    if total > 0:
        accuracy = (ok / total) * 100
        print(f"The accuracy is {accuracy:.2f}%")
    else:
        print("No data to evaluate accuracy.")

file_path_truth = 'train_path/truth.dsv'
file_path_classification = 'classification.dsv'

dictionary_truth = read_dsv_file(file_path_truth)
dictionary_classification = read_dsv_file(file_path_classification)

find_similarities_and_precision(dictionary_truth, dictionary_classification)