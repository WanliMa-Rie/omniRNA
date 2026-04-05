import json
import math


FITNESS_LABEL = "Fitness Spearman"
FITNESS_R2_LABEL = "Fitness R2"
STRUCTURE_TM_LABEL = "Structure TM-score"
STRUCTURE_RMSD_LABEL = "Structure RMSD"
STRUCTURE_F1_LABEL = "Structure F1"
EVOLUTION_ACC_LABEL = "Evolution Accuracy"
EVOLUTION_F1_LABEL = "Evolution Macro-F1"
OVERALL_LABEL = "Overall"


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def mean(values):
    return sum(values) / len(values)


def average_ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            ranks[ordered[index][0]] = rank
        start = end + 1
    return ranks


def pearson(xs, ys):
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_denom * y_denom)


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def r2_score(targets, predictions):
    target_mean = mean(targets)
    residual = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    total = sum((target - target_mean) ** 2 for target in targets)
    return 1.0 - residual / total


def index_by_id(rows):
    return {row["id"]: row for row in rows}


def evaluate_fitness(annotation_rows, submission_rows):
    predictions = index_by_id(submission_rows)
    grouped = {}
    for row in annotation_rows:
        key = (row["dataset"], row["evaluation_split"])
        grouped.setdefault(key, {"targets": [], "predictions": []})
        grouped[key]["targets"].append(row["target"])
        grouped[key]["predictions"].append(predictions[row["id"]]["prediction"])

    spearman_scores = []
    r2_scores = []
    for group in grouped.values():
        spearman_scores.append(spearman(group["targets"], group["predictions"]))
        r2_scores.append(r2_score(group["targets"], group["predictions"]))

    return mean(spearman_scores), mean(r2_scores)


def pairs_to_set(pairs):
    return {tuple(pair) for pair in pairs}


def f1_from_sets(targets, predictions):
    true_positive = len(targets & predictions)
    if not targets and not predictions:
        return 1.0
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(targets) if targets else 0.0
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def tm_d0(length):
    if length <= 15:
        return 0.5
    return max(0.5, 1.24 * ((length - 15) ** (1.0 / 3.0)) - 1.8)


def center_coordinates(points):
    count = len(points)
    return [
        sum(point[dimension] for point in points) / count
        for dimension in range(3)
    ]


def subtract_center(points, center):
    return [
        [point[0] - center[0], point[1] - center[1], point[2] - center[2]]
        for point in points
    ]


def build_quaternion_matrix(reference, prediction):
    sxx = sxy = sxz = 0.0
    syx = syy = syz = 0.0
    szx = szy = szz = 0.0

    for predicted_point, reference_point in zip(prediction, reference):
        px, py, pz = predicted_point
        rx, ry, rz = reference_point
        sxx += px * rx
        sxy += px * ry
        sxz += px * rz
        syx += py * rx
        syy += py * ry
        syz += py * rz
        szx += pz * rx
        szy += pz * ry
        szz += pz * rz

    return [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]


def multiply_matrix_vector(matrix, vector):
    return [
        sum(entry * value for entry, value in zip(row, vector))
        for row in matrix
    ]


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector]


def dominant_quaternion(matrix, iterations=50):
    vector = [1.0, 0.0, 0.0, 0.0]
    for _ in range(iterations):
        vector = normalize(multiply_matrix_vector(matrix, vector))
    return vector


def quaternion_to_rotation(quaternion):
    w, x, y, z = quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def rotate_point(rotation, point):
    return [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ]


def structure_distances(reference_coords, prediction_coords):
    reference_center = center_coordinates(reference_coords)
    prediction_center = center_coordinates(prediction_coords)
    reference_shifted = subtract_center(reference_coords, reference_center)
    prediction_shifted = subtract_center(prediction_coords, prediction_center)
    quaternion_matrix = build_quaternion_matrix(reference_shifted, prediction_shifted)
    rotation = quaternion_to_rotation(dominant_quaternion(quaternion_matrix))

    distances = []
    for reference_point, predicted_point in zip(reference_shifted, prediction_shifted):
        rotated = rotate_point(rotation, predicted_point)
        difference = [
            rotated[0] - reference_point[0],
            rotated[1] - reference_point[1],
            rotated[2] - reference_point[2],
        ]
        distances.append(math.sqrt(sum(value * value for value in difference)))
    return distances


def tm_score(distances):
    d0 = tm_d0(len(distances))
    return mean([1.0 / (1.0 + (distance / d0) ** 2) for distance in distances])


def rmsd(distances):
    return math.sqrt(mean([distance ** 2 for distance in distances]))


def evaluate_structure(annotation_rows, submission_rows):
    predictions = index_by_id(submission_rows)
    tm_scores = []
    rmsd_scores = []
    f1_scores = []

    for row in annotation_rows:
        prediction = predictions[row["id"]]
        distances = structure_distances(row["tertiary_coords"], prediction["tertiary_coords"])
        tm_scores.append(tm_score(distances))
        rmsd_scores.append(rmsd(distances))
        f1_scores.append(
            f1_from_sets(
                pairs_to_set(row["secondary_pairs"]),
                pairs_to_set(prediction["secondary_pairs"]),
            )
        )

    return mean(tm_scores), mean(rmsd_scores), mean(f1_scores)


def accuracy(targets, predictions):
    matches = [target == prediction for target, prediction in zip(targets, predictions)]
    return sum(matches) / len(matches)


def class_f1(targets, predictions, label):
    true_positive = sum(
        target == label and prediction == label
        for target, prediction in zip(targets, predictions)
    )
    false_positive = sum(
        target != label and prediction == label
        for target, prediction in zip(targets, predictions)
    )
    false_negative = sum(
        target == label and prediction != label
        for target, prediction in zip(targets, predictions)
    )
    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def macro_f1(targets, predictions):
    labels = sorted(set(targets) | set(predictions))
    return mean([class_f1(targets, predictions, label) for label in labels])


def evaluate_evolution(annotation_block, submission_block):
    quartet_predictions = index_by_id(submission_block["quartet"])
    quartet_targets = [row["label"] for row in annotation_block["quartet"]]
    quartet_outputs = [quartet_predictions[row["id"]]["prediction"] for row in annotation_block["quartet"]]

    covariation_predictions = index_by_id(submission_block["covariation"])
    covariation_targets = [row["label"] for row in annotation_block["covariation"]]
    covariation_outputs = [
        covariation_predictions[row["id"]]["prediction"] for row in annotation_block["covariation"]
    ]

    return accuracy(quartet_targets, quartet_outputs), macro_f1(covariation_targets, covariation_outputs)


def score_split(annotation, submission):
    fitness_spearman, fitness_r2 = evaluate_fitness(annotation["fitness"], submission["fitness"])
    structure_tm, structure_rmsd, structure_f1 = evaluate_structure(annotation["structure"], submission["structure"])
    evolution_accuracy, evolution_macro_f1 = evaluate_evolution(annotation["evolution"], submission["evolution"])
    evolution_score = (evolution_accuracy + evolution_macro_f1) / 2.0
    overall = mean([(1.0 + fitness_spearman) / 2.0, structure_tm, evolution_score])

    return {
        FITNESS_LABEL: round(fitness_spearman, 10),
        FITNESS_R2_LABEL: round(fitness_r2, 10),
        STRUCTURE_TM_LABEL: round(structure_tm, 10),
        STRUCTURE_RMSD_LABEL: round(structure_rmsd, 10),
        STRUCTURE_F1_LABEL: round(structure_f1, 10),
        EVOLUTION_ACC_LABEL: round(evolution_accuracy, 10),
        EVOLUTION_F1_LABEL: round(evolution_macro_f1, 10),
        OVERALL_LABEL: round(overall, 10),
    }


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    annotation = load_json(test_annotation_file)
    submission = load_json(user_submission_file)

    if phase_codename == "dev":
        split_name = "train_split"
    elif phase_codename == "test":
        split_name = "test_split"
    else:
        raise ValueError("Unknown phase codename: {}".format(phase_codename))

    metrics = score_split(annotation, submission)
    return {
        "result": [{split_name: metrics}],
        "submission_result": metrics,
    }
