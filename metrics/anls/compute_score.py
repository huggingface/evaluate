from Levenshtein import ratio


def compute_score(predictions, ground_truths):
    theta = 0.5
    anls_score = 0
    for qid, prediction in predictions.items():
        max_value = 0
        if qid in ground_truths:
            for x in ground_truths[qid]:
                nl = ratio(prediction, x)
                if nl < theta:
                    score = 1 - nl
                    if score > max_value:
                        max_value = score
            anls_score += max_value

    return anls_score


if __name__ == "__main__":
    predictions = [{'question_id': '10285', 'prediction_text': 'Denver Broncos'},
                   {'question_id': '18601', 'prediction_text': '12/15/89'},
                   {'question_id': '16734', 'prediction_text': 'Dear dr. Lobo'}]

    references = [{"answers": ["Denver Broncos", "Denver R. Broncos"], 'question_id': '10285'},
               {'answers': ['12/15/88'], 'question_id': '18601'},
               {'answers': ['Dear Dr. Lobo', 'Dr. Lobo'], 'question_id': '16734'}]
    ground_truths = {x['question_id']: x['answers'] for x in references}
    predictions = {x['question_id']: x['prediction_text'] for x in predictions}
    anls_score = compute_score(predictions=predictions, ground_truths=ground_truths)
    print(anls_score)
