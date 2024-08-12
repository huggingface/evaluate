import datasets
import evaluate
import re

_DESCRIPTION = """
VQA accuracy is a evaluation metric which is robust to inter-human variability in phrasing the answers:
$\\text{Acc}(ans) = \\min \\left( \\frac{\\text{# humans that said }ans}{3}, 1 \\right)$
Where `ans` is answered by machine. In order to be consistent with 'human accuracies', machine accuracies are averaged over all 10 choose 9 sets of human annotators.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `str`): Predicted answers.
    references (`list` of `str` lists): Ground truth answers. 
    answer_types (`list` of `str`, *optional*): Answer types corresponding to each questions.
    questions_type (`list` of `str`, *optional*): Question types corresponding to each questions.

Returns:
    visual question answering accuracy (`float`): Accuracy accuracy. Minimum possible value is 0. Maximum possible value is 100.

"""


_CITATION = """
@InProceedings{{VQA},
author      = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title       = {{VQA}: {V}isual {Q}uestion {A}nswering},
booktitle   = {International Conference on Computer Vision (ICCV)},
year        = {2015},
}
"""

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]

periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class VQAAccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="sequence"), id="references"
                    ),
                    "answer_types": datasets.Value("string", id="sequence"),
                    "question_types": datasets.Value("string", id="sequence"),
                }
            ),
            reference_urls=[
                "https://visualqa.org/evaluation.html",
                "https://github.com/GT-Vision-Lab/VQA/blob/master",
            ],
        )

    def _compute(self, predictions, references, answer_types=None, question_types=None):
        if answer_types is None:
            answer_types = [None] * len(predictions)

        if question_types is None:
            question_types = [None] * len(predictions)

        if not len(predictions) == len(answer_types) == len(question_types):
            raise ValueError(
                "The length of predictions, answer_types and question_types doesn't match."
            )

        total, ans_type_dict, ques_type_dict = [], {}, {}

        for pred, gts, ans_type, ques_type in zip(
            predictions, references, answer_types, question_types
        ):
            # to align with offical data postprocess
            pred = pred.replace("\n", " ").replace("\t", " ").strip()
            pred = processDigitArticle(processPunctuation(pred))
            gts = [processDigitArticle(processPunctuation(gt_ans)) for gt_ans in gts]

            # calculate vqa accuracy
            accuracy = []
            for i in range(len(gts)):
                other_gt = gts[:i] + gts[i + 1 :]
                matching_ans = [item for item in other_gt if item == pred]
                accuracy.append(min(1, len(matching_ans) / 3))

            vqa_acc = sum(accuracy) / len(accuracy)
            total.append(vqa_acc)

            if ans_type is not None:
                if ans_type not in ans_type_dict:
                    ans_type_dict[ans_type] = []
                ans_type_dict[ans_type].append(vqa_acc)

            if ques_type is not None:
                if ques_type not in ques_type_dict:
                    ques_type_dict[ques_type] = []
                ques_type_dict[ques_type].append(vqa_acc)

        # the following key names follow the naming of the official evaluation results
        result = {"overall": 100 * sum(total) / len(total)}

        if len(ans_type_dict) > 0:
            result["perAnswerType"] = {
                ans_type: 100 * sum(accuracy_list) / len(accuracy_list)
                for ans_type, accuracy_list in ans_type_dict.items()
            }

        if len(ques_type_dict) > 0:
            result["perQuestionType"] = {
                ques_type: 100 * sum(accuracy_list) / len(accuracy_list)
                for ques_type, accuracy_list in ques_type_dict.items()
            }

        return result
