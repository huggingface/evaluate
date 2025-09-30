import datasets
import simpleder

import evaluate


_CITATION = """\
@inproceedings{gallibert2013methodologies,
    author = {Olivier Galibert},
    title = {Methodologies for the evaluation of speaker diarization and automatic speech recognition in the presence of overlapping speech},
    booktitle = {Interspeech},
    year = {2013}}
"""

_DESCRIPTION = """
The primary metric utilized in speaker diarization experiments is the Diarization Error Rate (DER), as defined and employed by NIST in the RT evaluations (NIST Fall Rich Transcription on meetings 2006 Evaluation Plan, 2006). DER measures the fraction of time that is incorrectly attributed to a speaker or non-speech segments. To evaluate DER, the MD-eval-v12.pl script (NIST MD-eval-v21 DER evaluation script, 2006), developed by NIST, is utilized.

According to the task definition, the diarization output from the system hypothesis does not necessarily need to identify speakers by name or definitive ID. Thus, the ID tags assigned to speakers in both the hypothesis and reference segmentation do not need to match. This contrasts with non-speech tags, which are identified as unlabeled gaps between two speaker segments and therefore must be identified explicitly.

The evaluation script initially establishes an optimal one-to-one mapping of all speaker label IDs between hypothesis and reference files. This facilitates the scoring of different ID tags between the two files.
"""

_KWARGS_DESCRIPTION = """
Compute Diarization Error Rate (DER) between two lists of speaker segments.
Args:
    predictions: List of tuples (speaker_id, start_time, end_time) representing the diarization hypothesis.
    references: List of tuples (speaker_id, start_time, end_time) representing the ground truth diarization.
Returns:
    float: Diarization Error Rate (DER).

Examples:

    >>> predictions = [("1", 0.0, 0.8), ("2", 0.8, 1.4), ("3", 1.5, 1.8), ("1", 1.8, 2.0)]
    >>> references = [("A", 0.0, 1.0), ("B", 1.0, 1.5), ("A", 1.6, 2.1)]
    >>> der = evaluate.load("der")
    >>> der_score = der.compute(predictions=predictions, references=references)
    >>> print(der_score)
    0.350
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DER(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/wq2012/SimpleDER"],
            reference_urls=["https://xavieranguera.com/phdthesis/node108.html"],
        )

    def _compute(self, predictions, references):
        error = simpleder.DER(references, predictions)
        return error
