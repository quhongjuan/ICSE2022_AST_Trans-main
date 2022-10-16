from collections import Counter

from valid_metrices.google_bleu import corpus_bleu
from valid_metrices.meteor import Meteor, nltk
from valid_metrices.rouge import Rouge


# def compute_score2(references, hypotheses):
#     score = 0;
#     list_size = len(hypotheses)
#     if (list_size != len(references)):
#         return 0
#     #
#     for i in range(0,list_size):
#         #print("references[1]:"+references[i])
#         score += nltk.translate.meteor_score.single_meteor_score(reference=[references[i].split()],
#                                                                  hypothesis=[hypotheses[i].split()])
#
#     return score * 1.0 / list_size


def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    #meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    meteor= meteor_calculator.compute_score2(references, hypotheses)

    return bleu * 100, rouge_l * 100, meteor * 100, ind_bleu, ind_rouge
