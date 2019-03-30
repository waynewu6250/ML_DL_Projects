import torch as t
import numpy as np
from eval_model.main import evaluate

class CLRScorer:
    def __init__(self, label):
        """
        Candidate input: ["Sample Sentence"]
        refs input: ["Ref1", "Ref2", ...]
        """
        self.label = label
    
    def calc_score(self, candidate, refs):
        scores = []
        pred_labels = []
        for ref in refs:
            score, pred_label, _ = evaluate(candidate[0], ref, self.label)
            scores.append(score)
            pred_labels.append(pred_label)
        return max(scores), pred_labels

    
    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        scores = []
        pred_labels = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score, pred_labs = self.calc_score(hypo, ref)
            scores.append(score)
            pred_labels.append(max(pred_labs.count(x) for x in set(pred_labs)))
        
        average_score = np.mean(np.array(score))
        return average_score, scores, pred_labels
    
    def method(self):
        return "CLRScorer"



            

        


