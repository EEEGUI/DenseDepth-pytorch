import numpy as np


class DepthEstimateScore(object):
    def __init__(self):
        self.score_dict = {'a1': [],
                           'a2': [],
                           'a3': [],
                           'abs_rel': [],
                           'rmse': [],
                           'log_10': []}

    def reset(self):
        self.score_dict = {'a1': [],
                           'a2': [],
                           'a3': [],
                           'abs_rel': [],
                           'rmse': [],
                           'log_10': []}

    def update(self, label_true, label_pred):
        errors = self._compute_errors(label_true, label_pred)
        for i, key in enumerate(self.score_dict.keys()):
            self.score_dict[key].append(errors[i])

    def get_scores(self):
        scores = {}
        for key in self.score_dict.keys():
            scores[key] = np.mean(self.score_dict[key])
        return scores

    def _compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        return [a1, a2, a3, abs_rel, rmse, log_10]


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
