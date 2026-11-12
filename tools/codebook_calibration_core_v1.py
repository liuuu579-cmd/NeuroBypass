"""Training-only mean calibration and selection-only conservative mask decisions."""
import numpy as np


ALPHAS = (0., .25, .5, .75, 1.)
MARGINS = (0., .05, .1, .2, .4, .8)


def choose(scores, baseline, margin=0.):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 2 and np.isfinite(scores).all() and margin >= 0
    candidate = scores.argmin(1)
    improvement = scores[:, baseline] - scores[np.arange(len(scores)), candidate]
    # Equality always keeps the training-selected static mask.
    return np.where(improvement > margin, candidate, baseline)


def calibrated(scores, train_prediction_mean, train_cost_mean, alpha):
    return np.asarray(train_cost_mean) + alpha * (np.asarray(scores) - train_prediction_mean)


def fit_policy(scores, costs, train_rows, selection_rows, baseline, scale):
    """No non-selection validation costs are read during policy fitting."""
    scores, costs = np.asarray(scores), np.asarray(costs)
    prediction_mean = scores[train_rows].mean(0)
    cost_mean = costs[train_rows].mean(0)
    candidates = [{'family': 'static', 'alpha': 0., 'margin_units': 0.}]
    candidates += [{'family': family, 'alpha': alpha, 'margin_units': margin}
        for family in ('raw_gate', 'mean_calibrated')
        for alpha in ((1.,) if family == 'raw_gate' else ALPHAS)
        for margin in MARGINS]
    records = []
    for serial, policy in enumerate(candidates):
        if policy['family'] == 'static':
            routes = np.full(len(selection_rows), baseline)
        else:
            values = scores[selection_rows] if policy['family'] == 'raw_gate' else calibrated(
                scores[selection_rows], prediction_mean, cost_mean, policy['alpha'])
            routes = choose(values, baseline, policy['margin_units'] * scale)
        value = float(costs[selection_rows, routes].mean())
        switches = int((routes != baseline).sum())
        records.append({**policy, 'selection_kl': value, 'selection_switches': switches, 'serial': serial})
    # At identical selection quality prefer fewer switches, then fixed grid order.
    best = min(records, key=lambda r: (r['selection_kl'], r['selection_switches'], r['serial']))
    return {'chosen': best, 'grid': records, 'train_prediction_mean': prediction_mean.tolist(),
        'train_cost_mean': cost_mean.tolist(), 'scale': float(scale), 'baseline': int(baseline)}


def apply_policy(scores, fitted):
    policy, baseline = fitted['chosen'], fitted['baseline']
    if policy['family'] == 'static':
        return np.full(len(scores), baseline, dtype=np.int64)
    values = scores if policy['family'] == 'raw_gate' else calibrated(scores,
        np.asarray(fitted['train_prediction_mean']), np.asarray(fitted['train_cost_mean']), policy['alpha'])
    return choose(values, baseline, policy['margin_units'] * fitted['scale'])


def cpu_checks():
    scores = np.array([[0., 0., 1.], [-.1, 0., 1.], [-.3, 0., 1.]])
    assert choose(scores, 1, .1).tolist() == [1, 1, 0]
    train = np.array([0, 1]); selection = np.array([2]); costs = np.array([[.4, .3, .8], [.5, .4, .8], [.2, .3, .8], [99., -99., 0.]])
    predictions = np.vstack([scores, [1., 0., 2.]])
    fitted = fit_policy(predictions, costs, train, selection, 1, .1)
    altered = costs.copy(); altered[3] *= -100
    assert fitted == fit_policy(predictions, altered, train, selection, 1, .1)
    adjusted = calibrated(predictions, predictions[train].mean(0), costs[train].mean(0), 1.)
    assert np.allclose(adjusted[train].mean(0), costs[train].mean(0))
    assert apply_policy(predictions, fitted).shape == (4,)
    return {'passed': True, 'ties_and_exact_margin_keep_static': True,
        'training_mean_correction_exact': True, 'nonselection_validation_costs_cannot_change_fit': True}
