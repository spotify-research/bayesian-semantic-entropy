"""Compute overall performance metrics from predicted uncertainties."""
import argparse
import functools
import logging
import os
import pickle

import numpy as np
#import wandb
from aim import Run
from pathlib import Path

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


utils.setup_logger()

result_dict = {}

UNC_MEAS = 'uncertainty_measures.pkl'


# def init_wandb(wandb_runid, assign_new_wandb_id, experiment_lot, entity):
#     """Initialize wandb session."""
#     user = os.environ['USER']
#     slurm_jobid = os.getenv('SLURM_JOB_ID')
#     scratch_dir = os.getenv('SCRATCH_DIR', '.')
#     kwargs = dict(
#         entity=entity,
#         project='semantic_uncertainty',
#         dir=f'{scratch_dir}/{user}/uncertainty',
#         notes=f'slurm_id: {slurm_jobid}, experiment_lot: {experiment_lot}',
#     )
#     if not assign_new_wandb_id:
#         # Restore wandb session.
#         wandb.init(
#             id=wandb_runid,
#             resume=True,
#             **kwargs)
#         wandb.restore(UNC_MEAS)
#     else:
#         api = wandb.Api()
#         wandb.init(**kwargs)

#         old_run = api.run(f'{entity}/semantic_uncertainty/{wandb_runid}')
#         old_run.file(UNC_MEAS).download(
#             replace=True, exist_ok=False, root=wandb.run.dir)


def analyze_run(
        aim_run, answer_fractions_mode='default'):
    """Analyze the uncertainty measures for a given wandb run id."""
    logging.info('Analyzing aim run hash `%s`.', aim_run.hash)
    
    # Set up evaluation metrics.
    if answer_fractions_mode == 'default':
        answer_fractions = [0.8, 0.9, 0.95, 1.0]
    elif answer_fractions_mode == 'finegrained':
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20+1)]
    else:
        raise ValueError

    rng = np.random.default_rng(41)
    eval_metrics = dict(zip(
        ['AUROC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, area_under_thresholded_accuracy, np.mean],
            [compatible_bootstrap, compatible_bootstrap, bootstrap]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    # if wandb.run is None:
    #     init_wandb(
    #         wandb_runid, assign_new_wandb_id=assign_new_wandb_id,
    #         experiment_lot=experiment_lot, entity=entity)

    # elif wandb.run.id != wandb_runid:
    #     raise ValueError

    # Load the results dictionary from a pickle file.
    directory = os.path.join(aim_run.repo.path, aim_run.hash)
    Path(directory).mkdir(parents=True, exist_ok=True)

    with open(f'{directory}/{UNC_MEAS}', 'rb') as file:
        results_old = pickle.load(file)

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute simple accuracy metrics for model predictions.
    all_accuracies = dict()
    all_accuracies['accuracy'] = 1 - np.array(results_old['validation_is_false'])

    for name, target in all_accuracies.items():
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = np.mean(target)
        result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)

    rum = results_old['uncertainty_measures']
    if 'p_false' in rum and 'p_false_fixed' not in rum:
        # Restore log probs true: y = 1 - x --> x = 1 - y.
        # Convert to probs --> np.exp(1 - y).
        # Convert to p_false --> 1 - np.exp(1 - y).
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    # Next: Uncertainty Measures.
    # Iterate through the dictionary and compute additional metrics for each measure.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)

        # Validation accuracy.
        validation_is_falses = [
            results_old['validation_is_false'],
            results_old['validation_unanswerable']
        ]

        logging_names = ['', '_UNANSWERABLE']

        # Iterate over predictions of 'falseness' or 'answerability'.
        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):
            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_is_false = np.array(validation_is_false)
            validation_accuracy = 1 - validation_is_false
            if len(measure_values) > len(validation_is_false):
                # This can happen, but only for p_false.
                if 'p_false' not in measure_name:
                    raise ValueError
                logging.warning(
                    'More measure values for %s than in validation_is_false. Len(measure values): %d, Len(validation_is_false): %d',
                    measure_name, len(measure_values), len(validation_is_false))
                measure_values = measure_values[:len(validation_is_false)]

            fargs = {
                'AUROC': [validation_is_false, measure_values],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values],
                'mean_uncertainty': [measure_values]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [validation_accuracy, measure_values]

            for fname, (function, bs_function) in eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = metric_i
                logging.info("%s for measure name `%s`: %f", fname, name, metric_i)
                result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                    function, rng)(*fargs[fname])

    aim_run['results_dict']  = result_dict
    logging.info(
        'Analysis for wandb_runid `%s` finished. Full results dict: %s',
        aim_run.hash, result_dict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim_run_hashes', nargs='+', type=str,
                        help='Aim run hashes of the datasets to evaluate on.')
    parser.add_argument('--answer_fractions_mode', type=str, default='default')
    
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    aim_run_hashes = args.aim_run_hashes
    for wid in aim_run_hashes:
        logging.info('Evaluating hash `%s`.', wid)
        analyze_run(
            Run(run_hash=wid), answer_fractions_mode=args.answer_fractions_mode)
