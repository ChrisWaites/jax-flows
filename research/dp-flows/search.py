from hyperopt import hp, fmin, tpe, space_eval
from ax.service.managed_loop import optimize
from train import main as train_and_evaluate_model
from hyperopt.pyll import stochastic

"""
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.normal('y', 0, 1),
    'name': hp.choice('name', ['alice', 'bob']),

    'b1':               hp.choice('b1', [0.9]),
    'b2':               hp.choice('b2', [0.999]),
    'dataset':          hp.choice('dataset', [moons]),
    'delta':            hp.choice('delta', [1e-5]),
    'target_epsilon':   hp.choice('target_epsilon', [5.]),
    'flow':             hp.choice('flow', ['realnvp', 'glow', 'maf', 'maf-glow']),
    'iterations':       hp.choice('iterations', [50000]),
    'l2_norm_clip':     hp.choice('l2_norm_clip', [7.]),
    'log':              hp.choice('log', [False]),
    'lr':               hp.choice('lr', [1e-4, 1e-1]),
    'microbatch_size':  hp.choice('microbatch_size', [1]),
    'minibatch_size':   hp.choice('minibatch_size', [128]), # 512
    'noise_multiplier': hp.choice('noise_multiplier', [1.9]),
    'num_blocks':       hp.choice('num_blocks', [1, 5]),
    'num_hidden':       hp.choice('num_hidden', [8, 128]),
    'private':          hp.choice('private' [True]),
    'weight_decay':     hp.choice('weight_decay', [0]),
}
print stochastic.sample(space)
"""

parameters = [
    {'name': 'b1', 'type': 'fixed', 'value': 0.9},
    {'name': 'b2', 'type': 'fixed', 'value': 0.999},
    {'name': 'dataset', 'type': 'fixed', 'value': 'moons'},
    {'name': 'delta', 'type': 'fixed', 'value': 1e-5},
    {'name': 'target_epsilon', 'type': 'fixed', 'value': 5.},
    {'name': 'flow', 'type': 'choice', 'values': ['realnvp', 'glow', 'maf', 'maf-glow']},
    {'name': 'iterations', 'type': 'fixed', 'value': 50000},
    {'name': 'l2_norm_clip', 'type': 'fixed', 'value': 7.},
    {'name': 'log', 'type': 'fixed', 'value': False},
    {'name': 'lr', 'type': 'range', 'bounds': [1e-4, 1e-1], 'log_scale': True},
    {'name': 'microbatch_size', 'type': 'fixed', 'value': 1},
    {'name': 'minibatch_size', 'type': 'fixed', 'value': 128}, # 512
    {'name': 'noise_multiplier', 'type': 'fixed', 'value': 1.9},
    {'name': 'num_blocks', 'type': 'range', 'value_type': 'int', 'bounds': [1, 5]},
    {'name': 'num_hidden', 'type': 'range', 'value_type': 'int', 'bounds': [8, 128]},
    {'name': 'private', 'type': 'fixed', 'value': True},
    {'name': 'weight_decay', 'type': 'fixed', 'value': 0},
]

best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    evaluation_function=train_and_evaluate_model,
    minimize=True,
    objective_name='nll',
    total_trials=50,
)

print(best_parameters)
print(values)
