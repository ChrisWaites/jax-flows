from ax.service.managed_loop import optimize

from train import main as train_and_evaluate_model

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
