import configparser
import sys

config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
config = config['DEFAULT']

print(dict(config))

b1 = float(config['b1'])
b2 = float(config['b2'])
composition = config['composition'].lower()
dataset = config['dataset']
experiment = config['experiment']
flow = config['flow']
iterations = int(config['iterations'])
l2_norm_clip = float(config['l2_norm_clip'])
log_params = None if config['log_params'].lower() == 'false' else int(config['log_params'])
log_performance = int(config['log_performance'])
lr = float(config['lr'])
lr_schedule = config['lr_schedule'].lower()
pieces = int(config['pieces'])
minibatch_size = int(config['minibatch_size'])
noise_multiplier = float(config['noise_multiplier'])
normalization = str(config['normalization']).lower() == 'true'
num_blocks = int(config['num_blocks'])
num_hidden = int(config['num_hidden'])
optimizer = config['optimizer'].lower()
pieces_to_run = int(config['pieces_to_run'])
private = str(config['private']).lower() == 'true'
prior_type = str(config['prior_type']).lower()
sampling = config['sampling'].lower()
save_dataset = config['save_dataset'].lower() == 'true'
weight_decay = float(config['weight_decay'])
overwrite = config['overwrite'].lower() == 'true'
