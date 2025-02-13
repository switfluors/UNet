import itertools
import subprocess

# Define hyperparameter values
lr_values = [0.001, 0.0005, 0.0001]
batch_sizes = [50]
weight_decay_values = [0.001, 0.0005, 0.0001]
scheduler_gamma_values = [0.2]

train_dataset_size = 160000
noise_level = 100000
loss_function = 'mae'

# Generate all combinations
hyperparameter_combinations = list(itertools.product(lr_values, batch_sizes,
                                                     weight_decay_values, scheduler_gamma_values))

for lr, batch_size, weight_decay, scheduler_gamma in hyperparameter_combinations:
    command = (f"python main.py --train --test --lr {lr} --bs {batch_size} "
               f"--weight_decay {weight_decay} --scheduler_gamma {scheduler_gamma} "
               f"--train_dataset_size {train_dataset_size} "
               f"--noise_level {noise_level} --loss_fn {loss_function}")
    subprocess.run(command, shell=True)