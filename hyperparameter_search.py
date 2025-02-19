import itertools
import subprocess
import datetime

# Define hyperparameter values
lr_values = [0.00075, 0.0005, 0.00025]
batch_sizes = [50]
weight_decay_values = [0.00075, 0.0005, 0.00025]
scheduler_gamma_values = [0.2]

train_dataset_size = 80000
noise_level = 50000
noise_scale = 8
loss_function = 'mae'

# Generate all combinations
hyperparameter_combinations = list(itertools.product(lr_values, batch_sizes,
                                                     weight_decay_values, scheduler_gamma_values))

for lr, batch_size, weight_decay, scheduler_gamma in hyperparameter_combinations:
    command = (f"python main.py --train --test --lr {lr} --bs {batch_size} "
               f"--weight_decay {weight_decay} --scheduler_gamma {scheduler_gamma} "
               f"--train_dataset_size {train_dataset_size} "
               f"--noise_level {noise_level} --loss_fn {loss_function} "
               f"--noise_scale {noise_scale}")
    subprocess.run(command, shell=True)

print("Finished at time: ", datetime.datetime.now())