from tensorboard import notebook

log_dir = "results"
notebook.start("--logdir {}".format(log_dir))
