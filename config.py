class config:
    features_start = 1
    features_end = 9
    features_cols = list(range(features_start, features_end))
    num_inputs = len(features_cols)
    num_outputs = 1
    num_hidden = 32
    num_layers = 2
    num_dropout = 0.2

    labels_index = -1
    time_step = 5

    num_fold = 40
    num_epochs = 10000
    batch_size = 3000
    learning_rate = 1e-3
    weight_decay = 0.1

    random_seed = 42

    do_train = True
