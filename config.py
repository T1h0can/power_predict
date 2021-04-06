class config:
    features_start = 1
    features_end = 9
    features_cols = list(range(features_start, features_end))
    num_inputs = len(features_cols)
    num_outputs = 1
    num_hidden = 32
    num_layers = 1
    is_batch_first = True
    num_dropout = 0.2

    labels_index = -1
    time_step = 5

    num_fold = 8
    num_epochs = 64
    batch_size = 8192
    learning_rate = 0.01
    weight_decay = 0.1

    random_seed = 42

    do_train_shuffle = True

    althletes_weight = {'ZhouZheng': 77, 'ShiYW': 65, 'ZhouZheng-2021': 72.5, 'ChengQiang': 72, 'ChenYuke': 78, 'LiangFan': 72}
