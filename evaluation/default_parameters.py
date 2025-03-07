def set_seq_len(dataset):
    dataset = dataset.lower()
    if "weather" in dataset:
        return 11
    elif "air_quality" in dataset:
        return 10
    elif "sine" in dataset:
        return 10
    elif "pen_digit" in dataset and "nospaces" in dataset:
        return 8
    elif "pen_digit" in dataset:
        return 9
    if "characters" in dataset:
        return 64
    if "dog" in dataset:
        return 10
    if "activity" in dataset:
        return 15
    if "energy" in dataset:
        return 48
    else:
        return 10


def set_hidden_size(dataset):
    dataset = dataset.lower()
    return 50


def set_batch_size(dataset):
    dataset = dataset.lower()
    if "pen_digit" in dataset and "nospaces" in dataset:
        return 128
    if "pen_digit" in dataset:
        return 126
    if "geolife" in dataset:
        return 512
    if "characters" in dataset:
        return 256
    if "dog" in dataset:
        return 128
    if "activity" in dataset:
        return 128
    if "energy" in dataset:
        return 128
    else:
        return 128


def set_output_size(dataset):
    dataset = dataset.lower()
    if "pen_digit" in dataset:
        return 10
    if "geolife" in dataset:
        return 8
    if "characters" in dataset:
        return 2
    if "dog" in dataset:
        return 3
    if "energy" in dataset:
        return 2
    if "activity" in dataset:
        if "equal" in dataset:
            return 2
        return 11
    else:
        return 2


def set_delay(dataset):
    dataset = dataset.lower()
    if "future" in dataset:
        if "weather" in dataset:
            return 11
        elif "air_quality" in dataset:
            return 6
        elif "energy" in dataset:
            return 21
        return 0
    return 0


def set_adwin_delta(dataset):
    dataset = dataset.lower()
    return 0.002


def set_deltas_test_detector(dataset):
    dataset = dataset.lower()
    return [0.002]
