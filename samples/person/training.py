from samples.person.person_dataset import PersonDataset


def train(model, dataset_dir, config):
    dataset_train = PersonDataset()
    dataset_train.load_person(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = PersonDataset()
    dataset_val.load_person(dataset_dir, "val")
    dataset_val.prepare()

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=60, layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=120, layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=160, layers='all')
