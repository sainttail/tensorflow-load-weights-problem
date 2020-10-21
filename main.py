import os

import tensorflow as tf

import constants
import population


def create_model(input_features, output_layer, hidden_layers):
    weight_decay = 1e-4
    initializer = tf.keras.initializers.VarianceScaling(2.0)

    inputs = tf.keras.Input(shape=(input_features,))
    x = inputs

    # 2 hidden layers
    for i, unit in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(unit,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  activation=tf.nn.relu,
                                  name=f'hidden_{i + 1}')(x)

    output = tf.keras.layers.Dense(output_layer,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name='output')(x)

    model = tf.keras.Model(inputs=inputs,
                           outputs=output,
                           name='boundarysampling')

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy()
                  ])
    model.summary()
    return model


def create_standardization(data, stats, apply_columns=None):
    if apply_columns is None:
        apply_columns = ['class']

    exclude_columns = data.columns.difference(apply_columns)
    class_value = data[exclude_columns]
    mean = stats['mean']
    std = stats['std']
    # Make sure the constant column is not `NaN`
    std[std == 0] = 1
    standardize_data = (data - mean) / std
    standardize_data[exclude_columns] = class_value

    return standardize_data


def extract_sample(df, feature_columns):
    features = df[feature_columns]
    labels = df[['class']]
    return features.values, labels.values


def get_data(population_type):
    if population_type == constants.POPULATION_ABALONE:
        pop = population.abalone_population()
    else:
        pop = population.cmc_population()

    pop_train = pop['train'].copy()
    pop_test = pop['test'].copy()

    columns = pop['continuous_columns']

    train_stats = pop_train.describe()
    train_stats = train_stats.transpose()

    # Always apply standardization
    pop_train = create_standardization(pop_train, train_stats, apply_columns=columns)
    pop_test = create_standardization(pop_test, train_stats, apply_columns=columns)

    number_of_classes = pop_train['class'].nunique()

    train_features, train_labels = extract_sample(pop_train, feature_columns=columns)
    test_features, test_labels = extract_sample(pop_test, feature_columns=columns)

    return train_features, train_labels, test_features, test_labels, number_of_classes


def main(population_type, is_train=False):
    train_features, train_labels, test_features, test_labels, number_of_classes = get_data(population_type)

    _, D = train_features.shape
    model = create_model(input_features=D, output_layer=number_of_classes, hidden_layers=[D])

    weights_folder = os.path.join(f'./weights/', population_type)
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    path = os.path.join(weights_folder, 'weights.h5')

    if is_train:
        hist = model.fit(train_features,
                         train_labels,
                         validation_data=(test_features, test_labels),
                         shuffle=True,
                         epochs=100,
                         batch_size=constants.BATCH_SIZE_CONFIGURATION[population_type],
                         verbose=2)
        model.save_weights(path)
        print(hist.history)
        return model, hist.history
    else:
        model.load_weights(path)
        loss, acc = model.evaluate(test_features, test_labels, verbose=1)
        print(f'Restored model, loss: {loss}, accuracy: {acc}')


if __name__ == '__main__':
    main(population_type=constants.POPULATION_CMC, is_train=True)
