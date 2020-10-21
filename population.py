import os

import numpy as np
import pandas as pd


def encode_categorical_column_data(df, continuous_columns, categorical_columns, class_column):
    categorical_infos = {c: pd.DataFrame({c: df[c].unique()}) for c in categorical_columns}
    encode_categorical_columns = []
    for k, v in categorical_infos.items():
        encode_categorical_columns += [f'{k}_{c}' for c in v.to_numpy().flatten().tolist()]
        df = pd.concat([df, pd.get_dummies(df[k], prefix=k)], axis=1)

    df = df.drop(columns=categorical_columns)
    df = cast_column_data(df, continuous_columns, encode_categorical_columns, class_column)
    return df, encode_categorical_columns


def cast_column_data(df, continuous_columns, encode_categorical_columns, class_column):
    df = df.astype({k: np.float32 for k in continuous_columns})
    df = df.astype({k: np.uint8 for k in encode_categorical_columns + class_column})
    return df


def filter_remaining_data(df, path, noise_ratio=0.0):
    if noise_ratio == 0.0:
        return df

    remaining_sample = pd.read_csv(os.path.join(path, f'noise_split-{noise_ratio}.csv'))
    remaining_sample = remaining_sample[remaining_sample['is_noise'] == False]
    df = df.iloc[remaining_sample.index, :]

    assert_value = remaining_sample['class'] - df['class']
    assert assert_value.abs().sum() == 0
    return df.reset_index(drop=True)


def resample(root, data, noise_ratio=0.0):
    if noise_ratio:
        file_path = 'noise_train_test_split.csv'
    else:
        file_path = 'data_split.csv'

    split_info = pd.read_csv(os.path.join(root, file_path))
    train_ind = split_info[split_info['is_train'] == 1].index
    test_ind = split_info[split_info['is_train'] == 0].index

    train_data = data.iloc[train_ind, :]
    test_data = data.iloc[test_ind, :]
    return train_data, test_data


def create_population_information(train_data, test_data, class_column, continuous_columns, encode_categorical_columns):
    # Reset index
    return {
        'train': train_data.reset_index(drop=True),
        'test': test_data.reset_index(drop=True),
        'feature_columns': continuous_columns + encode_categorical_columns,
        'class_column': class_column,
        'continuous_columns': continuous_columns,
        'categorical_columns': encode_categorical_columns,
    }


def abalone_population(is_resample=True, noise_ratio=0.0):
    root = './data/abalone'
    continuous_columns = [
        'length',
        'diameter',
        'height',
        'whole_weight',
        'shucked_weight',
        'viscera_weight',
        'shell_weight'
    ]
    categorical_columns = [
        'sex'
    ]

    # Bucket into 3 classes
    class_column = ['class']
    column_names = categorical_columns + continuous_columns + class_column

    df = pd.read_csv(os.path.join(root, 'abalone.data'),
                     names=column_names,
                     na_values='?',
                     sep=',')

    # Bucket into 3 classes
    mapping_class = {
        0: [i for i in range(1, 9)],
        1: [i for i in range(9, 11)],
        2: [i for i in range(11, 30)]
    }

    for i in range(0, len(mapping_class)):
        df.loc[df['class'].isin(mapping_class[i]), 'new_class'] = i
        df.loc[df['class'].isin(mapping_class[i]), 'new_class'] = i

    df['class'] = df['new_class']
    df = df.drop(columns=['new_class'])

    df, encode_categorical_columns = encode_categorical_column_data(df, continuous_columns, categorical_columns,
                                                                    class_column)
    df = filter_remaining_data(df, root, noise_ratio)

    if is_resample:
        train_data, test_data = resample(root, df, noise_ratio=noise_ratio)
    else:
        train_data = df.head(3133)
        test_data = df.tail(1044)

    return create_population_information(train_data, test_data, class_column, continuous_columns,
                                         encode_categorical_columns)


def cmc_population(is_resample=True, noise_ratio=0.0):
    root = './data/cmc'
    all_columns = [
        'wife_age',
        'wife_education',
        'number_of_children',
        'wife_religion',
        'wife_working',
        'husband_occupation',
        'living_index',
        'media_exposure'
    ]
    categorical_columns = [
        'husband_occupation'
    ]

    continuous_columns = list(set(all_columns) - set(categorical_columns))

    class_column = ['class']
    column_names = all_columns + class_column

    train_data = pd.read_csv(os.path.join(root, 'train.data'),
                             names=column_names,
                             na_values='?',
                             sep=',')
    test_data = pd.read_csv(os.path.join(root, 'test.data'),
                            names=column_names,
                            na_values='?',
                            sep=',')

    df = pd.concat([train_data, test_data], ignore_index=True)
    df, encode_categorical_columns = encode_categorical_column_data(df, continuous_columns, categorical_columns,
                                                                    class_column)
    df['class'] -= 1
    df = filter_remaining_data(df, root, noise_ratio)

    if is_resample:
        train_data, test_data = resample(root, df, noise_ratio=noise_ratio)
    else:
        train_data = df.head(len(train_data.index))
        test_data = df.tail(len(test_data.index))

    return create_population_information(train_data, test_data, class_column, continuous_columns,
                                         encode_categorical_columns)
