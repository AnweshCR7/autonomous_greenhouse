import json
import pandas as pd
import config
import pickle
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted, order_by_index
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Defunct
def convert_to_csv():
    # f = open(config.JSON_FILE)
    f = open("./GroundTrhyth_June8.json")
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]
    columns = ['ImageName', 'Variety', 'FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']
    data = []
    scaler = MinMaxScaler()

    for key in measurements.keys():
        cur_image_features = measurements[key]
        data.append([key, cur_image_features['Variety'], cur_image_features['FreshWeightShoot'], cur_image_features['DryWeightShoot'], cur_image_features['Height'], cur_image_features['Diameter'], cur_image_features['LeafArea']])

    measurements_df = pd.DataFrame(data, columns=columns)
    # measurements_df[config.FEATURES] = scaler.fit_transform(measurements_df[config.FEATURES])

    # scaled_measurements_df = scaler.fit_transform(measurements_df[config.FEATURES])
    measurements_df.to_csv("./GroundTruth_June8.csv", index=False)

# Use this
def sort_json2csv(save_scaling=False):
    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]
    columns = ['ImageName', 'Variety', 'RGBImage', 'DebthInformation', 'FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']
    # columns = ['Unnamed: 0', 'LeafArea', 'FreshWeightShoot', 'DryWeightShoot']
    data = []
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    for key in measurements.keys():
        cur_image_features = measurements[key]
        # Lazy coding
        data.append([key, cur_image_features['Variety'],cur_image_features['RGBImage'],
                     cur_image_features['DebthInformation'], cur_image_features['FreshWeightShoot'], cur_image_features['DryWeightShoot'],
                     cur_image_features['Height'], cur_image_features['Diameter'], cur_image_features['LeafArea']])

        # data.append([key, cur_image_features['LeafArea'], cur_image_features['FreshWeightShoot'], cur_image_features['DryWeightShoot']])

    measurements_df = pd.DataFrame(data, columns=columns)
    # measurements_df.sort_values(by=['ImageName'])
    # measurements_df = measurements_df.reindex(index=order_by_index(measurements_df.index, index_natsorted(measurements_df['ImageName'], reverse=False)))
    # measurements_df = measurements_df.reindex(index=order_by_index(measurements_df.index, index_natsorted(measurements_df['Unnamed: 0'], reverse=False)))
    measurements_df.reset_index(drop=True, inplace=True)
    # measurements_df.to_csv("../data/master_data.csv", index=False)
    # measurements_df.drop(['index'], axis=1)

    # train_idx = pd.read_csv("../data/features/X_train.csv")["Unnamed: 0"].values
    # test_idx = pd.read_csv("../data/features/X_eval.csv")["Unnamed: 0"].values
    # # Lets try splitting like this for now
    # # train_df, test_df = train_test_split(measurements_df, test_size=0.2)
    # train_df = measurements_df[measurements_df['ImageName'].isin(train_idx)]
    # test_df = measurements_df[measurements_df['ImageName'].isin(test_idx)]
    # train_df.to_csv("./TrainGroundTruth.csv", index=False)
    # test_df.to_csv("./TestGroundTruth.csv", index=False)

    measurements_df = pd.read_csv("../data/final_data/Feature_all.csv")
    train_x = measurements_df[config.ADD_FEATURES]
    measurements_df = pd.read_csv("../data/master_data.csv")
    train_y = measurements_df[config.FEATURES]

    # Fit the scaler on training data and pickle it
    if save_scaling:
        scalerfile = "ft_scaler_x.sav"
        # Only use the numerical values for scaling
        x_scaler.fit(train_x[config.ADD_FEATURES])
        y_scaler.fit(train_y[config.FEATURES])
        pickle.dump(x_scaler, open(scalerfile, 'wb'))
        scalerfile = "ft_scaler_y.sav"
        pickle.dump(y_scaler, open(scalerfile, 'wb'))


    # measurements_df.to_csv("./GroundTruth.csv", index=False)
    # measurements_df[config.FEATURES] = scaler.fit_transform(measurements_df[config.FEATURES])
    # scaled_measurements_df = scaler.fit_transform(measurements_df[config.FEATURES])

    print('hshshs')


def generate_additional_features(save_scaling=True):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    train_x = pd.read_csv(config.TRAIN_ADD_FEATURES)
    eval_x = pd.read_csv(config.TEST_ADD_FEATURES)
    train_y = pd.read_csv(config.TRAIN_ADD_FEATURES_Y)


    # Engineer Features here:
    train_x['Shuffle'] = train_x['Area'] * train_x['Height'] * train_x['Volume'] * train_x['Diameter']
    eval_x['Shuffle'] = eval_x['Area'] * eval_x['Height'] * eval_x['Volume'] * eval_x['Diameter']
    train_x['Shuffle'] = np.log(train_x['Shuffle'])
    eval_x['Shuffle'] = np.log(eval_x['Shuffle'])

    # save the df with new features
    train_x.to_csv(config.TRAIN_ADD_FEATURES, index=False)
    eval_x.to_csv(config.TEST_ADD_FEATURES, index=False)


    # Fit the scaler on training data and pickle it
    if save_scaling:
        scalerfile = f"{config.SCALER_PATH}/{config.SCALERFILE_X}"
        # Only use the numerical values for scaling
        x_scaler.fit(train_x[config.ADD_FEATURES])
        y_scaler.fit(train_y[config.FEATURES])
        pickle.dump(x_scaler, open(scalerfile, 'wb'))
        scalerfile = f"{config.SCALER_PATH}/{config.SCALERFILE_Y}"
        pickle.dump(y_scaler, open(scalerfile, 'wb'))


if __name__ == '__main__':
    sort_json2csv(save_scaling=True)
    # generate_additional_features(save_scaling=True)
    # sort_json2csv(save_scaling=True)