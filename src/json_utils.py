import json
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted, order_by_index

def convert_to_csv():
    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]
    columns = ['ImageName', 'FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']
    data = []
    scaler = MinMaxScaler()

    for key in measurements.keys():
        cur_image_features = measurements[key]
        data.append([key, cur_image_features['FreshWeightShoot'], cur_image_features['DryWeightShoot'], cur_image_features['Height'], cur_image_features['Diameter'], cur_image_features['LeafArea']])

    measurements_df = pd.DataFrame(data, columns=columns)
    measurements_df[config.FEATURES] = scaler.fit_transform(measurements_df[config.FEATURES])
    # scaled_measurements_df = scaler.fit_transform(measurements_df[config.FEATURES])


def sort_json2csv():
    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]
    columns = ['ImageName', 'Variety', 'RGBImage', 'DebthInformation', 'FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']
    data = []

    for key in measurements.keys():
        cur_image_features = measurements[key]
        # Lazy coding
        data.append([key, cur_image_features['Variety'],cur_image_features['RGBImage'],
                     cur_image_features['DebthInformation'], cur_image_features['FreshWeightShoot'], cur_image_features['DryWeightShoot'],
                     cur_image_features['Height'], cur_image_features['Diameter'], cur_image_features['LeafArea']])

    measurements_df = pd.DataFrame(data, columns=columns)
    # measurements_df.sort_values(by=['ImageName'])
    measurements_df = measurements_df.reindex(index=order_by_index(measurements_df.index, index_natsorted(measurements_df['ImageName'], reverse=False)))
    measurements_df.reset_index(drop=True, inplace=True)
    # measurements_df.drop(['index'], axis=1)
    measurements_df.to_csv("./sorted_GroundTruth.csv", index=False)
    # measurements_df[config.FEATURES] = scaler.fit_transform(measurements_df[config.FEATURES])
    # scaled_measurements_df = scaler.fit_transform(measurements_df[config.FEATURES])

    print('hshshs')

if __name__ == '__main__':
    sort_json2csv()
    # convert_to_csv()