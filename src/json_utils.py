import json
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler


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

    print('hshshs')

if __name__ == '__main__':
    convert_to_csv()