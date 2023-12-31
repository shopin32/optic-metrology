import json
import os
import sys
from typing import Optional

import cv2
from optic_metrology.feature_impact import compute_feature_impact
from optic_metrology.modeling import train, train_predefined

from optic_metrology.reader import DataSetReader, InmemoryDataSet


def run(train_dataset_path: str, target_name: str, json_path: Optional[str], random_state=1234):
    reader = DataSetReader()
    dataset = reader.read(train_dataset_path)
    if json_path:
        models = train_predefined(dataset, target_name, json.load(open(json_path)), random_state=random_state)
    else:
        models = train(dataset, target_name, random_state=random_state)
    
    for model, metric in models:
        print("----------------------Trained Model---------------------")
        print(model)
        print()
        print("Accuracy: ", metric)
        print()
        feature_importance = compute_feature_impact(dataset, model)
        print('Feature Importance:')
        for f, mean2std in feature_importance.items():
            print('{}: {} +/- {}'.format(f, mean2std[0], mean2std[1]))
        print("--------------------------------------------------------")

if __name__ == '__main__':
    # run(sys.argv[1], sys.argv[2], json_path=sys.argv[3] if len(sys.argv) > 3 else None)
    path = '/home/petro/Downloads/article/'
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        if '_gray' in img_name:
            continue
        output = os.path.join(path, img_name.replace('.png', '_gray') + '.png')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output, gray)




