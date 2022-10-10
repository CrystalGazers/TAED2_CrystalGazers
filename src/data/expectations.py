import great_expectations as ge
import numpy as np
import pandas as pd
import json

DATASET = "ca-2"

for subset in ["train", "test", "valid"]:
    dataset_npz = np.load(f"./preprocessed/{DATASET}/ca.wiki.{subset}.npz")
    context, target = dataset_npz['idata'], np.expand_dims(dataset_npz['target'], axis=1)
    df_pandas = pd.DataFrame(np.hstack((context, target)), columns=["L3", "L2", "L1", "R1", "R2", "R3", "T"])
    df = ge.dataset.PandasDataset(df_pandas)

    all_expectations = []
    for col in df.columns:
        expectation = df.expect_column_values_to_be_of_type(col, 'int32')
        if not expectation.success:
            print("Something isn't as expected:")
            print(expectation)
        all_expectations.append(expectation.to_json_dict())
    
    with open('expectations.json', 'w') as f:
        json.dump(all_expectations, f, indent=2)
