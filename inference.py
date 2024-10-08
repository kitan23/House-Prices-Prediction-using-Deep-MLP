"""
@ Description: This file is used to make inference on the model
@ Author: Kien Tran
@ Create Time:
"""


# import your pre-trained model here and conduct inference
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from model_src import MLP
import torch 
import argparse
from data_src import load_and_preprocess_data

# Load the test data
parser = argparse.ArgumentParser(description = "Load the CSV file for testing the model")
parser.add_argument('--test_path', type=str, required=True, help="Path to the CSV file")
args = parser.parse_args()

test_data = pd.read_csv(args.test_path)
_, test_features, _ = load_and_preprocess_data('data_src/train.csv', 'data_src/test.csv')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP()
model_path = "saved_models/best_mlp_house_price_model.pth"
model.load_state_dict(torch.load(model_path))

model.eval()
with torch.no_grad():
    test_predictions = model(test_features.to(device)).cpu().numpy()

# Save the predictions to a CSV file
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions.flatten()
})
submission.to_csv('submission.csv', index=False)

print("Predictions saved to 'submission.csv'.")
