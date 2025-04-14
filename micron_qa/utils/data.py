import pickle as pkl
import numpy as np
import pandas as pd
import os
import torch

from sqlalchemy import create_engine
import mysql.connector
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', image_size=224):
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.image_size = image_size
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([

            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        self.load_images()

    def load_images(self):
        class_names = sorted(os.listdir(self.root_dir))
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        for cls in tqdm(class_names, desc = "Loading_Dataset", unit='class'):
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def get_database(params):

    root_path = params['paths']['data']['source']\

    # some code line to access to sql and convert it to csv
    USER = "root"
    PASSWORD = "junyoung4u"
    HOST = "localhost"
    PORT = "3306"
    DATABASE = "secom"
    TABLE_NAME = "secom_data"
    
    conn = mysql.connector.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            database=DATABASE
            )
    query = f"SELECT * FROM {TABLE_NAME};"
    df = pd.read_sql(query, con=conn)
    
    path_save = os.path.join(root_path, "secom.csv")
    df.to_csv(path_save, index=False)
    conn.close()
    return df

def preprocess_tab_data(df, params):
    print("[Preprocessing Tabular Data]")

    # Drop the time column
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    # Separate features and labels
    if 'Pass/Fail' not in df.columns:
        raise ValueError("Expected label column 'Pass/Fail' not found.")
    
    labels = df['Pass/Fail'].replace(-1, 0).values
    features = df.drop(columns=['Pass/Fail'])

    # EDA
    print(f"Data shape: {df.shape}")
    print("Label Distribution (0 = Fail, 1 = Pass):")
    print(pd.Series(labels).value_counts())
    print("Percentage of Missing Values Per Column:")
    nan_ratios = features.isnull().mean() * 100
    print(nan_ratios[nan_ratios > 0].sort_values(ascending=False))

    # Compare mean vs median imputation
    nan_cols = features.columns[features.isnull().any()]
    if len(nan_cols) > 0:
        sample_col = nan_cols[0]
        mean_imputed = features[sample_col].fillna(features[sample_col].mean())
        median_imputed = features[sample_col].fillna(features[sample_col].median())
        print(f"\nComparison of Mean vs Median Imputation for '{sample_col}':")
        print("Mean Imputed:")
        print(mean_imputed.describe())
        print("Median Imputed:")
        print(median_imputed.describe())

    # Apply default imputation method (mean)
    features = features.fillna(features.mean())

    return {
        'features': features.to_numpy(),
        'labels': labels
    }

def preprocess_img_data(data, params):
    pass


def get_img_data(root_dir, batch_size, image_size):
    
    train_dataset = ImageDataset(root_dir, 'train', image_size)
    valid_dataset = ImageDataset(root_dir, 'valid', image_size)
    test_dataset = ImageDataset(root_dir, 'test', image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_data(params):

    img_dir= params['paths']['data']['image']
    batch_size = params['task']['image']['batch_size']
    image_size = params['task']['image']['image_size']

    if params['system']['datatype']==0:
        data_path = params['paths']['data']['tabular']
        #data = get_database(params)
        data = pd.read_csv(data_path)
        if params['preprocess']:
            data_path = params['paths']['data']['image']
            data = preprocess_tab_data(data, params)
            #if params['plots']['data']:
                # connect to tableau
        return data

    else:
        data_path = params['paths']['data']['image']
        train_loader, valid_loader, test_loader = get_img_data(img_dir, batch_size, image_size)
        if params['preprocess']:
            data = preprocess_img_data(data, params)
            

        return train_loader, valid_loader, test_loader
