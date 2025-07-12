import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import os
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

class DatasetPreprocessor:
    def __init__(self, min_checkins=10, min_venue_visits=10, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Initialize the dataset preprocessor for FLLL³M
        
        Args:
            min_checkins: Minimum number of check-ins per user
            min_venue_visits: Minimum number of visits per venue
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        self.min_checkins = min_checkins
        self.min_venue_visits = min_venue_visits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def load_gowalla_data(self):
        """Load and preprocess Gowalla dataset"""
        print("Loading Gowalla dataset...")
        
        # Load check-ins data
        checkins_df = pd.read_csv('Gowalla/loc-gowalla_totalCheckins.txt/Gowalla_totalCheckins.txt', 
                                 sep='\t', header=None,
                                 names=['user_id', 'timestamp', 'lat', 'lon', 'location_id'])
        
        # Convert timestamp to datetime
        checkins_df['timestamp'] = pd.to_datetime(checkins_df['timestamp'])
        
        # Load social network edges
        edges_df = pd.read_csv('Gowalla/loc-gowalla_edges.txt/Gowalla_edges.txt', 
                              sep='\t', header=None,
                              names=['user1', 'user2'])
        
        return checkins_df, edges_df
    
    def load_brightkite_data(self):
        """Load and preprocess Brightkite dataset"""
        print("Loading Brightkite dataset...")
        
        # Load check-ins data
        checkins_df = pd.read_csv('Brightkite/loc-brightkite_totalCheckins.txt/Brightkite_totalCheckins.txt', 
                                 sep='\t', header=None,
                                 names=['user_id', 'timestamp', 'lat', 'lon', 'location_hash'])
        
        # Convert timestamp to datetime
        checkins_df['timestamp'] = pd.to_datetime(checkins_df['timestamp'])
        
        # Load social network edges
        edges_df = pd.read_csv('Brightkite/loc-brightkite_edges.txt/Brightkite_edges.txt', 
                              sep='\t', header=None,
                              names=['user1', 'user2'])
        
        return checkins_df, edges_df
    
    def load_foursquare_data(self):
        """Load and preprocess Foursquare dataset"""
        print("Loading Foursquare dataset...")
        
        # Load NYC data with encoding specification
        nyc_df = pd.read_csv('Foursquare/dataset_tsmc2014/dataset_TSMC2014_NYC.txt', 
                            sep='\t', header=None, encoding='latin-1',
                            names=['user_id', 'venue_id', 'venue_category_id', 'venue_category_name', 
                                   'lat', 'lon', 'timezone_offset', 'utc_time'])
        
        # Load Tokyo data with encoding specification
        tky_df = pd.read_csv('Foursquare/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', 
                            sep='\t', header=None, encoding='latin-1',
                            names=['user_id', 'venue_id', 'venue_category_id', 'venue_category_name', 
                                   'lat', 'lon', 'timezone_offset', 'utc_time'])
        
        # Convert timestamp to datetime
        nyc_df['timestamp'] = pd.to_datetime(nyc_df['utc_time'])
        tky_df['timestamp'] = pd.to_datetime(tky_df['utc_time'])
        
        # Add city information
        nyc_df['city'] = 'NYC'
        tky_df['city'] = 'TKY'
        
        # Combine datasets
        foursquare_df = pd.concat([nyc_df, tky_df], ignore_index=True)
        
        return foursquare_df, None  # No social network data for Foursquare
    
    def load_weeplace_data(self):
        """Load and preprocess Weeplace dataset"""
        print("Loading Weeplace dataset...")
        
        # Load check-ins data
        checkins_df = pd.read_csv('Weeplace/weeplace_checkins.csv')
        
        # Convert timestamp to datetime
        checkins_df['datetime'] = pd.to_datetime(checkins_df['datetime'])
        checkins_df = checkins_df.rename(columns={'datetime': 'timestamp', 'lat': 'lat', 'lon': 'lon', 'userid': 'user_id'})
        
        return checkins_df, None  # No social network data for Weeplace
    
    def filter_data(self, checkins_df, venue_col='location_id'):
        """
        Filter data according to FLLL³M paper specifications:
        - Filter users with <10 check-ins
        - Filter venues visited <10 times
        """
        print("Filtering data...")
        
        # Filter users with minimum check-ins
        user_counts = checkins_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_checkins].index
        checkins_df = checkins_df[checkins_df['user_id'].isin(valid_users)]
        
        # Filter venues with minimum visits
        venue_counts = checkins_df[venue_col].value_counts()
        valid_venues = venue_counts[venue_counts >= self.min_venue_visits].index
        checkins_df = checkins_df[checkins_df[venue_col].isin(valid_venues)]
        
        print(f"After filtering: {len(checkins_df)} check-ins, {len(valid_users)} users, {len(valid_venues)} venues")
        
        return checkins_df
    
    def apply_median_filtering(self, checkins_df, window_size=5):
        """
        Apply median filtering for GPS noise reduction
        """
        print("Applying median filtering for GPS noise reduction...")
        
        # Sort by user and timestamp
        checkins_df = checkins_df.sort_values(['user_id', 'timestamp'])
        
        # Apply median filtering to coordinates
        checkins_df['lat_filtered'] = checkins_df.groupby('user_id')['lat'].rolling(
            window=window_size, center=True).median().reset_index(0, drop=True)
        checkins_df['lon_filtered'] = checkins_df.groupby('user_id')['lon'].rolling(
            window=window_size, center=True).median().reset_index(0, drop=True)
        
        # Fill NaN values with original coordinates
        checkins_df['lat_filtered'] = checkins_df['lat_filtered'].fillna(checkins_df['lat'])
        checkins_df['lon_filtered'] = checkins_df['lon_filtered'].fillna(checkins_df['lon'])
        
        return checkins_df
    
    def normalize_timestamps(self, checkins_df):
        """
        Normalize timestamps to relative time from first check-in
        """
        print("Normalizing timestamps...")
        
        # Convert to Unix timestamp
        checkins_df['unix_timestamp'] = checkins_df['timestamp'].astype(np.int64) // 10**9
        
        # Normalize per user (relative to their first check-in)
        checkins_df['normalized_time'] = checkins_df.groupby('user_id')['unix_timestamp'].transform(
            lambda x: x - x.min())
        
        return checkins_df
    
    def split_data(self, checkins_df):
        """
        Split data into training/validation/test sets (6:2:2 ratio)
        """
        print("Splitting data into train/val/test sets...")
        
        # Sort by user and timestamp
        checkins_df = checkins_df.sort_values(['user_id', 'timestamp'])
        
        train_data, val_data, test_data = [], [], []
        
        for user_id in tqdm(checkins_df['user_id'].unique(), desc="Splitting data"):
            user_data = checkins_df[checkins_df['user_id'] == user_id].copy()
            
            # Calculate split indices
            n_checkins = len(user_data)
            train_end = int(n_checkins * self.train_ratio)
            val_end = int(n_checkins * (self.train_ratio + self.val_ratio))
            
            # Split data
            train_data.append(user_data.iloc[:train_end])
            val_data.append(user_data.iloc[train_end:val_end])
            test_data.append(user_data.iloc[val_end:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_location_mapping(self, checkins_df, venue_col='location_id'):
        """
        Create location ID mapping for consistent indexing
        """
        print("Creating location mapping...")
        
        unique_locations = checkins_df[venue_col].unique()
        # Convert to string keys to avoid JSON serialization issues
        location_to_idx = {str(loc): idx for idx, loc in enumerate(unique_locations)}
        idx_to_location = {idx: str(loc) for loc, idx in location_to_idx.items()}
        
        # Add location index to dataframe
        checkins_df['location_idx'] = checkins_df[venue_col].astype(str).map(location_to_idx)
        
        return checkins_df, location_to_idx, idx_to_location
    
    def preprocess_dataset(self, dataset_name):
        """
        Complete preprocessing pipeline for a dataset
        """
        print(f"\n=== Preprocessing {dataset_name} dataset ===")
        
        # Load data
        if dataset_name == 'gowalla':
            checkins_df, edges_df = self.load_gowalla_data()
            venue_col = 'location_id'
        elif dataset_name == 'brightkite':
            checkins_df, edges_df = self.load_brightkite_data()
            venue_col = 'location_hash'
        elif dataset_name == 'foursquare':
            checkins_df, edges_df = self.load_foursquare_data()
            venue_col = 'venue_id'
        elif dataset_name == 'weeplace':
            checkins_df, edges_df = self.load_weeplace_data()
            venue_col = 'placeid'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Apply preprocessing steps
        checkins_df = self.filter_data(checkins_df, venue_col)
        checkins_df = self.apply_median_filtering(checkins_df)
        checkins_df = self.normalize_timestamps(checkins_df)
        checkins_df, location_to_idx, idx_to_location = self.create_location_mapping(checkins_df, venue_col)
        
        # Split data
        train_df, val_df, test_df = self.split_data(checkins_df)
        
        # Save processed data
        output_dir = f'processed_data/{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)
        
        # Save mappings
        with open(f'{output_dir}/location_mapping.json', 'w') as f:
            json.dump({'location_to_idx': location_to_idx, 'idx_to_location': idx_to_location}, f)
        
        # Save social network data if available
        if edges_df is not None:
            edges_df.to_csv(f'{output_dir}/social_network.csv', index=False)
        
        print(f"Processed data saved to {output_dir}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'edges': edges_df,
            'location_mapping': {'location_to_idx': location_to_idx, 'idx_to_location': idx_to_location}
        }

def main():
    """Main preprocessing function"""
    preprocessor = DatasetPreprocessor()
    
    # Process all datasets
    datasets = ['gowalla', 'brightkite', 'foursquare', 'weeplace']
    
    for dataset in datasets:
        try:
            preprocessor.preprocess_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue

if __name__ == "__main__":
    main() 