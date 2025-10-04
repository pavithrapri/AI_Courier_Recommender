import pandas as pd
import numpy as np
import os
from datetime import datetime
import xml.etree.ElementTree as ET


class DataProcessor:
    def __init__(self):
        self.data_path = "data/DispatchesFR.xlsx"
        self.processed_data_path = "data/processed_data.csv"
        
    def load_data(self):
        """Load and merge Excel sheets with proper column mapping"""
        try:
            # Load all sheets from the Excel file
            df_rfr = pd.read_excel(self.data_path, sheet_name='RFR')
            df_res = pd.read_excel(self.data_path, sheet_name='RES')
            df_rde = pd.read_excel(self.data_path, sheet_name='RDE')
        
            # First, check what columns you actually have
            print("RFR columns:", df_rfr.columns.tolist())
            print("RES columns:", df_res.columns.tolist())
            print("RDE columns:", df_rde.columns.tolist())
        
            # Just add source identifier
            df_rfr['Source'] = 'RFR'
            df_res['Source'] = 'RES'
            df_rde['Source'] = 'RDE'
        
            # Merge all dataframes
            merged_df = pd.concat([df_rfr, df_res, df_rde], ignore_index=True)
        
            print(f"Merged dataframe columns: {merged_df.columns.tolist()}")
            print(f"Merged dataframe shape: {merged_df.shape}")
        
            # Check if CourierName column exists and has data
            if 'CourierName' in merged_df.columns:
                print(f"Courier distribution: {merged_df['CourierName'].value_counts()}")
            else:
                print("WARNING: CourierName column not found!")
        
            return merged_df
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def extract_xml_features(self, xml_string):
        """Extract features from ShippingConsignmentListXml"""
        try:
            if pd.isna(xml_string) or xml_string.strip() == '' or not isinstance(xml_string, str):
                return {
                    'package_count': 1,
                    'has_dangerous_goods': False,
                    'insurance_value': 0,
                    'declared_value': 0
                }
            
            # Clean the XML string if it contains malformed content
            xml_clean = xml_string.strip()
            if not xml_clean.startswith('<'):
                # Try to find the start of XML content
                start_idx = xml_clean.find('<')
                if start_idx >= 0:
                    xml_clean = xml_clean[start_idx:]
                else:
                    return {
                        'package_count': 1,
                        'has_dangerous_goods': False,
                        'insurance_value': 0,
                        'declared_value': 0
                    }
            
            root = ET.fromstring(xml_clean)
            package_count = len(root.findall('.//Package')) or 1
            
            # Check for dangerous goods
            has_dangerous_goods = False
            for elem in root.iter():
                if 'dangerous' in elem.tag.lower() or 'hazard' in elem.tag.lower():
                    has_dangerous_goods = True
                    break
            
            # Try to extract insurance and declared values
            insurance_value = 0
            declared_value = 0
            
            insurance_elem = root.find('.//InsuranceValue')
            if insurance_elem is not None and insurance_elem.text:
                try:
                    insurance_value = float(insurance_elem.text)
                except:
                    pass
            
            declared_elem = root.find('.//DeclaredValue')
            if declared_elem is not None and declared_elem.text:
                try:
                    declared_value = float(declared_elem.text)
                except:
                    pass
            
            return {
                'package_count': package_count,
                'has_dangerous_goods': has_dangerous_goods,
                'insurance_value': insurance_value,
                'declared_value': declared_value
            }
        except ET.ParseError:
            # Return defaults if XML parsing fails
            return {
                'package_count': 1,
                'has_dangerous_goods': False,
                'insurance_value': 0,
                'declared_value': 0
            }
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return {
                'package_count': 1,
                'has_dangerous_goods': False,
                'insurance_value': 0,
                'declared_value': 0
            }
    
    def preprocess_data(self, df):
        """Clean and preprocess the data - FIXED PANDAS WARNINGS"""
        if df.empty:
            print("No data to process")
            return df
            
        # Make a proper copy to avoid modifying the original
        processed_df = df.copy()
        
        # CRITICAL FIX: Handle missing values properly without inplace warnings
        if 'CourierName' in processed_df.columns:
            processed_df.loc[:, 'CourierName'] = processed_df['CourierName'].fillna('Unknown')
        else:
            processed_df.loc[:, 'CourierName'] = 'Unknown'
        
        if 'DeliveryAddressCountry' in processed_df.columns:
            processed_df.loc[:, 'DeliveryAddressCountry'] = processed_df['DeliveryAddressCountry'].fillna('DE')
        else:
            processed_df.loc[:, 'DeliveryAddressCountry'] = 'DE'
        
        if 'WarehouseName' in processed_df.columns:
            processed_df.loc[:, 'WarehouseName'] = processed_df['WarehouseName'].fillna('Germany')
        else:
            processed_df.loc[:, 'WarehouseName'] = 'Germany'
        
        if 'PackCount' in processed_df.columns:
            processed_df.loc[:, 'PackCount'] = processed_df['PackCount'].fillna(1)
        else:
            processed_df.loc[:, 'PackCount'] = 1
        
        if 'IsPallet' in processed_df.columns:
            processed_df.loc[:, 'IsPallet'] = processed_df['IsPallet'].fillna(False)
            # Convert to boolean if it's not already
            if processed_df['IsPallet'].dtype == 'object':
                processed_df.loc[:, 'IsPallet'] = processed_df['IsPallet'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
        else:
            processed_df.loc[:, 'IsPallet'] = False
        
        if 'TotalWeight' in processed_df.columns:
            # Convert to numeric, coercing errors to NaN
            processed_df.loc[:, 'TotalWeight'] = pd.to_numeric(processed_df['TotalWeight'], errors='coerce')
            # Fill NaN with median
            median_weight = processed_df['TotalWeight'].median()
            if pd.isna(median_weight):
                median_weight = 10.0  # Default median if all are NaN
            processed_df.loc[:, 'TotalWeight'] = processed_df['TotalWeight'].fillna(median_weight)
        else:
            processed_df.loc[:, 'TotalWeight'] = 10.0  # Default weight
            
        # Add missing business columns if they don't exist
        if 'DeliveryType' not in processed_df.columns:
            processed_df.loc[:, 'DeliveryType'] = 'Standard'
        else:
            processed_df.loc[:, 'DeliveryType'] = processed_df['DeliveryType'].fillna('Standard')
            
        if 'ItemType' not in processed_df.columns:
            processed_df.loc[:, 'ItemType'] = 'General'
        else:
            processed_df.loc[:, 'ItemType'] = processed_df['ItemType'].fillna('General')
            
        if 'CourierServiceName' not in processed_df.columns:
            processed_df.loc[:, 'CourierServiceName'] = 'Standard'
        else:
            processed_df.loc[:, 'CourierServiceName'] = processed_df['CourierServiceName'].fillna('Standard')
        
        # Convert date columns
        date_columns = ['Date', 'DispatchDate']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df.loc[:, col] = pd.to_datetime(processed_df[col], errors='coerce')
                processed_df.loc[:, col] = processed_df[col].fillna(pd.Timestamp.now())
                
        # Use Date as primary date if available, otherwise use DispatchDate
        if 'Date' in processed_df.columns:
            processed_df.loc[:, 'OrderDate'] = processed_df['Date']
        elif 'DispatchDate' in processed_df.columns:
            processed_df.loc[:, 'OrderDate'] = processed_df['DispatchDate']
        else:
            processed_df.loc[:, 'OrderDate'] = pd.Timestamp.now()
        
        # Extract features from XML
        if 'ShippingConsignmentListXml' in processed_df.columns:
            xml_features = processed_df['ShippingConsignmentListXml'].apply(self.extract_xml_features)
            xml_df = pd.json_normalize(xml_features)
            processed_df = pd.concat([processed_df, xml_df], axis=1)
            
            # Drop the original XML column
            processed_df = processed_df.drop(columns=['ShippingConsignmentListXml'])
        else:
            # Add default XML features if column doesn't exist
            processed_df.loc[:, 'package_count'] = 1
            processed_df.loc[:, 'has_dangerous_goods'] = False
            processed_df.loc[:, 'insurance_value'] = 0
            processed_df.loc[:, 'declared_value'] = 0
        
        processed_df = self.normalize_countries_in_training_data(processed_df)
        return processed_df
    
    def normalize_countries_in_training_data(self, processed_df):
        """Normalize country names in training data to match form data"""
        country_mapping = {
            'Germany': 'DE',
            'France': 'FR', 
            'United Kingdom': 'GB',
            'Switzerland': 'CH',
            'Austria': 'AT',
            'Netherlands': 'NL',
            'Belgium': 'BE',
            'Spain': 'ES',
            'Denmark': 'DK',
            'Luxembourg': 'LU',
        }

        if 'DeliveryAddressCountry' in processed_df.columns:
            processed_df['DeliveryAddressCountry'] = processed_df['DeliveryAddressCountry'].map(
                lambda x: country_mapping.get(x, x)
            )

        if 'WarehouseName' in processed_df.columns:
            processed_df['WarehouseName'] = processed_df['WarehouseName'].map(
                lambda x: country_mapping.get(x, x) if isinstance(x, str) else x
            )

        return processed_df


    
    def prepare_training_data(self, processed_df):
        """Prepare data for accurate country-based courier training"""
        processed_df = processed_df.copy()
    
        # Clean courier data
        processed_df = processed_df[processed_df['CourierName'].notna()]
        processed_df = processed_df[processed_df['CourierName'] != 'Unknown']
    
        # Filter out couriers with too few samples for reliable training
        courier_counts = processed_df['CourierName'].value_counts()
        print("Original courier distribution:", courier_counts)
    
        min_samples = 5  # Keep couriers with at least 5 samples
        valid_couriers = courier_counts[courier_counts >= min_samples].index
        processed_df = processed_df[processed_df['CourierName'].isin(valid_couriers)]
    
        print("Filtered courier distribution:", processed_df['CourierName'].value_counts())
    
        # CRITICAL: Only use business-relevant features that affect courier choice
        business_features = [
            'DeliveryAddressCountry',  # Most important - where it's going
            'WarehouseName',           # Where it's coming from
            'TotalWeight',             # Package weight affects courier choice
            'PackCount',               # Number of packages
            'IsPallet',                # Pallet vs regular shipment
            'DeliveryType',            # Express vs Standard service
            'CourierServiceName',      # Service level
            'ItemType' ,              # Product type
            'package_count',           # From XML parsing
            'has_dangerous_goods',     # From XML parsing  
            'declared_value'
        ]
    
        # Keep only features that exist in your data
        available_features = [col for col in business_features if col in processed_df.columns]
        print(f"Using business features: {available_features}")
    
        # Select ONLY business features (no time-based features)
        X = processed_df[available_features].copy()
        y = processed_df['CourierName'].copy()
    
        # Handle NaN values in selected features - FIXED VERSION
        for col in available_features:
            if col in X.columns:
                if X[col].dtype in ['object']:
                    X.loc[:, col] = X[col].fillna('Unknown')
                else:
                    X.loc[:, col] = X[col].fillna(0)
    
        # Remove any rows with remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
    
        print(f"Final training features: {X.columns.tolist()}")
        print(f"Final training data shape: {X.shape}")
        print(f"Final courier distribution: {y.value_counts()}")
    
        return X, y
    
    def process_and_save(self):
        """Main method to process data and save it"""
        df = self.load_data()
        
        if df.empty:
            print("No data found. Please ensure your Excel file exists at data/DispatchesFR.xlsx")
            return pd.DataFrame()
            
        processed_df = self.preprocess_data(df)
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        processed_df.to_csv(self.processed_data_path, index=False)
        
        print(f"Processed {len(processed_df)} records")
        return processed_df

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Test if we can load the data
    df = processor.load_data()
    
    if not df.empty:
        print("SUCCESS: Data loaded")
        print("Available columns:", df.columns.tolist())
        
        # Check for the most important column
        if 'CourierName' in df.columns:
            print("CourierName found:", df['CourierName'].value_counts())
        else:
            print("CRITICAL ERROR: No CourierName column - AI cannot learn!")
            
        # Check other key columns
        key_columns = ['DeliveryAddressCountry', 'TotalWeight', 'PackCount']
        for col in key_columns:
            if col in df.columns:
                print(f"{col}: ✅")
            else:
                print(f"{col}: ❌ MISSING")
    else:
        print("FAILED: Could not load data")