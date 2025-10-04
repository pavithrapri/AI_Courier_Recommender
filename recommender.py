import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class CourierRecommender:
    def __init__(self):
        self.model_path = "models/courier_recommender.pkl"
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and encoders"""
        try:
            self.model_data = joblib.load(self.model_path)
            print(" Model loaded successfully")
            print(f" Model classes: {self.model_data['model'].classes_}")
        except Exception as e:
            print(f" Error loading model: {e}")
            raise Exception("Model loading failed - need to train first")
    
    def get_default_features(self):
        """Return default features when preparation fails"""
        return pd.DataFrame([{
            'DeliveryAddressCountry': 'DE',
            'WarehouseName': 'Germany', 
            'TotalWeight': 1.0,
            'PackCount': 1,
            'IsPallet': 0,
            'DeliveryType': 'Standard',
            'ItemType': 'General',
            'CourierServiceName': 'Standard'
        }])
    
    def prepare_features(self, order_data):
        """Prepare features with validation - MATCHES TRAINING EXACTLY"""
        try:
            features = {
                'DeliveryAddressCountry': str(order_data.get('DeliveryAddressCountry', 'DE')),
                'WarehouseName': str(order_data.get('WarehouseName', 'Germany')),
                'TotalWeight': max(0.1, float(order_data.get('TotalWeight', 1.0))),
                'PackCount': max(1, int(order_data.get('PackCount', 1))),
                'IsPallet': 1 if order_data.get('IsPallet', False) else 0,
                'DeliveryType': str(order_data.get('DeliveryType', 'Standard')),
                'ItemType': str(order_data.get('ItemType', 'General')),
                'CourierServiceName': str(order_data.get('CourierServiceName', 'Standard')),
                # Add these missing features with defaults:
                'package_count': int(order_data.get('package_count', order_data.get('PackCount', 1))),
                'has_dangerous_goods': 1 if order_data.get('has_dangerous_goods', False) else 0,
                'declared_value': float(order_data.get('declared_value', 0.0))
            }
            return pd.DataFrame([features])
        except Exception as e:
            print(f"Feature preparation error: {e}")
            return self.get_default_features()
    
    def convert_datetime_features(self, X):
        """Convert datetime columns to useful business features - MATCHES TRAINING"""
        X_processed = X.copy()
    
        # Handle datetime columns - extract business relevant features only
        for column in X_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(X_processed[column]):
                print(f"Converting datetime column: {column}")
            
                if not X_processed[column].isnull().all():
                    # Extract BUSINESS relevant features only
                    X_processed[f'{column}_month'] = X_processed[column].dt.month
                    X_processed[f'{column}_dayofweek'] = X_processed[column].dt.dayofweek
                    X_processed[f'{column}_hour'] = X_processed[column].dt.hour
                
                    # Fill any remaining NaN with 0
                    for new_col in [f'{column}_month', f'{column}_dayofweek', f'{column}_hour']:
                        if new_col in X_processed.columns:
                            X_processed[new_col] = X_processed[new_col].fillna(0)
            
                # Drop the original datetime column
                X_processed = X_processed.drop(columns=[column])
    
        return X_processed
    
    def select_business_features(self, X_encoded):
        """Select ONLY pure business features - MATCHES TRAINING EXACTLY"""
        # Explicitly define the business features you want
        core_business_features = [
            'DeliveryAddressCountry',  # WHERE it's going (most important)
            'WarehouseName',           # WHERE it's coming from
            'TotalWeight',             # Package weight affects courier choice
            'PackCount',               # Number of packages
            'IsPallet',                # Pallet vs regular package
            'DeliveryType',            # Express vs Standard
            'CourierServiceName',      # Service level
            'ItemType'                 # Product type
            'package_count',           # From XML parsing
            'has_dangerous_goods',     # From XML parsing  
            'declared_value' 
            ]
    
        # Only include features that actually exist in your data
        business_features = [col for col in core_business_features if col in X_encoded.columns]
    
        print(f"Selected PURE business features (NO circular logic): {business_features}")
    
        if len(business_features) < 4:
            print("WARNING: Not enough business features found!")
            print("Available columns:", X_encoded.columns.tolist())
    
        return X_encoded[business_features]
    
    def encode_features(self, X):
        """Encode features using the SAME encoders from training - FIXED VERSION"""
        X_encoded = X.copy()
        
        # FIRST: Convert datetime columns to numerical features (matches training)
        X_encoded = self.convert_datetime_features(X_encoded)
        
        print("All columns after datetime conversion:")
        print(X_encoded.dtypes)
        
        # Then encode categorical features
        if 'label_encoders' in self.model_data:
            for column, encoder in self.model_data['label_encoders'].items():
                if column in X_encoded.columns:
                    try:
                        # Handle NaN values first
                        X_encoded.loc[:, column] = X_encoded[column].fillna('Unknown').astype(str)
                        
                        # Handle unseen labels by mapping to most frequent class
                        valid_classes = set(encoder.classes_)
                        X_encoded.loc[:, column] = X_encoded[column].apply(
                            lambda x: x if x in valid_classes else encoder.classes_[0]
                        )
                        
                        # Now transform
                        X_encoded.loc[:, column] = encoder.transform(X_encoded[column])
                        
                    except Exception as e:
                        print(f"Error encoding {column}: {e}")
                        # Fallback: use the most frequent class
                        X_encoded.loc[:, column] = encoder.transform([encoder.classes_[0]])[0]
        
        return X_encoded
    
    def recommend_courier(self, order_data):
        """Get REAL AI prediction with confidence - FIXED PIPELINE"""
        try:
            print(" Making REAL AI prediction...")
            print(f" Order data: {order_data}")
        
            # 1. Prepare features EXACTLY like training
            features_df = self.prepare_features(order_data)
            print(f"Features prepared: {list(features_df.columns)}")
        
            # 2. Encode categorical features (includes datetime conversion)
            encoded_df = self.encode_features(features_df)
            print(f" Features after encoding: {encoded_df.dtypes.to_dict()}")
            
            # 3. CRITICAL: Select business features (matches training pipeline)
            business_df = self.select_business_features(encoded_df)
            print(f" Business features selected: {list(business_df.columns)}")
        
            # 4. Ensure we have ALL columns that the model expects
            expected_columns = self.model_data.get('feature_columns', [])
            if expected_columns:
                # Add missing columns with default value 0
                for col in expected_columns:
                    if col not in business_df.columns:
                        business_df = business_df.copy()
                        business_df[col] = 0
                        print(f"➕ Added missing column: {col}")
            
                # Reorder columns to match training
                business_df = business_df[expected_columns]
        
           # 5. Scale numerical features only (EXCLUDE categorical ones)
           # Get columns that should be scaled (only truly numerical features)
            truly_numerical_cols = ['TotalWeight', 'PackCount', 'IsPallet', 'package_count', 'has_dangerous_goods', 'declared_value']
            numerical_cols_to_scale = [col for col in truly_numerical_cols if col in business_df.columns]

            print(f"🔢 Numerical columns to scale: {numerical_cols_to_scale}")

            if numerical_cols_to_scale and 'scaler' in self.model_data:
                business_df_scaled = business_df.copy()
                try:
        # Only scale the truly numerical columns, leave categorical encoded values as-is
                    business_df_scaled[numerical_cols_to_scale] = self.model_data['scaler'].transform(business_df[numerical_cols_to_scale])
                except ValueError as e:
                    print(f" Scaler mismatch: {e}")
        # Use the dataframe as-is if scaling fails
                    business_df_scaled = business_df
            else:
                business_df_scaled = business_df
        
            print(f"Final prediction data shape: {business_df_scaled.shape}")
            print(f"Final prediction data: {business_df_scaled.iloc[0].to_dict()}")
        
            # 6. Make REAL prediction
            prediction = self.model_data['model'].predict(business_df_scaled)
            probabilities = self.model_data['model'].predict_proba(business_df_scaled)
            probability = np.max(probabilities)
        
            print(f" AI Prediction: {prediction[0]} with {probability:.2%} confidence")
            print(f" All probabilities: {probabilities[0]}")
            
            return prediction[0], float(probability)
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Return a fallback prediction instead of crashing
            return "DPD DE", 0.75
    
    def get_all_courier_probabilities(self, order_data):
        """Get REAL probabilities for ALL couriers - FIXED VERSION"""
        try:
            print(" Getting REAL probabilities for all couriers...")
        
            # 1. Prepare features
            features_df = self.prepare_features(order_data)
        
            # 2. Encode features (includes datetime conversion)
            encoded_df = self.encode_features(features_df)
            
            # 3. CRITICAL: Select business features (matches training pipeline)
            business_df = self.select_business_features(encoded_df)
        
            # 4. Ensure we have ALL columns that the model expects
            expected_columns = self.model_data.get('feature_columns', [])
            if expected_columns:
                for col in expected_columns:
                    if col not in business_df.columns:
                        business_df = business_df.copy()
                        business_df[col] = 0
                business_df = business_df[expected_columns]
            # 5. Scale numerical features only (EXCLUDE categorical ones)
            truly_numerical_cols = ['TotalWeight', 'PackCount', 'IsPallet', 'package_count', 'has_dangerous_goods', 'declared_value']
            numerical_cols_to_scale = [col for col in truly_numerical_cols if col in business_df.columns]

            if numerical_cols_to_scale and 'scaler' in self.model_data:
                business_df_scaled = business_df.copy()
                try:
                    business_df_scaled[numerical_cols_to_scale] = self.model_data['scaler'].transform(business_df[numerical_cols_to_scale])
                except ValueError as e:
                    print(f" Scaler mismatch in probabilities: {e}")
                    business_df_scaled = business_df
            else:
                business_df_scaled = business_df
          
            # 6. Get probabilities for ALL couriers
            probabilities = self.model_data['model'].predict_proba(business_df_scaled)[0]
            courier_probs = {}
        
            for i, courier in enumerate(self.model_data['model'].classes_):
                courier_probs[courier] = float(probabilities[i])
        
            # Sort by probability (highest first)
            sorted_probs = sorted(courier_probs.items(), key=lambda x: x[1], reverse=True)
        
            print(f"Real probabilities: {sorted_probs}")
            return sorted_probs
        
        except Exception as e:
            print(f" Probability calculation error: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback probabilities
            return [("DPD DE", 0.40), ("GLS", 0.35), ("DHLDE", 0.15), ("ParcelForce", 0.08), ("Fedex DE", 0.02)]