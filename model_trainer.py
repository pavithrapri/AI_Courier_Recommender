import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor_module import DataProcessor


class ModelTrainer:
    def __init__(self):
        self.model_path = "models/courier_recommender.pkl"
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        
    def load_model(self):
        """Load a pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                model_data = joblib.load(self.model_path)
            
                self.model = model_data['model']
                self.label_encoders = model_data.get('label_encoders', {})
                self.scaler = model_data.get('scaler', StandardScaler())
                self.model_data = model_data  # Store the full model data for reference

            
                print("Model loaded successfully")
                return model_data
            else:
                print(f"Model file not found at {self.model_path}")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def convert_datetime_features(self, X):
        """Convert datetime columns to useful business features, not timestamps"""
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
                
                # Don't include timestamps, year, or day - these cause overfitting
                
                # Fill any remaining NaN with 0
                    for new_col in [f'{column}_month', f'{column}_dayofweek', f'{column}_hour']:
                        if new_col in X_processed.columns:
                            X_processed[new_col] = X_processed[new_col].fillna(0)
            
            # Drop the original datetime column
                X_processed = X_processed.drop(columns=[column])
    
        return X_processed
    


    def select_business_features(self, X_encoded):
        """Select ONLY pure business features - FIXED VERSION"""
    
    # REMOVED CourierServiceName - it creates circular dependency!
        core_business_features = [
            'DeliveryAddressCountry',  # WHERE it's going (most important)
            'WarehouseName',           # WHERE it's coming from
            'TotalWeight',             # Package weight affects courier choice
            'PackCount',               # Number of packages
            'IsPallet',                # Pallet vs regular package
            'DeliveryType',            # Express vs Standard
            'ItemType',                # Product type
        # REMOVED: 'CourierServiceName' - This was causing circular logic!
        
            'package_count',           # From XML parsing
            'has_dangerous_goods',     # From XML parsing  
            'declared_value'           # From XML parsing
        ]
    
    # Only include features that actually exist in your data
        business_features = [col for col in core_business_features if col in X_encoded.columns]
    
        print(f"Selected PURE business features (NO circular logic): {business_features}")
    
        if len(business_features) < 4:
            print("WARNING: Not enough business features found!")
            print("Available columns:", X_encoded.columns.tolist())
    
        return X_encoded[business_features]

    def encode_features(self, X):
        """Encode categorical features"""
        X_encoded = X.copy()
    
    # FIRST: Convert datetime columns to numerical features
        X_encoded = self.convert_datetime_features(X_encoded)
    
        print("All columns after datetime conversion:")
        print(X_encoded.dtypes)
    
    # Then encode categorical features
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                # Handle NaN values before encoding
                    X_encoded[column] = X_encoded[column].fillna('Unknown')
                    X_encoded[column] = self.label_encoders[column].fit_transform(X_encoded[column].astype(str))
                else:
                # Handle unseen labels by mapping to most frequent
                    valid_classes = set(self.label_encoders[column].classes_)
                    X_encoded[column] = X_encoded[column].fillna('Unknown')
                    X_encoded[column] = X_encoded[column].astype(str).apply(
                        lambda x: x if x in valid_classes else self.label_encoders[column].classes_[0]
                    )
                    X_encoded[column] = self.label_encoders[column].transform(X_encoded[column])
    
        return X_encoded
    
    def train_model(self, use_grid_search=False):
        """Train the Random Forest model with actual data"""
        # Load and process data
        processor = DataProcessor()
        processed_df = processor.process_and_save()
        
        if processed_df.empty:
            print("No data available for training. Please check your data file.")
            return None
        
        # Prepare training data
        X, y = processor.prepare_training_data(processed_df)
        
        # Remove rows with NaN values in target
        print(f"Before NaN removal: {len(y)} samples")
        print(f"NaN values in target: {y.isnull().sum()}")
        
        # Clean the data
        valid_mask = y.notna()  # Keep only non-NaN values
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"After NaN removal: {len(y)} samples")
        print(f"Unique couriers: {y.unique()}")
        
        if len(y) == 0:
            print("No valid data after cleaning!")
            return None
        
        if X.empty or y.empty:
            print("No valid data for training after preprocessing.")
            return None
        
        print(f"Training on {len(X)} samples with {len(y.unique())} unique couriers")
        print(f"Couriers: {y.unique()}")
        
        # Encode features (this now includes datetime conversion)
        X_encoded = self.encode_features(X)
        
        X_encoded = self.select_business_features(X_encoded)

        print(f"Feature columns after business selection: {list(X_encoded.columns)}")
        print(f"Data types: {X_encoded.dtypes.value_counts()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            print(f"Scaling numerical columns: {numerical_cols}")
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        if use_grid_search:
            # Hyperparameter tuning with GridSearch
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', None]
            }
            
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            print("Performing grid search for optimal parameters...")
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
        else:
            # Standard training
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            best_model.fit(X_train_scaled, y_train)
        
        # Store the trained model
        self.model = best_model
        
        # Evaluate model
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        
        # Plot confusion matrix and feature importance
        try:
            self.plot_confusion_matrix(y_test, y_pred)
            self.plot_feature_importance(feature_importance)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
        
        # Save model and encoders
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': best_model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': X_encoded.columns.tolist(),
            'feature_importance': feature_importance.to_dict()
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return model_data
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            model_data = self.load_model()
            if model_data is None:
                raise ValueError("No model available. Please train a model first.")
        
        # Encode features using the same process as training
        X_encoded = self.encode_features(X)
        # Ensure we have the same columns as training
        expected_columns = self.model_data.get('feature_columns', [])
        if expected_columns:
        # Add missing columns with default values
            for col in expected_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0  # Default value
        
        # Keep only expected columns in correct order
            X_encoded = X_encoded[expected_columns]
        
        # Scale numerical features
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X_scaled = X_encoded.copy()
            X_scaled[numerical_cols] = self.scaler.transform(X_encoded[numerical_cols])
        else:
            X_scaled = X_encoded
        
        return self.model.predict(X_scaled)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=np.unique(y_true), 
                       yticklabels=np.unique(y_true))
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Confusion matrix saved to models/confusion_matrix.png")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        try:
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)  # Show top 15 features
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 15 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Feature importance plot saved to models/feature_importance.png")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    def analyze_data_distribution(self):
        """Analyze the distribution of data for insights"""
        try:
            processor = DataProcessor()
            processed_df = processor.process_and_save()
            
            if processed_df.empty:
                print("No data available for analysis.")
                return
            
            print("Data Distribution Analysis:")
            print(f"Total records: {len(processed_df)}")
            print(f"Unique couriers: {processed_df['CourierName'].nunique()}")
            print("\nCourier distribution:")
            print(processed_df['CourierName'].value_counts())
            
            if 'DeliveryAddressCountry' in processed_df.columns:
                print("\nCountry distribution:")
                print(processed_df['DeliveryAddressCountry'].value_counts())
            
            if 'WarehouseName' in processed_df.columns:
                print("\nWarehouse distribution:")
                print(processed_df['WarehouseName'].value_counts())
            
            # Plot courier distribution
            plt.figure(figsize=(12, 6))
            courier_counts = processed_df['CourierName'].value_counts()
            courier_counts.plot(kind='bar')
            plt.title('Courier Distribution')
            plt.xlabel('Courier Name')
            plt.ylabel('Number of Orders')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('models/courier_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Courier distribution plot saved to models/courier_distribution.png")
            
        except Exception as e:
            print(f"Error in data analysis: {e}")

    def verify_data_mapping(self):
        """Verify that training data matches form expectations"""
        processor = DataProcessor()
        processed_df = processor.process_and_save()
    
        print("Available columns in training data:")
        print(processed_df.columns.tolist())
    
        print("\nSample values:")
        if 'DeliveryAddressCountry' in processed_df.columns:
            print(f"Countries: {processed_df['DeliveryAddressCountry'].unique()[:10]}")
        if 'WarehouseName' in processed_df.columns:
            print(f"Warehouses: {processed_df['WarehouseName'].unique()[:10]}")
        if 'CourierName' in processed_df.columns:
            print(f"Couriers: {processed_df['CourierName'].unique()}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Check data mapping first
    print("Verifying data mapping...")
    trainer.verify_data_mapping()
    
    # Analyze the data
    print("\nAnalyzing data distribution...")
    trainer.analyze_data_distribution()
    
    # Train with business features only
    print("\nStarting model training with business features...")
    model_data = trainer.train_model(use_grid_search=False)
    
    if model_data:
        print("Model training completed successfully!")
        print("Now the model should give better AI recommendations based on business logic!")
    else:
        print("Model training failed. Please check your data.")