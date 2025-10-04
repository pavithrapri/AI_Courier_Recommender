# database.py - MISSING FILE THAT'S BREAKING YOUR SYSTEM
import json
import os
from datetime import datetime
import uuid

class Database:
    def __init__(self):
        self.orders_file = "data/orders.json"
        self.orders = self.load_orders()
        
    def load_orders(self):
        """Load orders from JSON file"""
        try:
            if os.path.exists(self.orders_file):
                with open(self.orders_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading orders: {e}")
            return []
    
    def save_orders(self):
        """Save orders to JSON file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.orders_file, 'w') as f:
                json.dump(self.orders, f, indent=2)
        except Exception as e:
            print(f"Error saving orders: {e}")
    
    def add_order(self, order_data):
        """Add a new order - FIXED TO HANDLE AI RECOMMENDATIONS"""
        order_id = len(self.orders) + 1
        tracking_number = f"CB{order_id:06d}"
        
        # CRITICAL FIX: Properly handle AI recommendation data
        order = {
            'id': order_id,
            'tracking_number': tracking_number,
            'customer_name': order_data.get('customer_name', ''),
            'customer_email': order_data.get('customer_email', ''),
            'customer_phone': order_data.get('customer_phone', ''),
            'delivery_address': order_data.get('delivery_address', {}),
            'sender_country': order_data.get('sender_country', 'Germany'),
            'product_name': order_data.get('product_name', ''),
            'product_quantity': order_data.get('product_quantity', 1),
            'product_weight': order_data.get('product_weight', 1.0),
            'delivery_urgency': order_data.get('delivery_urgency', 'Standard'),
            'package_value': order_data.get('package_value', 0),
            'is_pallet': order_data.get('is_pallet', False),
            'special_instructions': order_data.get('special_instructions', ''),
            
            # AI RECOMMENDATION DATA - THIS WAS MISSING
            'courier_recommendation': order_data.get('courier_recommendation', 'DPD DE'),
            'courier_confidence': order_data.get('courier_confidence', 0.0),
            'courier_options': order_data.get('courier_options', []),
            
            # STATUS TRACKING
            'status': 'pending',
            'courier': None,  # Selected courier (different from recommendation)
            'created_at': datetime.now().isoformat(),
            'dispatched_at': None,
            'delivered_at': None
        }
        
        self.orders.append(order)
        self.save_orders()
        print(f"✅ Order {order_id} created with AI recommendation: {order['courier_recommendation']} ({order['courier_confidence']:.2%})")
        return order
    
    def get_order(self, order_id):
        """Get order by ID"""
        for order in self.orders:
            if order['id'] == order_id:
                return order
        return None
    
    def get_order_by_tracking(self, tracking_number):
        """Get order by tracking number"""
        for order in self.orders:
            if order['tracking_number'] == tracking_number:
                return order
        return None
    
    def update_order(self, order_id, updates):
        """Update an existing order"""
        for i, order in enumerate(self.orders):
            if order['id'] == order_id:
                self.orders[i].update(updates)
                self.save_orders()
                return self.orders[i]
        return None
    
    def get_pending_orders(self):
        """Get all pending orders"""
        return [order for order in self.orders if order['status'] == 'pending']
    
    def get_dispatched_orders(self):
        """Get all dispatched orders"""
        return [order for order in self.orders if order['status'] == 'dispatched']
    
    def get_delivered_orders(self):
        """Get all delivered orders"""
        return [order for order in self.orders if order['status'] == 'delivered']

# Global database instance
db = Database()