from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from data_processor_module import DataProcessor
from recommender import CourierRecommender
from database import db
import os
from datetime import datetime


def normalize_country_name(country_name):
    """Normalize country names to match training data format"""
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
    return country_mapping.get(country_name, country_name)

def setup_system():
    """Setup the system automatically"""
    try:
        print("Auto-setting up system...")
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)

        processor = DataProcessor()
        processed_data = processor.process_and_save()

        if not processed_data.empty:
            from model_trainer import ModelTrainer
            trainer = ModelTrainer()
            trainer.train_model(use_grid_search=False)  # Changed to False for faster training
            print("Auto-setup completed successfully!")
        else:
            print("Auto-setup: No data found, using demo mode")

    except Exception as e:
        print(f"Auto-setup error: {e}")

# Run setup when app starts
setup_system()

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if value is None:
        return ""
    # Convert string to datetime if needed
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

app = Flask(__name__)
app.secret_key = '4f0c2c8dc9646041387e0d36bab5e0fbe7a957f7dda2e00eda23b492550ad395'

try:
    recommender = CourierRecommender()
    print("Recommender loaded successfully")
except Exception as e:
    print(f"Recommender loading failed: {e}")
    recommender = None

app.jinja_env.filters['datetimeformat'] = datetimeformat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/order', methods=['GET', 'POST'])
def order():
    if request.method == 'POST':
        print("=== FORM SUBMISSION DEBUG ===")
        print("Form data received:", dict(request.form))
        
        try:
            # Extract form data with proper validation
            order_data = {
                'customer_name': request.form.get('customer_name', '').strip(),
                'customer_email': request.form.get('customer_email', '').strip(),
                'customer_phone': request.form.get('customer_phone', '').strip(),
                'delivery_address': {
                    'street': request.form.get('delivery_street', '').strip(),
                    'city': request.form.get('delivery_city', '').strip(),
                    'postcode': request.form.get('delivery_postcode', '').strip(),
                    'country': request.form.get('delivery_country', 'DE').strip()
                },
                'sender_country': request.form.get('sender_country', 'Germany').strip(),
                'product_name': request.form.get('product_name', 'General').strip(),
                'product_quantity': max(1, int(request.form.get('product_quantity', 1))),
                'product_weight': max(0.1, float(request.form.get('product_weight', 1.0))),
                'delivery_urgency': request.form.get('delivery_urgency', 'Standard').strip(),
                'package_value': max(0, float(request.form.get('package_value', 0))),
                'is_pallet': 'is_pallet' in request.form,
                'special_instructions': request.form.get('special_instructions', '').strip(),
            }
            print("Order data extracted successfully:", order_data)

            # CRITICAL FIX: Prepare prediction data with EXACT feature names that match training
            prediction_data = {
                'DeliveryAddressCountry': normalize_country_name(order_data['delivery_address']['country']),
                'WarehouseName': normalize_country_name(order_data['sender_country']), 
                'TotalWeight': float(order_data['product_weight']),
                'PackCount': int(order_data['product_quantity']),
                'IsPallet': bool(order_data['is_pallet']),
                'DeliveryType': order_data['delivery_urgency'],
                'ItemType': order_data['product_name'],
                
                # ADD THESE MISSING FEATURES:
                'package_count': int(order_data['product_quantity']),  # Same as PackCount
                'has_dangerous_goods': False,  # Default to False for now
                'declared_value': float(order_data['package_value'])   # Use package value        
                
            }
            print("AI prediction data:", prediction_data)

            # Get AI recommendation with proper error handling
            try:
                if recommender:
                    print("Getting AI recommendation...")
                    recommended_courier, confidence = recommender.recommend_courier(prediction_data)
                    all_probabilities = recommender.get_all_courier_probabilities(prediction_data)
                    print(f"AI Recommendation: {recommended_courier} (confidence: {confidence:.2%})")
                    print(f"All probabilities: {all_probabilities}")
                    
                    # Save order with AI data
                    order = db.add_order({
                        **order_data,
                        'courier_recommendation': recommended_courier,
                        'courier_confidence': confidence,
                        'courier_options': all_probabilities
                    })
                else:
                    print("Recommender not available, using fallback")
                    # Save order without AI data
                    order = db.add_order(order_data)

                return redirect(url_for('order_success', order_id=order['id']))

            except Exception as ai_error:
                print(f"AI prediction failed: {str(ai_error)}")
                import traceback
                traceback.print_exc()
                flash("AI recommendation temporarily unavailable. Order created successfully.", "warning")
                
                # Fallback: Save order without AI data
                order = db.add_order(order_data)
                return redirect(url_for('order_success', order_id=order['id']))

        except (KeyError, ValueError, TypeError) as e:
            print(f"Form validation error: {str(e)}")
            flash(f"Please check all required fields: {str(e)}", "error")
            return render_template('order.html')
        except Exception as e:
            print(f"General error: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f"Error creating order: {str(e)}", "error")
            return redirect(url_for('order'))

    return render_template('order.html')


@app.route('/order/success/<int:order_id>')
def order_success(order_id):
    order = db.get_order(order_id)
    if order:
        return render_template('order_success.html', order=order)
    return redirect(url_for('index'))

@app.route('/tracking', methods=['GET', 'POST'])
def tracking():
    if request.method == 'POST':
        tracking_number = request.form['tracking_number']
        order = db.get_order_by_tracking(tracking_number)
        if order:
            return render_template('tracking.html', order=order, found=True)
        return render_template('tracking.html', found=False, tracking_number=tracking_number)
    return render_template('tracking.html', found=None)

@app.route('/admin')
def admin_dashboard():
    pending_orders = db.get_pending_orders()
    dispatched_orders = db.get_dispatched_orders()
    delivered_orders = db.get_delivered_orders()
    return render_template('admin_dashboard.html',
                           pending_orders=pending_orders,
                           dispatched_orders=dispatched_orders,
                           delivered_orders=delivered_orders)

@app.route('/admin/order/<int:order_id>', methods=['GET', 'POST'])
def admin_order_detail(order_id):
    order = db.get_order(order_id)
    if not order:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        selected_courier = request.form['courier']
        updates = {
            'courier': selected_courier,
            'status': 'dispatched',
            'dispatched_at': datetime.now().isoformat()
        }
        db.update_order(order_id, updates)
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_order_detail.html', order=order)

@app.route('/admin/order/<int:order_id>/mark-delivered')
def mark_delivered(order_id):
    order = db.get_order(order_id)
    if order and order['status'] == 'dispatched':
        db.update_order(order_id, {
            'status': 'delivered',
            'delivered_at': datetime.now().isoformat()
        })
    return redirect(url_for('admin_dashboard'))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json
        
        # CRITICAL FIX: Ensure data has the correct structure for prediction
        prediction_data = {
            'DeliveryAddressCountry': data.get('delivery_country', 'DE'),
            'WarehouseName': data.get('sender_country', 'Germany'),
            'TotalWeight': max(0.1, float(data.get('product_weight', 1.0))),
            'PackCount': max(1, int(data.get('product_quantity', 1))),
            'IsPallet': bool(data.get('is_pallet', False)),
            'DeliveryType': data.get('delivery_urgency', 'Standard'),
            'ItemType': data.get('product_name', 'General'),
            
            # ADD THESE MISSING FEATURES:
            'package_count': max(1, int(data.get('product_quantity', 1))),
            'has_dangerous_goods': bool(data.get('has_dangerous_goods', False)),
            'declared_value': max(0.0, float(data.get('package_value', 0.0)))
        }
        
        if recommender:
            recommendation, confidence = recommender.recommend_courier(prediction_data)
            all_probs = recommender.get_all_courier_probabilities(prediction_data)
            
            return jsonify({
                'recommended_courier': recommendation,
                'confidence': float(confidence),
                'all_probabilities': dict(all_probs)
            })
        else:
            return jsonify({
                'recommended_courier': 'DPD DE',
                'confidence': 0.0,
                'all_probabilities': {},
                'error': 'AI recommender not available'
            })
        
    except Exception as e:
        print(f"API recommendation error: {str(e)}")
        return jsonify({
            'error': str(e),
            'recommended_courier': 'DPD DE',
            'confidence': 0.0,
            'all_probabilities': {}
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run(debug=True, host='0.0.0.0', port=5000)