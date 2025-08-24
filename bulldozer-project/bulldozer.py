import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
try:
    model = joblib.load("ideal_model.pkl")
except FileNotFoundError:
    st.error("Model file 'ideal_model.pkl' not found. Please ensure the file is in the same directory.")
    st.stop()

st.title("🚜 Predicting the Sale Price of Bulldozers using Machine Learning")
st.divider()
st.write("This app predicts bulldozer sale prices using machine learning. Enter the values below and click the prediction button.")
st.divider()

# Create input fields for the main features
col1, col2 = st.columns(2)

with col1:
    year_made = st.number_input("Year Made", min_value=1000, max_value=2013, value=1000)
    sale_year = st.number_input("Sale Year", min_value=1989, max_value=2011, value=1989)
    
with col2:
    product_size = st.selectbox("Product Size", 
                               options=["Medium", "Large / Medium", "Small", "Mini", "Large", "Compact"],
                               index=2)
    machine_hours = st.number_input("Machine Hours", min_value=0, max_value=2483300, value=0)

# Additional important features
with st.expander("Additional Features (Optional)"):
    col3, col4 = st.columns(2)
    with col3:
        usage_band = st.selectbox("Usage Band", 
                                 options=["Low", "Medium", "High"], 
                                 index=1)
        enclosure = st.selectbox("Enclosure", 
                               options=["EROPS w AC", "EROPS", "OROPS", "NO ROPS", "EROPS AC"], 
                               index=0)
    with col4:
        hydraulics = st.selectbox("Hydraulics", 
                                options=["2 Valve", "3 Valve", "4 Valve", "Auxiliary", "Standard"], 
                                index=0)
        drive_system = st.selectbox("Drive System", 
                                  options=["4WD", "2WD", "All Wheel Drive", "No"], 
                                  index=0)

st.divider()

def create_feature_vector():
    """
    Create a feature vector that matches the model's expected input format.
    Features must be in the exact same order as during training.
    """
    # Exact feature order from training (from the document)
    feature_order = [
        'SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'YearMade',
        'MachineHoursCurrentMeter', 'UsageBand', 'fiModelDesc', 'fiBaseModel',
        'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
        'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
        'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick',
        'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width',
        'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper',
        'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System',
        'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width',
        'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type',
        'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type',
        'Steering_Controls', 'saleYear', 'saleMonth', 'saleDay', 'saleDayOfWeek',
        'saleDayOfYear', 'auctioneerID_is_missing',
        'MachineHoursCurrentMeter_is_missing', 'UsageBand_is_missing',
        'fiModelDesc_is_missing', 'fiBaseModel_is_missing',
        'fiSecondaryDesc_is_missing', 'fiModelSeries_is_missing',
        'fiModelDescriptor_is_missing', 'ProductSize_is_missing',
        'fiProductClassDesc_is_missing', 'state_is_missing',
        'ProductGroup_is_missing', 'ProductGroupDesc_is_missing',
        'Drive_System_is_missing', 'Enclosure_is_missing', 'Forks_is_missing',
        'Pad_Type_is_missing', 'Ride_Control_is_missing', 'Stick_is_missing',
        'Transmission_is_missing', 'Turbocharged_is_missing',
        'Blade_Extension_is_missing', 'Blade_Width_is_missing',
        'Enclosure_Type_is_missing', 'Engine_Horsepower_is_missing',
        'Hydraulics_is_missing', 'Pushblock_is_missing', 'Ripper_is_missing',
        'Scarifier_is_missing', 'Tip_Control_is_missing', 'Tire_Size_is_missing',
        'Coupler_is_missing', 'Coupler_System_is_missing',
        'Grouser_Tracks_is_missing', 'Hydraulics_Flow_is_missing',
        'Track_Type_is_missing', 'Undercarriage_Pad_Width_is_missing',
        'Stick_Length_is_missing', 'Thumb_is_missing', 'Pattern_Changer_is_missing',
        'Grouser_Type_is_missing', 'Backhoe_Mounting_is_missing',
        'Blade_Type_is_missing', 'Travel_Controls_is_missing',
        'Differential_Type_is_missing', 'Steering_Controls_is_missing'
    ]
    
    # Create feature values in the exact order
    feature_values = []
    
    # Encode categorical variables
    product_size_mapping = {"Mini": 0, "Small": 1, "Compact": 2, "Medium": 3, "Large / Medium": 4, "Large": 5}
    usage_band_mapping = {"Low": 0, "Medium": 1, "High": 2}
    
    for feature in feature_order:
        if feature == 'YearMade':
            feature_values.append(year_made)
        elif feature == 'saleYear':
            feature_values.append(sale_year)
        elif feature == 'MachineHoursCurrentMeter':
            feature_values.append(machine_hours)
        elif feature == 'ProductSize':
            feature_values.append(product_size_mapping.get(product_size, 3))
        elif feature == 'UsageBand':
            feature_values.append(usage_band_mapping.get(usage_band, 1))
        elif feature == 'saleMonth':
            feature_values.append(6)  # Default to June
        elif feature == 'saleDay':
            feature_values.append(15)  # Default to mid-month
        elif feature == 'saleDayOfWeek':
            feature_values.append(3)  # Default to Wednesday
        elif feature == 'saleDayOfYear':
            feature_values.append(165)  # Default to mid-year
        elif feature == 'MachineHoursCurrentMeter_is_missing':
            feature_values.append(0 if machine_hours > 0 else 1)
        elif feature == 'ProductSize_is_missing':
            feature_values.append(0)  # We always have product size
        elif feature == 'UsageBand_is_missing':
            feature_values.append(0)  # We always have usage band
        elif feature == 'auctioneerID_is_missing':
            feature_values.append(1)  # We don't have auctioneer ID
        elif feature.endswith('_is_missing'):
            feature_values.append(1)  # Most features are missing
        else:
            feature_values.append(0)  # Default value for all other features
    
    return feature_values

if st.button("🔮 Predict Price"):
    try:
        st.balloons()
        
        # Create feature vector in the exact order expected by the model
        feature_values = create_feature_vector()
        
        # Convert to numpy array (the format most models expect)
        X = np.array([feature_values])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Display result
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")
        
        # Show some additional info
        st.info(f"""
        **Prediction Details:**
        - Year Made: {year_made}
        - Sale Year: {sale_year}
        - Product Size: {product_size}
        - Machine Hours: {machine_hours:,}
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("This might be due to a mismatch between the model's expected features and the provided inputs.")
        st.write("Debug info:")
        st.write(f"Feature vector length: {len(create_feature_vector()) if 'create_feature_vector' in locals() else 'N/A'}")
        
else:
    st.info("👆 Enter the bulldozer details above and click 'Predict Price' to get an estimate")

# Add some helpful information
with st.expander("ℹ️ About this App"):
    st.write("""
    This app uses a machine learning model to predict bulldozer sale prices based on various features.
    
    **Key Features:**
    - Year Made: The manufacturing year of the bulldozer
    - Sale Year: The year when the bulldozer was sold
    - Product Size: The size category of the bulldozer
    - Machine Hours: Total operating hours on the machine
    
    **Note:** This is a simplified version of the prediction model. In practice, the model uses
    many more features for more accurate predictions.
    """)