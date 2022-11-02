<img width="635" alt="Screenshot 2022-11-01 at 8 18 45 PM" src="https://user-images.githubusercontent.com/89152255/199372175-77f84832-ad8e-47d4-835d-192ecd641811.png">


# Quick Navigation

**[Goal](#Goal)**

**[Proposal](Proposal)**

**[Data](#Data)**

**[Environment](#Environment)**

**[Package](#Package)**


**[Resource](#Resource)**

**[Contract](#Contract)**



# Goal
The Goal of the project is to developing a deep learning model for a logistic company to predict transportation rate. The company is able to use the model to predict the markt transportation rate.

# Proposal
https://docs.google.com/presentation/d/1620GSp25dgO1NSkLDJX9eVQEMCR3U1WrjcSCX4yYud0/edit?usp=sharing


# Data
The Data is provide by DHL Supply Chain. The training data has been masked based on DHL Policy. 

## Historical Shipment Data (MT_Data)

**Description:** The dataset includes 4 month historical shipment information

**Column:** 31

**Rows:** 4,963,508

**Column Names:**[Unnamed: 0.1', 'SHIPMENT ID', 'CUSTOMER', 'DISTANCE', 'CASES', 'WEIGHT', 'VOLUME', 'SOURCE LOCATION ID', 'ORIGIN NAME','ORIGIN CITY', 'ORIGIN STATE', 'ORIGIN ZIP', 'DEST LOCATION ID','CONSIGNEE NAME', 'DEST CITY', 'DEST STATE', 'DEST ZIP', 'ACTUAL CARRIER', 'ACTUAL MODE', 'ACTUAL EQUIP', 'LINEHAUL COSTS', 'FUEL COSTS', 'ACC. COSTS', 'TOTAL ACTUAL COST', 'PU_APPT', 'DL_APPT', 'PU_ARRIVAL (X3)', 'PU_DEPARTED (AF)', 'DL_ARRIVAL (X1)', 'DL_DEPARTED (D1)', 'Insert Date']

## Historical Quoted Price Data (DAT_Data)

**Description:** The dataset includes 4 month historical quoted price data information

**Column:** 40

**Rows:** 63,486

**Column Names:** ['PC_MILER_PRACTICAL_MILEAGE', 'SPOT_AVG_LINEHAUL_RATE', 'SPOT_LOW_LINEHAUL_RATE', 'SPOT_HIGH_LINEHAUL_RATE', 'SPOT_FUEL_SURCHARGE', 'SPOT_TIME_FRAME', 'SPOT_ORIGIN_GEO_EXPANSION','SPOT_DESTINATION_GEO_EXPANSION', 'SPOT_NUMBER_OF_COMPANIES','SPOT_NUMBER_OF_REPORTS', 'SPOT_LINEHAUL_RATE_STDDEV','SPOT_YOUR_OWN_AVG_LINEHAUL_RATE', 'SPOT_YOUR_OWN_NUMBER_OF_REPORTS','SPOT_ERROR', 'CONTRACT_AVG_LINEHAUL_RATE','CONTRACT_LOW_LINEHAUL_RATE', 'CONTRACT_HIGH_LINEHAUL_RATE','CONTRACT_FUEL_SURCHARGE', 'CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL', 'CONTRACT_TIME_FRAME', 'CONTRACT_ORIGIN_GEO_EXPANSION','CONTRACT_DESTINATION_GEO_EXPANSION', 'CONTRACT_NUMBER_OF_COMPANIES','CONTRACT_NUMBER_OF_REPORTS', 'CONTRACT_LINEHAUL_RATE_STDDEV','CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE','CONTRACT_YOUR_OWN_NUMBER_OF_REPORTS', 'CONTRACT_ERROR', 'ORIG_CITY'， 'ORIG_STATE', 'ORIG_POSTAL_CODE', 'DEST_CITY', 'DEST_STATE'， 'DEST_POSTAL_CODE', 'TRUCK_TYPE', 'ID_OR_OPTIONAL_FIELD', 'NOTE'， 'RUN_ID', 'SOURCE_FILE_NAME', 'DWH_INSERT_DATE']


# Environment
The primiary language of the project is **Python**

**Colab** or **ACCRE** are major Virtual Machine for model training

The project will use Notebook, Pycharm, and other IDEs

# Resource

**Model:** https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

# Contract
**Peter Tian:** zhengqi.tian@vanderbilt.edu



**Shuyang Lin:** shuyang.lin@vanderbilt.edu

**Weixi Chen:** weixi.chen@vanderbilt.edu
