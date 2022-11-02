# -*- coding: utf-8 -*-
"""datacleaning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zQLMI5mGN3Vedx_h4EyFQHGtqp62cECB
"""

import pandas as pd
import datetime as dt

class data_cleaning:

  US_list = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
  Canada_list = ['AB','BC','MB','NB','NL','NT','NS','NU','ON','PE','QC','SK','YT']

  acutual_mode_list = ['LTL','TL','INTERMODAL']

  required_column = ['SHIPMENT ID', 'CUSTOMER', 'DISTANCE', 'CASES', 'WEIGHT', 'VOLUME',
       'SOURCE LOCATION ID', 'ORIGIN NAME', 'ORIGIN CITY', 'ORIGIN STATE',
       'ORIGIN ZIP', 'DEST LOCATION ID', 'CONSIGNEE NAME', 'DEST CITY',
       'DEST STATE', 'DEST ZIP', 'ACTUAL CARRIER', 'ACTUAL MODE',
       'ACTUAL EQUIP', 'LINEHAUL COSTS','TOTAL ACTUAL COST', 'PU_APPT','DL_APPT',
       'LINEHAUL UNIT COSTS','PC_MILER_PRACTICAL_MILEAGE',
       'SPOT_AVG_LINEHAUL_RATE', 'SPOT_LOW_LINEHAUL_RATE',
       'SPOT_HIGH_LINEHAUL_RATE', 'SPOT_FUEL_SURCHARGE', 'SPOT_TIME_FRAME',
       'SPOT_ORIGIN_GEO_EXPANSION', 'SPOT_DESTINATION_GEO_EXPANSION',
       'SPOT_NUMBER_OF_COMPANIES', 'SPOT_NUMBER_OF_REPORTS',
       'SPOT_LINEHAUL_RATE_STDDEV', 'SPOT_YOUR_OWN_AVG_LINEHAUL_RATE',
       'SPOT_YOUR_OWN_NUMBER_OF_REPORTS', 'SPOT_ERROR',
       'CONTRACT_AVG_LINEHAUL_RATE', 'CONTRACT_LOW_LINEHAUL_RATE',
       'CONTRACT_HIGH_LINEHAUL_RATE', 'CONTRACT_FUEL_SURCHARGE',
       'CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL', 'CONTRACT_TIME_FRAME',
       'CONTRACT_ORIGIN_GEO_EXPANSION', 'CONTRACT_DESTINATION_GEO_EXPANSION',
       'CONTRACT_NUMBER_OF_COMPANIES', 'CONTRACT_NUMBER_OF_REPORTS',
       'CONTRACT_LINEHAUL_RATE_STDDEV', 'CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE',
       'CONTRACT_YOUR_OWN_NUMBER_OF_REPORTS', 'CONTRACT_ERROR', 'ORIG_CITY',
       'ORIG_STATE', 'ORIG_POSTAL_CODE', 'DEST_CITY', 'DEST_STATE',
       'DEST_POSTAL_CODE', 'TRUCK_TYPE', 'RUN_ID', 'SOURCE_FILE_NAME']

  # Import data
  def __init__(self, MT_data,DAT_data):
    self.MT_data = MT_data
    self.DAT_data = DAT_data
    self.clean_MT_data()
    self.clean_DAT_data()
    self.final_data()

  #Clean MT_data 
  def clean_MT_data(self):
    # Create a column called ID_OR_OPTIONAL_FIELD, which join first three letter of ORIGIN zip
    # and the first three letter of DEST zip and connect them by using "_"
    self.MT_data['ID_OR_OPTIONAL_FIELD'] = self.MT_data['ORIGIN ZIP'].astype(str).str[:3] + '_' + self.MT_data['DEST ZIP'].astype(str).str[:3] 
    self.MT_data['ID_OR_OPTIONAL_FIELD'] = self.MT_data['ID_OR_OPTIONAL_FIELD'].astype(str)

    #Remove Columns of 'Unnamed: 0.1','Unnamed: 0'
    self.MT_data = self.MT_data.drop(['Unnamed: 0.1','Unnamed: 0'], axis = 1)

    # Drop the row if na is found in the MT_data['DISTANCE']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['DISTANCE'])]

    # Drop the row if na is found in the MT_data['WEIGHT]
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['WEIGHT'])]

    # Drop the row if na is found in the MT_data['VOLUME']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['VOLUME'])]

    # Drop the row if na is found in the MT_data['ORIGIN CITY']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ORIGIN CITY'])]

    # Only conisder shipment has Original zip code in USA and Canada
    # Drop the row if ns is found in the MT_data['ORIGIN STATE']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ORIGIN STATE'])]
    self.MT_data = self.MT_data.reset_index(drop=True)

    State_list = self.US_list + self.Canada_list

    for i in range(len(self.MT_data)):
        if self.MT_data.loc[i,'ORIGIN STATE'] in State_list:
            pass
        else:
            self.MT_data = self.MT_data.drop(i)
        
    # Drop the row if na is found in the MT_data['ORIGIN ZIP']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ORIGIN ZIP'])]
    self.MT_data = self.MT_data.reset_index(drop=True)
    # Change 4 digits zips to 5 digis
    for i in (range(len(self.MT_data['ORIGIN ZIP']))):
        zipCode = str(self.MT_data.loc[i,'ORIGIN ZIP'])
        if zipCode.isdigit():
            if len(zipCode) < 5:
                MT_data.loc[i,'ORIGIN ZIP'] = "0" * (5 - len(zipCode)) + zipCode
        else:
            pass
            
    # Actual mode will only consider LTL, and TL
    # Drop the row if ns is found in the MT_data['ACTUAL MODE']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ACTUAL MODE'])]
    self.MT_data['ACTUAL MODE'] = self.MT_data['ACTUAL MODE'].str.split('.').str[-1] 
    self.MT_data['ACTUAL MODE'] = self.MT_data['ACTUAL MODE'].str.upper()
    self.MT_data = self.MT_data.reset_index(drop=True)

    for i in range(len(self.MT_data)):
        if self.MT_data.loc[i,'ACTUAL MODE'] in self.acutual_mode_list:
            pass
        else:
          self.MT_data = self.MT_data.drop(i)
        
    # Remove any nan cell in ACTUAL EQUIP
    # Tokenize part information from the original cells
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ACTUAL EQUIP'])]
    self.MT_data['ACTUAL EQUIP'] = self.MT_data['ACTUAL EQUIP'].str.split('.').str[-1]

        
    # Remove any nan cell in LINEHAUL COSTS
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['LINEHAUL COSTS'])]

    # Remove any nan cell in FUEL COSTS
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['FUEL COSTS'])]

    # Remove any nan cell in ACC. COSTS
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['ACC. COSTS'])]

    # Remove any nan cell in total ACTUAL COST
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['TOTAL ACTUAL COST'])]

    # Add a LINEHAUL UNIT COSTS to the dataframe
    self.MT_data['LINEHAUL UNIT COSTS'] = self.MT_data['LINEHAUL COSTS']/self.MT_data['DISTANCE']

    # Drop the row if na is found in the MT_data['DEST CITY']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['DEST CITY'])]

    # Only conisder shipment has dest zip code in USA and Canada
    # Drop the row if ns is found in the MT_data['ORIGIN STATE']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['DEST STATE'])]
    self.MT_data = self.MT_data.reset_index(drop=True)

    for i in range(len(self.MT_data)):
        if self.MT_data.loc[i,'DEST STATE'] in State_list:
            pass
        else:
            self.MT_data = self.MT_data.drop(i)
        
    # Drop the row if na is found in the MT_data['DEST ZIP']
    self.MT_data = self.MT_data[pd.notnull(self.MT_data['DEST ZIP'])]
    self.MT_data = self.MT_data.reset_index(drop=True)
    # Change 4 digits zips to 5 digis
    for i in (range(len(self.MT_data['DEST ZIP']))):
        zipCode = str(self.MT_data.loc[i,'DEST ZIP'])
        if zipCode.isdigit():
            if len(zipCode) < 5:
                self.MT_data.loc[i,'DEST ZIP'] = "0" * (5 - len(zipCode)) + zipCode
        else:
            pass

    # Clean the column of PU_APPT as a new column called File_paried_data
    self.MT_data['File_paried_data'] = pd.to_datetime(self.MT_data['PU_APPT'].str.split(' ').str[0], errors = 'coerce')
    self.MT_data['File_paried_data'] = self.MT_data['File_paried_data'].dt.strftime('%Y-%m-%d')

  #DATA_data Cleaning
  def clean_DAT_data(self):
    # Set ID_OR_OPTIONAL_FIELD as an objective column
    self.DAT_data['ID_OR_OPTIONAL_FIELD'] = self.DAT_data['ID_OR_OPTIONAL_FIELD'].astype(str)
    
    # Clean the column of File_data  and add year '2022' before the month
    self.DAT_data['File_data'] = self.DAT_data['SOURCE_FILE_NAME'].str.split(' ').str[-1]
    self.DAT_data['File_data'] = self.DAT_data['File_data'].str.split('.').str[0]
    self.DAT_data['File_data'] = '2022-' + self.DAT_data['File_data']
    self.DAT_data['File_data'] = pd.to_datetime(self.DAT_data['File_data'])

    # Remove columns have ALL na
    self.DAT_data = self.DAT_data.drop('NOTE',axis = 1)


    # Drop the row if na is found in the DAT_data['SPOT_AVG_LINEHAUL_RATE']
    self.DAT_data = self.DAT_data[pd.notnull(self.DAT_data['SPOT_AVG_LINEHAUL_RATE'])]

    # Drop the row if na is found in the DAT_data['SPOT_TIME_FRAME']
    self.DAT_data = self.DAT_data[pd.notnull(self.DAT_data['SPOT_TIME_FRAME'])]

    # Drop the row if na is found in the DAT_data['CONTRACT_AVG_LINEHAUL_RATE']
    self.DAT_data = self.DAT_data[pd.notnull(self.DAT_data['CONTRACT_AVG_LINEHAUL_RATE'])]

    # Drop the row if na is found in the DAT_data['CONTRACT_TIME_FRAME']
    self.DAT_data = self.DAT_data[pd.notnull(self.DAT_data['CONTRACT_TIME_FRAME'])]
  
  def final_data(self):
    #Join the table by setting ID_OR_OPTIONAL_FIELD as the key
    merge_df_remove_NA = self.MT_data.merge(self.DAT_data, on = 'ID_OR_OPTIONAL_FIELD', how = 'left')
    merge_df_remove_NA = merge_df_remove_NA[pd.notnull(merge_df_remove_NA['File_data'])] 
    merge_df_remove_NA = merge_df_remove_NA.reset_index(drop=True)

    for i in range(len(merge_df_remove_NA)):
        if pd.to_datetime(merge_df_remove_NA.loc[i, 'File_paried_data']) <= pd.to_datetime(merge_df_remove_NA.loc[i, 'File_data']):
            merge_df_remove_NA = merge_df_remove_NA.drop(index=i)
    self.cleaned_df = merge_df_remove_NA[self.required_column]
    self.LTL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE'] == 'LTL']
    self.TL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE'] == 'TL']
    self.INTERMODAL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE'] == 'INTERMODAL']
    #return cleaned_df,LTL_Cleaned_Data,TL_Cleaned_Data,INTERMODAL_Cleaned_Data

from google.colab import drive
drive.mount('/content/gdrive')

path = "/content/gdrive/MyDrive/ML-Transportation-Rate/Raw_Data_Past_Four_Month"
MT_data_ori = pd.read_csv(path +'/Merged_DataFrame_hash_Version_addition.csv')
DAT_data = pd.read_excel(path +'/DAT_Rates_from_DB-9-21.xlsx')

MT_data = MT_data_ori.sample(1000)

dc = data_cleaning(MT_data,DAT_data)

dc.cleaned_df.head()

dc.TL_Cleaned_Data.head()