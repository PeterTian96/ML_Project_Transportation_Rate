#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:53:40 2022

@author: petertian
"""

import numpy
import pandas as pd
import xlrd
import unicodedata
import numpy as np
import math

class Data_cleaning:
  """
  This class is used to clean the MT_data and DAT_data, and combine the two datasets as the final clean dataset.
  """

  def __init__ (self, MT_data, DAT_data, zip_data, mapping):
    self.mt = MT_data
    self.dat = DAT_data
    self.zip_data = zip_data
    self.mapping = mapping
    self.us_list = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    self.canada_list = ['AB','BC','MB','NB','NL','NT','NS','NU','ON','PE','QC','SK','YT']
    self.actual_mode_list = ['LTL','TL','INTERMODAL']
    self.mt_required_column = ['CUSTOMER', 'DISTANCE', 'CASES', 'WEIGHT', 'VOLUME', 
        'ACTUAL CARRIER', 'ACTUAL MODE','ACTUAL EQUIP', 'PU_APPT', 'DL_APPT', 'LINEHAUL COSTS']
    self.dat_required_column = ['SPOT_AVG_LINEHAUL_RATE', 'SPOT_LOW_LINEHAUL_RATE',
       'SPOT_HIGH_LINEHAUL_RATE', 'SPOT_FUEL_SURCHARGE', 'SPOT_TIME_FRAME',
       'SPOT_YOUR_OWN_AVG_LINEHAUL_RATE', 'CONTRACT_AVG_LINEHAUL_RATE', 
       'CONTRACT_LOW_LINEHAUL_RATE', 'CONTRACT_HIGH_LINEHAUL_RATE', 'CONTRACT_FUEL_SURCHARGE',
       'CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL', 'CONTRACT_TIME_FRAME',
       'CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE']
    self.dummy_column = ['CUSTOMER', 'ACTUAL CARRIER', 'ACTUAL MODE', 'ACTUAL EQUIP']

  def clean_mt_data(self):
    # first, drop the data that has wrong origin state or destination state
    for i in self.mt_required_column:
      if i not in self.mt.columns:
        self.mt = pd.concat([self.mt, pd.DataFrame(columns=[i])], sort=False)
        self.mt[i] = self.mt[i].replace(np.nan,0)

    self.mt['ID_OR_OPTIONAL_FIELD'] = self.mt['ORIGIN ZIP'].astype(str).str[:3] + '_' + self.mt['DEST ZIP'].astype(str).str[:3] 
    self.mt['ID_OR_OPTIONAL_FIELD'] = self.mt['ID_OR_OPTIONAL_FIELD'].astype(str)

    self.mt = self.mt[pd.notnull(self.mt['CUSTOMER'])]

    # Drop the row if na is found in the MT_data['DISTANCE']
    self.mt = self.mt[pd.notnull(self.mt['DISTANCE'])]

    # Drop the row if na is found in the MT_data['WEIGHT]
    self.mt = self.mt[pd.notnull(self.mt['WEIGHT'])]

    # Drop the row if na is found in the MT_data['VOLUME']
    self.mt = self.mt[pd.notnull(self.mt['VOLUME'])]

    self.mt = self.mt[pd.notnull(self.mt['LINEHAUL COSTS'])]
    self.mt = self.mt.reset_index(drop=True)
    self.mt = self.mt[pd.notnull(self.mt['ORIGIN ZIP'])]
    self.mt = self.mt.reset_index(drop=True)
    self.mt['ORIGIN ZIP'] = [str(i) for i in self.mt['ORIGIN ZIP']]
    self.mt['ORIGIN ZIP'] = ["0" * (5 - len(i)) + i for i in self.mt['ORIGIN ZIP']]
    self.mt = self.mt[pd.notnull(self.mt['ACTUAL CARRIER'])]
    self.mt = self.mt[pd.notnull(self.mt['ACTUAL MODE'])]
    self.mt['ACTUAL MODE'] = self.mt['ACTUAL MODE'].str.split('.').str[-1] 
    self.mt['ACTUAL MODE'] = self.mt['ACTUAL MODE'].str.upper()
    self.mt = self.mt.reset_index(drop=True)
    self.mt = self.mt[self.mt['ACTUAL MODE'].isin(self.actual_mode_list)]
    self.mt = self.mt[pd.notnull(self.mt['ACTUAL EQUIP'])]
    self.mt['ACTUAL EQUIP'] = self.mt['ACTUAL EQUIP'].str.split('.').str[-1]

    # Remove any nan cell in LINEHAUL COSTS
    self.mt = self.mt[pd.notnull(self.mt['LINEHAUL COSTS'])]

    self.mt = self.mt.reset_index(drop=True)

    self.mt = self.mt[pd.notnull(self.mt['DEST ZIP'])]
    self.mt = self.mt.reset_index(drop=True)
    self.mt['DEST ZIP'] = [str(i) for i in self.mt['DEST ZIP']]
    self.mt['DEST ZIP'] = ["0" * (5 - len(i)) + i for i in self.mt['DEST ZIP']]

    self.mt = self.mt[pd.notnull(self.mt['PU_APPT'])]
    self.mt = self.mt[pd.notnull(self.mt['DL_APPT'])]
    self.mt = self.mt.drop(self.mt[self.mt['DL_APPT'] == '2021-11-122'].index)
    self.mt = self.mt.reset_index(drop=True)
    # Clean the column of PU_APPT as a new column called File_paried_data
    self.mt['File_paried_data'] = pd.to_datetime(self.mt['PU_APPT'].str.split(' ').str[0], errors = 'coerce')
    self.mt['File_paried_data'] = self.mt['File_paried_data'].dt.strftime('%Y-%m-%d')

    # merge the zip data into MT_data via origin information and destination information, respectively
    self.mt = pd.merge(self.mt, self.zip_data, how = 'left', left_on = ['ORIGIN ZIP'], right_on = ['zipcode'])
    self.mt = pd.merge(self.mt, self.zip_data, how = 'left', left_on = ['DEST ZIP'], right_on = ['zipcode'])
    # delete unwanted columns and change the column names
    self.mt = self.mt.drop(['zipcode_x', 'state_x', 'zipcode_y', 'state_y'], axis = 1)
    self.mt = self.mt.rename(columns={'lng_x':'ORIGIN_LNG', 'lat_x':'ORIGIN_LAT', 'lng_y':'DEST_LNG', 'lat_y':'DEST_LAT'})
    # delete data that doesn't match
    self.mt = self.mt.dropna(subset = ['ORIGIN_LNG', 'ORIGIN_LAT', 'DEST_LNG', 'DEST_LAT'])
    self.mt = self.mt[self.mt_required_column + ['File_paried_data', 'ID_OR_OPTIONAL_FIELD', 'ORIGIN_LAT', 'ORIGIN_LNG', 'DEST_LAT', 'DEST_LNG']]
    return self.mt

  def clean_dat_data(self):
    for i in self.dat_required_column:
      if i not in self.dat.columns:
        self.dat = pd.concat([self.dat, pd.DataFrame(columns=[i])], sort=False)
        self.dat[i] = self.dat[i].replace(np.nan,0)

    # Set ID_OR_OPTIONAL_FIELD as an objective column
    self.dat['ID_OR_OPTIONAL_FIELD'] = self.dat['ID_OR_OPTIONAL_FIELD'].astype(str)
    
    # Clean the column of File_data  and add year '2022' before the month
    self.dat['File_data'] = self.dat['SOURCE_FILE_NAME'].str.split(' ').str[-1]
    self.dat['File_data'] = self.dat['File_data'].str.split('.').str[0]
    self.dat['File_data'] = '2022-' + self.dat['File_data']
    self.dat['File_data'] = pd.to_datetime(self.dat['File_data'])

    # Remove columns have ALL na
    self.dat = self.dat.drop('NOTE',axis = 1)

    # Drop the row if na is found in the DAT_data['SPOT_AVG_LINEHAUL_RATE']
    self.dat = self.dat[pd.notnull(self.dat['SPOT_AVG_LINEHAUL_RATE'])]
    self.dat = self.dat[pd.notnull(self.dat['SPOT_LOW_LINEHAUL_RATE'])]
    self.dat = self.dat[pd.notnull(self.dat['SPOT_HIGH_LINEHAUL_RATE'])]
    self.dat = self.dat[pd.notnull(self.dat['SPOT_FUEL_SURCHARGE'])]

    # Drop the row if na is found in the DAT_data['SPOT_TIME_FRAME']
    self.dat = self.dat[pd.notnull(self.dat['SPOT_TIME_FRAME'])]

    # Drop the row if na is found in the DAT_data['CONTRACT_AVG_LINEHAUL_RATE']
    self.data = self.dat[pd.notnull(self.dat['CONTRACT_AVG_LINEHAUL_RATE'])]
    self.data = self.dat[pd.notnull(self.dat['CONTRACT_LOW_LINEHAUL_RATE'])]
    self.data = self.dat[pd.notnull(self.dat['CONTRACT_HIGH_LINEHAUL_RATE'])]
    self.data = self.dat[pd.notnull(self.dat['CONTRACT_FUEL_SURCHARGE'])]

    # Drop the row if na is found in the DAT_data['CONTRACT_TIME_FRAME']
    self.dat = self.dat[pd.notnull(self.dat['CONTRACT_TIME_FRAME'])]
    self.dat['SPOT_YOUR_OWN_AVG_LINEHAUL_RATE'] = self.dat['SPOT_YOUR_OWN_AVG_LINEHAUL_RATE'].replace(np.nan,0)
    self.dat['CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE'] = self.dat['CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE'].replace(np.nan,0)
    self.dat = self.dat.reset_index(drop = True)
    self.dat = self.dat[self.dat_required_column + ['ID_OR_OPTIONAL_FIELD', 'File_data']]
    return self.dat
  
  def final_data(self):
    self.mt = self.clean_mt_data()
    self.dat = self.clean_dat_data()
    #Join the table by setting ID_OR_OPTIONAL_FIELD as the key
    merge_df_remove_NA = self.mt.merge(self.dat, on = 'ID_OR_OPTIONAL_FIELD', how = 'left')
    merge_df_remove_NA = merge_df_remove_NA[pd.notnull(merge_df_remove_NA['File_data'])] 
    merge_df_remove_NA = merge_df_remove_NA.reset_index(drop=True)

    merge_df_remove_NA = merge_df_remove_NA[pd.to_datetime(merge_df_remove_NA['File_paried_data']) <= pd.to_datetime(merge_df_remove_NA['File_data'])]
    merge_df_remove_NA = merge_df_remove_NA.drop(['File_paried_data', 'ID_OR_OPTIONAL_FIELD', 'File_data'], axis = 1)

    for col in self.dummy_column:
      temp_col = pd.get_dummies(merge_df_remove_NA[col], prefix = col)
      merge_df_remove_NA = pd.concat([merge_df_remove_NA, temp_col],axis=1)
    merge_df_remove_NA.drop(self.dummy_column, axis=1, inplace=True)
    for col in ['PU_APPT','DL_APPT']:
      temp_col = (merge_df_remove_NA[col].astype('datetime64[D]') - merge_df_remove_NA[col].astype('datetime64[Y]'))/ np.timedelta64(1, 'D')
      merge_df_remove_NA.drop(col, axis=1, inplace=True)
      merge_df_remove_NA = pd.concat([merge_df_remove_NA, temp_col],axis=1)
    for col in self.mapping['ACTUAL CARRIER']:
      if col not in merge_df_remove_NA.columns:
        merge_df_remove_NA = pd.concat([ merge_df_remove_NA, pd.DataFrame(columns=[col])], axis = 1, sort=False)
        merge_df_remove_NA[col] = merge_df_remove_NA[col].replace(np.nan,0)
    for col in self.mapping['ACTUAL EQUIP']:
      if col not in merge_df_remove_NA.columns:
        merge_df_remove_NA = pd.concat([ merge_df_remove_NA, pd.DataFrame(columns=[col])], axis = 1, sort=False)
        merge_df_remove_NA[col] = merge_df_remove_NA[col].replace(np.nan,0)
    for col in self.mapping['ACTUAL MODE']:
      if col not in merge_df_remove_NA.columns:
        merge_df_remove_NA = pd.concat([ merge_df_remove_NA, pd.DataFrame(columns=[col])], axis = 1, sort=False)
        merge_df_remove_NA[col] = merge_df_remove_NA[col].replace(np.nan,0)
    for col in self.mapping['CUSTOMER']:
      if col not in merge_df_remove_NA.columns:
        merge_df_remove_NA = pd.concat([ merge_df_remove_NA, pd.DataFrame(columns=[col])], axis = 1, sort=False)
        merge_df_remove_NA[col] = merge_df_remove_NA[col].replace(np.nan,0)

    merge_df_remove_NA = merge_df_remove_NA.astype('float64')

    self.cleaned_df = merge_df_remove_NA
    self.LTL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE_LTL'] == 1.0]
    self.TL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE_TL'] == 1.0]
    self.INTERMODAL_Cleaned_Data = self.cleaned_df[self.cleaned_df['ACTUAL MODE_INTERMODAL'] == 1.0]
    return self.cleaned_df, self.LTL_Cleaned_Data, self.TL_Cleaned_Data, self.INTERMODAL_Cleaned_Data
