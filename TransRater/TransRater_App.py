#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:46:31 2022

@author: petertian
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from tkinter import filedialog
import tkinter.messagebox as tkmb
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sys,traceback
import PIL.Image
import PIL.ImageTk
import os
import data_clean as dc
import CNN
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go 
from category_encoders import LeaveOneOutEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import ADF
from torchmetrics import SymmetricMeanAbsolutePercentageError
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
import copy


def CallInputSucceed():
    tkmb.showinfo(title = "Status", message = "Input Complete")
    
def CallTrainSucceed():
    tkmb.showinfo(title = "Status", message = "Training Complete")

     
def import_Upcoming_shipment():
    global Upcoming_shipment_path
    global Ucoming_shipment_file
    Upcoming_shipment_file = askopenfilename()
    Upcoming_shipment_path.set(Upcoming_shipment_file)
    
def import_DAT_Data():
    global Upcoming_Data_Data_path
    global Upcoming_Data_Data_file
    Upcoming_Data_Data_file = askopenfilename()
    Upcoming_Data_Data_path.set(Upcoming_Data_Data_file)

    
    
global transporation_info_dic
transporation_info_dic = {}


def manual_input_trans():
    new = tk.Toplevel(root)
    new.geometry("600x500")
    new.title("Input Upcoming Transporation Information")
    logo = tk.PhotoImage(file = resource_path('image/TransRater_Logo.png'))
    tk.Label(new, image=logo,bg ='#ffcc00' ).place(x=330,y=400)
    tk.Label(new, text = "Input Shipment Information Manually",bg ='#ffcc00', font='Helvetica 12 bold').place(x = 200, y = 10)
    new.config(background = "#ffcc00")
    
    transporation_info_list = ['DISTANCE','CASES', 'WEIGHT', 'VOLUME','ORIGIN ZIP', 'DEST ZIP','ACTUAL MODE', 'ACTUAL EQUIP', 'PUICK APPT', 'DELIVARY APPT']
    
    y = 50
    x = 80
    for info in transporation_info_list:
        tk.Label(new, text=info,bg ='#FFCC01', font = 'delivery 12 bold').place(x=x, y=y)
        y = y+30
    
    distance_input = tk.Entry(new,width= 10)
    distance_input.place(x = 400, y = 50)   
    case_input = tk.Entry(new,width= 10)
    case_input.place(x = 400, y = 80)   
    volume_input = tk.Entry(new,width= 10)
    volume_input.place(x = 400, y = 110)   
    weight_input = tk.Entry(new,width= 10)
    weight_input.place(x = 400, y = 140) 
    Original_Zip_input = tk.Entry(new,width= 10)
    Original_Zip_input.place(x = 400, y = 170) 
    Dest_Zip_input = tk.Entry(new,width= 10)
    Dest_Zip_input.place(x = 400, y = 200)
    act_model_input = tk.Entry(new,width= 10)
    act_model_input.place(x = 400, y = 230)
    act_equip_input = tk.Entry(new,width= 10)
    act_equip_input.place(x = 400, y = 260)
    PU_APPT_input = tk.Entry(new,width= 10)
    PU_APPT_input.place(x = 400, y = 290)
    DL_APPT_input = tk.Entry(new,width= 10)
    DL_APPT_input.place(x = 400, y = 320)

    def get_distance():
        global distance
        distance = distance_input.get()
        return distance
        
    def get_volume():
        global volume
        volume = volume_input.get()
        return volume
        
    def get_case():
        global case
        case = case_input.get()
        return case
            
    def get_weight():
        global weight
        weight = weight_input.get()
        return weight
        
    def get_Original_Zip():
        global Original_Zip
        Original_Zip = Original_Zip_input.get()
        return Original_Zip
        
    def get_Dest_Zip():
        global Dest_Zip
        Dest_Zip = Dest_Zip_input.get()   
        return Dest_Zip
    
    def get_act_model():
        global act_model
        act_model = act_model_input.get()   
        return act_model
    
    def get_act_equip():
        global act_equip
        act_equip = act_equip_input.get()
        return act_equip
    
    def get_PU_APPT():
        global PU_APPT
        PU_APPT = PU_APPT_input.get()
        return PU_APPT
    
    def get_DL_APPT():
        global DL_APPT
        DL_APPT = DL_APPT_input.get()
        return  DL_APPT
    
   
        
    def record_text():
        get_distance()
        get_volume()
        get_Original_Zip()
        get_Dest_Zip()
        get_case()
        get_weight()
        get_act_model()
        get_act_equip()
        get_PU_APPT()
        get_DL_APPT()
        
        
        transporation_info_dic['VOLUME'] = volume
        transporation_info_dic['distance'] = distance
        transporation_info_dic['WEIGHT'] = weight
        transporation_info_dic['CASES'] = case
        transporation_info_dic['ORIGIN ZIP'] = Original_Zip
        transporation_info_dic['DEST ZIP'] = Dest_Zip
        transporation_info_dic['ACTUAL MODE'] = act_model
        transporation_info_dic['ACTUAL EQUIP'] = act_equip
        transporation_info_dic['PU_APPT'] = PU_APPT
        transporation_info_dic['DL_APPT'] = DL_APPT
        
        print(transporation_info_dic)
        
        
        
        
        CallInputSucceed()
            
    tk.Button(new, text='Input',command = record_text,bg = '#FFCC01').place(x = 250, y = 400)


    new.mainloop()
    
    
global DAT_info_dic
DAT_info_dic = {}
def manual_input_DAT():
    new_DAT = tk.Toplevel(root)
    new_DAT.geometry("600x600")
    new_DAT.title("Input Upcoming DAT Information")
    logo = tk.PhotoImage(file = resource_path('image/TransRater_Logo.png'))
    tk.Label(new_DAT, image=logo,bg ='#ffcc00' ).place(x=330,y=500)
    tk.Label(new_DAT, text = "Input Shipment Information Manually",bg ='#ffcc00', font='Helvetica 12 bold').place(x = 200, y = 10)
    new_DAT.config(background = "#ffcc00")
    
    DAT_info_list = ['SPOT AVG LINEHAUL RATE', 'SPOT LOW LINEHAUL RATE', 'SPOT HIGH LINEHAUL RATE', 'SPOT FUEL SURCHARGE', 'SPOT TIME FRAME',
                               'SPOT YOUR OWN AVG LINEHAUL RATE', 'CONTRACT AVG LINEHAUL RATE', 'CONTRACT LOW LINEHAUL RATE', 'CONTRACT HIGH LINEHAUL RATE', 
                               'CONTRACT FUEL SURCHARGE', 'CONTRACT AVG ACCESSORIAL EXCLUDES FUEL', 'CONTRACT TIME FRAME','CONTRACT YOUR OWN AVG LINEHAUL RATE']
    
    y = 50
    x = 80
    for info in DAT_info_list:
        tk.Label(new_DAT, text=info,bg ='#FFCC01', font = 'delivery 12 bold').place(x=x, y=y)
        y = y+30
    
    SPOT_AVG_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    SPOT_AVG_LINEHAUL_RATE_input.place(x = 400, y = 350-300)
    SPOT_LOW_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    SPOT_LOW_LINEHAUL_RATE_input.place(x = 400, y = 380-300)
    SPOT_HIGH_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    SPOT_HIGH_LINEHAUL_RATE_input.place(x = 400, y = 410-300)
    SPOT_FUEL_SURCHARGE_input = tk.Entry(new_DAT,width= 10)
    SPOT_FUEL_SURCHARGE_input.place(x = 400, y = 440-300)
    SPOT_TIME_FRAME_input = tk.Entry(new_DAT,width= 10)
    SPOT_TIME_FRAME_input.place(x = 400, y = 470-300)
    SPOT_YOUR_OWN_AVG_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    SPOT_YOUR_OWN_AVG_LINEHAUL_RATE_input.place(x = 400, y = 500-300)
    
    CONTRACT_AVG_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_AVG_LINEHAUL_RATE_input.place(x = 400, y = 530-300)
    CONTRACT_LOW_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_LOW_LINEHAUL_RATE_input.place(x = 400, y = 560-300)
    CONTRACT_HIGH_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_HIGH_LINEHAUL_RATE_input.place(x = 400, y = 590-300)
    CONTRACT_FUEL_SURCHARGE_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_FUEL_SURCHARGE_input.place(x = 400, y = 620-300)
    CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL_input.place(x = 400, y = 650-300)
    CONTRACT_TIME_FRAME_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_TIME_FRAME_input.place(x = 400, y = 680-300)
    CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE_input = tk.Entry(new_DAT,width= 10)
    CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE_input.place(x = 400, y = 710-300)

    
    def get_SPOT_AVG_LINEHAUL_RATE():
        global SPOT_AVG_LINEHAUL_RATE
        SPOT_AVG_LINEHAUL_RATE = SPOT_AVG_LINEHAUL_RATE_input.get()
        return SPOT_AVG_LINEHAUL_RATE
    
    def get_SPOT_LOW_LINEHAUL_RATE():
        global SPOT_LOW_LINEHAUL_RATE
        SPOT_LOW_LINEHAUL_RATE = SPOT_LOW_LINEHAUL_RATE_input.get()
        return SPOT_AVG_LINEHAUL_RATE
        
    def get_SPOT_HIGH_LINEHAUL_RATE():
        global SPOT_HIGH_LINEHAUL_RATE
        SPOT_HIGH_LINEHAUL_RATE = SPOT_HIGH_LINEHAUL_RATE_input.get()
        return SPOT_HIGH_LINEHAUL_RATE
    
    def get_SPOT_FUEL_SURCHARGE():
        global SPOT_FUEL_SURCHARGE
        SPOT_FUEL_SURCHARGE = SPOT_FUEL_SURCHARGE_input.get()
        return SPOT_FUEL_SURCHARGE
    
    def get_SPOT_TIME_FRAME():
        global SPOT_TIME_FRAME
        SPOT_TIME_FRAME = SPOT_TIME_FRAME_input.get()
        return SPOT_TIME_FRAME
    
    def get_SPOT_YOUR_OWN_AVG_LINEHAUL_RATE():
        global SPOT_YOUR_OWN_AVG_LINEHAUL_RATE
        SPOT_YOUR_OWN_AVG_LINEHAUL_RATE = SPOT_YOUR_OWN_AVG_LINEHAUL_RATE_input.get()
        return SPOT_YOUR_OWN_AVG_LINEHAUL_RATE
    
    def get_CONTRACT_AVG_LINEHAUL_RATE():
        global CONTRACT_AVG_LINEHAUL_RATE
        CONTRACT_AVG_LINEHAUL_RATE = CONTRACT_AVG_LINEHAUL_RATE_input.get()
        return CONTRACT_AVG_LINEHAUL_RATE
    
    def get_CONTRACT_LOW_LINEHAUL_RATE():
        global CONTRACT_LOW_LINEHAUL_RATE
        CONTRACT_LOW_LINEHAUL_RATE = CONTRACT_LOW_LINEHAUL_RATE_input.get()
        return CONTRACT_AVG_LINEHAUL_RATE
        
    def get_CONTRACT_HIGH_LINEHAUL_RATE():
        global CONTRACT_HIGH_LINEHAUL_RATE
        CONTRACT_HIGH_LINEHAUL_RATE = CONTRACT_HIGH_LINEHAUL_RATE_input.get()
        return CONTRACT_HIGH_LINEHAUL_RATE
    
    def get_CONTRACT_FUEL_SURCHARGE():
        global CONTRACT_FUEL_SURCHARGE
        CONTRACT_FUEL_SURCHARGE = CONTRACT_FUEL_SURCHARGE_input.get()
        return CONTRACT_FUEL_SURCHARGE
    
    def get_CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL():
        global CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL
        CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL = CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL_input.get()
        return CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL
    
    def get_CONTRACT_TIME_FRAME():
        global CONTRACT_TIME_FRAME
        CONTRACT_TIME_FRAME = CONTRACT_TIME_FRAME_input.get()
        return CONTRACT_TIME_FRAME
    
    def get_CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE():
        global CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE
        CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE = CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE_input.get()
        return CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE
    

        
    def record_text():
        get_SPOT_AVG_LINEHAUL_RATE()
        get_SPOT_LOW_LINEHAUL_RATE()
        get_SPOT_HIGH_LINEHAUL_RATE()
        get_SPOT_FUEL_SURCHARGE()
        get_SPOT_TIME_FRAME()
        get_SPOT_YOUR_OWN_AVG_LINEHAUL_RATE()
        get_CONTRACT_AVG_LINEHAUL_RATE()
        get_CONTRACT_LOW_LINEHAUL_RATE()
        get_CONTRACT_HIGH_LINEHAUL_RATE()
        get_CONTRACT_FUEL_SURCHARGE()
        get_CONTRACT_TIME_FRAME()
        get_CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL()
        get_CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE()
        
        DAT_info_dic['SPOT_AVG_LINEHAUL_RATE'] = SPOT_AVG_LINEHAUL_RATE
        DAT_info_dic['SPOT_LOW_LINEHAUL_RATE'] = SPOT_LOW_LINEHAUL_RATE
        DAT_info_dic['SPOT_HIGH_LINEHAUL_RATE'] = SPOT_HIGH_LINEHAUL_RATE
        DAT_info_dic['SPOT_FUEL_SURCHARGE'] = SPOT_FUEL_SURCHARGE
        DAT_info_dic['SPOT_TIME_FRAME'] = SPOT_TIME_FRAME
        DAT_info_dic['SPOT_YOUR_OWN_AVG_LINEHAUL_RATE'] = SPOT_YOUR_OWN_AVG_LINEHAUL_RATE
        
        DAT_info_dic['CONTRACT_AVG_LINEHAUL_RATE'] = CONTRACT_AVG_LINEHAUL_RATE
        DAT_info_dic['CONTRACT_LOW_LINEHAUL_RATE'] = CONTRACT_LOW_LINEHAUL_RATE
        DAT_info_dic['CONTRACT_HIGH_LINEHAUL_RATE'] = CONTRACT_HIGH_LINEHAUL_RATE
        DAT_info_dic['CONTRACT_FUEL_SURCHARGE'] = CONTRACT_FUEL_SURCHARGE
        DAT_info_dic['CONTRACT_TIME_FRAME'] = CONTRACT_TIME_FRAME
        DAT_info_dic['CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL'] = CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL
        DAT_info_dic['CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE'] = CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE
        print(DAT_info_dic)
        
        
        
        
        CallInputSucceed()
            
    tk.Button(new_DAT, text='Input',command = record_text,bg = '#FFCC01').place(x = 250, y = 500)


    new_DAT.mainloop()
    

def update_train_data():
    update_train_data = tk.Toplevel(root)
    logo = tk.PhotoImage(file = resource_path('image/TransRater_Logo.png'))
    tk.Label(update_train_data, image=logo,bg ='#ffcc00' ).place(x=600,y=100)
    update_train_data.config(background = "#ffcc00")
    update_train_data.geometry("900x200")
    update_train_data.title('Update MT Dataset and DAT Datset')
    tk.Label(update_train_data, text = "Update following data to improve model",bg ='#ffcc00',font = 'Helvetica 12 bold').place(x = 300, y =10)
    tk.Label(update_train_data, text='MT DATA',bg ='#FFCC01', font = 'delivery 12 bold').place(x=80, y=50)
    tk.Label(update_train_data, text='DAT DATA',bg ='#FFCC01', font = 'delivery 12 bold').place(x=80, y=80)
    
    def import_train_MT_data():
        global MT_data_path
        global MT_data_file
        MT_data_file = askopenfilename()
        MT_data_path.set(MT_data_file)
    
    def import_train_DAT_data():
        global DAT_data_path
        global DAT_data_file
        DAT_data_file = askopenfilename()
        DAT_data_path.set(DAT_data_file)
    
    global MT_data_path
    MT_data_path = tk.StringVar(root)
    tk.Entry(update_train_data, textvariable=MT_data_path,width= 45).place(x = 200, y = 50)
    tk.Button(update_train_data, text='Browse',bg = '#FFCC01',command = import_train_MT_data).place(x = 650, y = 50) 
    global DAT_data_path
    DAT_data_path = tk.StringVar()
    tk.Entry(update_train_data, textvariable=DAT_data_path,width= 45).place(x = 200, y = 80)
    tk.Button(update_train_data, text='Browse',command = import_train_DAT_data, bg = '#FFCC01').place(x = 650, y = 80)
    
    def train_data_cleaning():
        zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
        MT_train_data = pd.read_csv(MT_data_path.get())
        DAT_train_data = pd.read_csv(DAT_data_path.get())
        clean_data = dc.Data_cleaning(MT_train_data, DAT_train_data, zip)
        global train_cleaned_df, train_ltl_cleaned_data, train_tl_cleaned_data, train_intermodal_cleaned_data 
        train_cleaned_df, train_ltl_cleaned_data, train_tl_cleaned_data, train_intermodal_cleaned_data  = clean_data.final_data()
        CallInputSucceed()
    
    def new_model_training():
        print(train_cleaned_df)
        X_columns = list(train_cleaned_df.columns)
        X_columns.remove('LINEHAUL COSTS')
        print(X_columns)
        
        cnn_model = CNN.time_NN(device_type='cpu')
        cnn_model.clean_data(train_cleaned_df, X_columns, [])
        cnn_model.modeling(epoch = 10000)
        savename = filedialog.asksaveasfilename()
        savename = savename.split('.')[0]+'.pt'
        torch.save(cnn_model.my_nn, savename)
        CallTrainSucceed() 
    
        
    tk.Button(update_train_data, text='Update Training Data',bg = '#FFCC01',command = train_data_cleaning).place(x = 400, y = 130)
    tk.Button(update_train_data, text='Train a new model',bg = '#FFCC01',command = new_model_training).place(x = 200, y = 130)
    
    update_train_data.mainloop()
    

def data_cleaning():  
    
    
    global manual_input_df
    global manual_DAT_df
    global manual_predict_df
    global DAT_info_dic_copy
    global transporation_info_dic_copy
    #global
    DAT_info_dic_copy = copy.deepcopy(DAT_info_dic)
    transporation_info_dic_copy = copy.deepcopy(transporation_info_dic)
    
    transporation_info_dic_copy.update(DAT_info_dic_copy)
    manual_input_df = pd.DataFrame.from_dict([transporation_info_dic_copy])
    manual_DAT_df = pd.DataFrame.from_dict([DAT_info_dic])
    manual_predict_df = pd.DataFrame.from_dict([transporation_info_dic])

    if manual_input_df.shape[1] == 0 :
        zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
        predict_data = pd.read_csv(Upcoming_shipment_path.get())
        dat_data = pd.read_csv(Upcoming_Data_Data_path.get())
        clean_data = dc.Data_cleaning(predict_data, dat_data, zip)
    else:
        zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
        predict_data = manual_predict_df
        dat_data = manual_DAT_df
        clean_data = dc.Data_cleaning(predict_data, dat_data, zip)
    global cleaned_df, ltl_cleaned_data, tl_cleaned_data, intermodal_cleaned_data 
    cleaned_df, ltl_cleaned_data, tl_cleaned_data, intermodal_cleaned_data = clean_data.final_data()
        

def ltl_model():
    data_cleaning()
    df = ltl_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X_columns = list(df.columns)
    X_columns.remove('LINEHAUL COSTS')
    print(X_columns)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.numpy())
    print(y_hat_df)

    
    
def tl_model():
    data_cleaning()
    df = tl_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X_columns = list(df.columns)
    X_columns.remove('LINEHAUL COSTS')
    print(X_columns)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.numpy())
    
    
def intermodal_model():
    data_cleaning()
    df = intermodal_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X_columns = list(df.columns)
    X_columns.remove('LINEHAUL COSTS')
    print(X_columns)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.numpy())
    
def all_model():
    data_cleaning()
    df = cleaned_df
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X_columns = list(df.columns)
    X_columns.remove('LINEHAUL COSTS')
    print(X_columns)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.numpy())    
    
    
   
def resource_path(relative_path):
    
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(
        sys,
        '_MEIPASS',
        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

root = tk.Tk()
icon=tk.PhotoImage(file=resource_path('image/icon.png'))
root.iconphoto(False,icon)
logo = tk.PhotoImage(file = resource_path('image/TransRater_Logo.png'))
tk.Label(root, image=logo,bg ='#ffcc00' ).place(x=600,y=200)
root.title("TransRater")
root.config(background = "#ffcc00")
root.geometry("900x300")
tk.Label(root, text='Upcoming Transpotation',bg ='#FFCC01', font = 'delivery 12 bold').place(x=80, y=50)
tk.Label(root, text='Recent Shipping DAT',bg ='#FFCC01', font = 'delivery 12 bold').place(x=80, y=80)
Upcoming_shipment_path = tk.StringVar()
Upcoming_Data_Data_path = tk.StringVar()

tk.Entry(root, textvariable = Upcoming_shipment_path,width= 30).place(x = 250, y = 50)
tk.Button(root, text='Browse',bg = '#FFCC01',command = import_Upcoming_shipment).place(x = 550, y = 50)
tk.Button(root, text='Manual Input',command = manual_input_trans,bg = '#FFCC01').place(x = 650, y = 50)
tk.Entry(root, textvariable=Upcoming_Data_Data_path,width= 30).place(x = 250, y = 80)
tk.Button(root, text='Browse',bg = '#FFCC01',command = import_DAT_Data).place(x = 550, y = 80)
tk.Button(root, text='Manual Input',bg = '#FFCC01',command = manual_input_DAT).place(x = 650, y = 80)
tk.Button(root, text='LTL Model',command = ltl_model).place(x = 50 ,y = 150)
tk.Button(root, text='TL Model',command = tl_model).place(x = 200, y = 150)
tk.Button(root, text='Railway Model',command = intermodal_model).place(x = 350, y = 150)
tk.Button(root, text='Predict All',command = all_model).place(x = 530, y = 150)
tk.Button(root, text='Update Training data', command = update_train_data).place(x = 650, y = 150)

root.mainloop()

