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
    
def Callpredict():
    tkmb.showinfo(title = "Status", message = "Predict Complete")

     
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
        with open(resource_path("zipcode/mapping.json"),'r', encoding='UTF-8') as f:
            mapping = json.load(f)
        MT_train_data = pd.read_csv(MT_data_path.get())
        DAT_train_data = pd.read_excel(DAT_data_path.get())
        clean_data = dc.Data_cleaning(MT_train_data, DAT_train_data, zip,mapping)
        global train_cleaned_df, train_ltl_cleaned_data, train_tl_cleaned_data, train_intermodal_cleaned_data 
        train_cleaned_df, train_ltl_cleaned_data, train_tl_cleaned_data, train_intermodal_cleaned_data  = clean_data.final_data()
        CallInputSucceed()
    
    def new_model_training():
        print(train_cleaned_df)
        
        
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
    
    
    #global manual_input_df
    #global manual_DAT_df
    #global manual_predict_df
    #global DAT_info_dic_copy
    #global transporation_info_dic_copy
    #global
    #DAT_info_dic_copy = copy.deepcopy(DAT_info_dic)
    #transporation_info_dic_copy = copy.deepcopy(transporation_info_dic)
    
    #transporation_info_dic_copy.update(DAT_info_dic_copy)
    #manual_input_df = pd.DataFrame.from_dict([transporation_info_dic_copy])
    #manual_DAT_df = pd.DataFrame.from_dict([DAT_info_dic])
    #manual_predict_df = pd.DataFrame.from_dict([transporation_info_dic])

    #if manual_input_df.shape[1] == 0 :
    #    zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
    #    with open(resource_path("zipcode/mapping.json"),'r', encoding='UTF-8') as f:
    #        mapping = json.load(f)
    #    predict_data = pd.read_csv(Upcoming_shipment_path.get())
    #    dat_data = pd.read_excel(Upcoming_Data_Data_path.get())
    #    clean_data = dc.Data_cleaning(predict_data, dat_data, zip,mapping)
    #else:
    #    zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
    #    with open(resource_path("zipcode/mapping.json"),'r', encoding='UTF-8') as f:
    #        mapping = json.load(f)
    #    predict_data = manual_predict_df
    #    dat_data = manual_DAT_df
    #    clean_data = dc.Data_cleaning(predict_data, dat_data, zip,mapping)
    
    zip = pd.read_csv(resource_path('zipcode/zipcode.csv'))
    with open(resource_path("zipcode/mapping.json"),'r', encoding='UTF-8') as f:
        mapping = json.load(f)
    global predict_data
    predict_data = pd.read_csv(Upcoming_shipment_path.get())
    dat_data = pd.read_excel(Upcoming_Data_Data_path.get())
    clean_data = dc.Data_cleaning(predict_data, dat_data, zip,mapping)
    
    global cleaned_df, ltl_cleaned_data, tl_cleaned_data, intermodal_cleaned_data 
    cleaned_df, ltl_cleaned_data, tl_cleaned_data, intermodal_cleaned_data = clean_data.final_data()
        

global X_columns
X_columns = ['DISTANCE','CASES','WEIGHT','VOLUME','SPOT_AVG_LINEHAUL_RATE','SPOT_LOW_LINEHAUL_RATE',
             'SPOT_HIGH_LINEHAUL_RATE','SPOT_FUEL_SURCHARGE','SPOT_TIME_FRAME','SPOT_YOUR_OWN_AVG_LINEHAUL_RATE',
             'CONTRACT_AVG_LINEHAUL_RATE','CONTRACT_LOW_LINEHAUL_RATE','CONTRACT_HIGH_LINEHAUL_RATE','CONTRACT_FUEL_SURCHARGE',
             'CONTRACT_AVG_ACCESSORIAL_EXCLUDES_FUEL','CONTRACT_TIME_FRAME','CONTRACT_YOUR_OWN_AVG_LINEHAUL_RATE','ORIGIN_LAT',
             'ORIGIN_LNG','DEST_LAT','DEST_LNG','CUSTOMER_-8262326848732291361','CUSTOMER_-7599049178258596018','CUSTOMER_-7260197904156018921',
             'CUSTOMER_-6009976806944763491','CUSTOMER_-5306179411193977156','CUSTOMER_-4943192759040850492','CUSTOMER_-3997495352595819392','CUSTOMER_-2573927653953271278',
             'CUSTOMER_-697198312127870698','CUSTOMER_0','CUSTOMER_562003601178532658','CUSTOMER_687381462817196742','CUSTOMER_978799939867630084','CUSTOMER_1065625063082354427',
             'CUSTOMER_2941159810065373912','CUSTOMER_5194020264032378374','CUSTOMER_5233586626808759321','CUSTOMER_5263627990365651396','CUSTOMER_6060507223344358401','CUSTOMER_6117196247426314632',
             'CUSTOMER_7485008617214758133','CUSTOMER_8163086784792219814','CUSTOMER_9207189799533540268','ACTUAL CARRIER_-9223002062524768283','ACTUAL CARRIER_-9220921004385285101',
             'ACTUAL CARRIER_-9182059244154842393','ACTUAL CARRIER_-9144606003389011363','ACTUAL CARRIER_-9115044820148023944','ACTUAL CARRIER_-9082679522214585929','ACTUAL CARRIER_-8994301836397468791',
             'ACTUAL CARRIER_-8991654280515406014','ACTUAL CARRIER_-8923000960803081197','ACTUAL CARRIER_-8883738172624837438','ACTUAL CARRIER_-8823412207940582452','ACTUAL CARRIER_-8792161052838654822',
             'ACTUAL CARRIER_-8750234425492318905','ACTUAL CARRIER_-8661959186970149954','ACTUAL CARRIER_-8625125415128480678','ACTUAL CARRIER_-8606708363753010016','ACTUAL CARRIER_-8548829809724027018',
             'ACTUAL CARRIER_-8487699365480563185','ACTUAL CARRIER_-8465075887843914600','ACTUAL CARRIER_-8405669362813997487','ACTUAL CARRIER_-8366995904870702193','ACTUAL CARRIER_-8358008263826631345',
             'ACTUAL CARRIER_-8294798573130660524','ACTUAL CARRIER_-8214375365429291512','ACTUAL CARRIER_-8207305449027225204','ACTUAL CARRIER_-8199173536979890324','ACTUAL CARRIER_-8193145700283022470',
             'ACTUAL CARRIER_-8153834403937413477','ACTUAL CARRIER_-8086320830110626685','ACTUAL CARRIER_-8001391004667089964','ACTUAL CARRIER_-7906192283756657891','ACTUAL CARRIER_-7859431051807500712',
             'ACTUAL CARRIER_-7852920357241808015','ACTUAL CARRIER_-7829946791798136108','ACTUAL CARRIER_-7697093431112346373','ACTUAL CARRIER_-7557682413698847609','ACTUAL CARRIER_-7476529189921751978',
             'ACTUAL CARRIER_-7357215173914051676','ACTUAL CARRIER_-7310588159621340103','ACTUAL CARRIER_-7178866327325467816','ACTUAL CARRIER_-7139055810772211147','ACTUAL CARRIER_-7134260760414246217',
             'ACTUAL CARRIER_-7062225448916259956','ACTUAL CARRIER_-7049269704139940601','ACTUAL CARRIER_-7017473451601760252','ACTUAL CARRIER_-6924888709936129701','ACTUAL CARRIER_-6815778282047798768',
             'ACTUAL CARRIER_-6782475734443134969','ACTUAL CARRIER_-6715742545853213510','ACTUAL CARRIER_-6623505222913531633','ACTUAL CARRIER_-6596827657876095331','ACTUAL CARRIER_-6575377191243595714',
             'ACTUAL CARRIER_-6454980538666072983','ACTUAL CARRIER_-6376845136749238634','ACTUAL CARRIER_-6355811523635777232','ACTUAL CARRIER_-6229428136240565732','ACTUAL CARRIER_-6021696110899143417',
             'ACTUAL CARRIER_-5949221455586641234','ACTUAL CARRIER_-5948403548037400758','ACTUAL CARRIER_-5884425188031380224','ACTUAL CARRIER_-5814052866522769857','ACTUAL CARRIER_-5770196567747361617',
             'ACTUAL CARRIER_-5715359703023741355','ACTUAL CARRIER_-5706315741021498722','ACTUAL CARRIER_-5672083717837453743','ACTUAL CARRIER_-5662115410790083477','ACTUAL CARRIER_-5572929728134247035',
             'ACTUAL CARRIER_-5568784457635823767','ACTUAL CARRIER_-5535081847814506567','ACTUAL CARRIER_-5529014407538792948','ACTUAL CARRIER_-5385055515261426755','ACTUAL CARRIER_-5314388145131251555',
             'ACTUAL CARRIER_-5235846057928989837','ACTUAL CARRIER_-5216470038082366552','ACTUAL CARRIER_-5185180896017321663','ACTUAL CARRIER_-5069755705203772832','ACTUAL CARRIER_-5055256217613623745',
             'ACTUAL CARRIER_-5031311640667930484','ACTUAL CARRIER_-4967024170598590363','ACTUAL CARRIER_-4832470166077192707','ACTUAL CARRIER_-4731843556383331676','ACTUAL CARRIER_-4695947271231758158',
             'ACTUAL CARRIER_-4645217886316007068','ACTUAL CARRIER_-4580867503849172302','ACTUAL CARRIER_-4576364265700260210','ACTUAL CARRIER_-4554843221218222556','ACTUAL CARRIER_-4513929889249989103',
             'ACTUAL CARRIER_-4511766386077662018','ACTUAL CARRIER_-4435326302655771268','ACTUAL CARRIER_-4420237939287173963','ACTUAL CARRIER_-4351262451097218734','ACTUAL CARRIER_-4309828226026393502',
             'ACTUAL CARRIER_-4304257741259358600','ACTUAL CARRIER_-4290289884386242506','ACTUAL CARRIER_-4171437437704048688','ACTUAL CARRIER_-4157929999197960535','ACTUAL CARRIER_-4074688728228927087','ACTUAL CARRIER_-4024664612135870951',
             'ACTUAL CARRIER_-3986567577613572219','ACTUAL CARRIER_-3963065821025785885','ACTUAL CARRIER_-3910479296377212238','ACTUAL CARRIER_-3778444070324274876','ACTUAL CARRIER_-3692705546297942756',
             'ACTUAL CARRIER_-3651706814019958520','ACTUAL CARRIER_-3591159172683823316','ACTUAL CARRIER_-3470309477999285556','ACTUAL CARRIER_-3457304065625965868','ACTUAL CARRIER_-3426409447812244863',
             'ACTUAL CARRIER_-3370528682177446253','ACTUAL CARRIER_-3316652578453904037','ACTUAL CARRIER_-3046788819412085489','ACTUAL CARRIER_-3035309043136360770','ACTUAL CARRIER_-3002395896300520736',
             'ACTUAL CARRIER_-2926996427684579427','ACTUAL CARRIER_-2901330945142013893','ACTUAL CARRIER_-2824392672827311722','ACTUAL CARRIER_-2632734622293769822','ACTUAL CARRIER_-2589801912543550488',
             'ACTUAL CARRIER_-2410029896142308861','ACTUAL CARRIER_-2388902566401410582','ACTUAL CARRIER_-2357820282496965405','ACTUAL CARRIER_-2258267211932045100','ACTUAL CARRIER_-2201080808980306543',
             'ACTUAL CARRIER_-2059451553786511244','ACTUAL CARRIER_-2013590210252458250','ACTUAL CARRIER_-1997929190172432397','ACTUAL CARRIER_-1994806715152009354','ACTUAL CARRIER_-1838057456427687357',
             'ACTUAL CARRIER_-1510112450961188396','ACTUAL CARRIER_-1355450955538740011','ACTUAL CARRIER_-1351425257790973292','ACTUAL CARRIER_-1298288681428199332','ACTUAL CARRIER_-1279486390322896842',
             'ACTUAL CARRIER_-1270200713903953107','ACTUAL CARRIER_-1178957832028810530','ACTUAL CARRIER_-1163576488348344214','ACTUAL CARRIER_-1086023780773586659','ACTUAL CARRIER_-1038784421417444895',
             'ACTUAL CARRIER_-975950624203799934','ACTUAL CARRIER_-909413422061181435','ACTUAL CARRIER_-798548042746209789','ACTUAL CARRIER_-550370160474185608','ACTUAL CARRIER_-522178489128368608','ACTUAL CARRIER_-494177721110594076',
             'ACTUAL CARRIER_-451425264924386066','ACTUAL CARRIER_-429674910911029843','ACTUAL CARRIER_-412674278127286866','ACTUAL CARRIER_-334105174026172035','ACTUAL CARRIER_-325634004728434274',
             'ACTUAL CARRIER_-233393129501276271','ACTUAL CARRIER_-188413310767595511','ACTUAL CARRIER_-159768898937569799','ACTUAL CARRIER_-144892085626435056','ACTUAL CARRIER_88119697740707300',
             'ACTUAL CARRIER_254483869161271673','ACTUAL CARRIER_313216971269301131','ACTUAL CARRIER_346933322086152262','ACTUAL CARRIER_388118943958106444','ACTUAL CARRIER_396786793112852276',
             'ACTUAL CARRIER_414736186532893312','ACTUAL CARRIER_419021896512096009','ACTUAL CARRIER_439742795446241203','ACTUAL CARRIER_448985426171054041','ACTUAL CARRIER_506116562772293284','ACTUAL CARRIER_645332426270404426',
             'ACTUAL CARRIER_651851005724257248','ACTUAL CARRIER_691553408915430911','ACTUAL CARRIER_694077031691304157','ACTUAL CARRIER_883253809201880075','ACTUAL CARRIER_979174748842602474','ACTUAL CARRIER_1009323822653269036',
             'ACTUAL CARRIER_1011597216220347888','ACTUAL CARRIER_1156222094854316348','ACTUAL CARRIER_1157916975620409757','ACTUAL CARRIER_1183217010820749086','ACTUAL CARRIER_1199331226302960096','ACTUAL CARRIER_1262657598820695207',
             'ACTUAL CARRIER_1272003557912075676','ACTUAL CARRIER_1327590664492842710','ACTUAL CARRIER_1429552171201558873','ACTUAL CARRIER_1438713307981887429','ACTUAL CARRIER_1547799302677420393','ACTUAL CARRIER_1572324641329892797',
             'ACTUAL CARRIER_1628111090481984826','ACTUAL CARRIER_1650620239407712743','ACTUAL CARRIER_1707606646843784732','ACTUAL CARRIER_1763372711640511368','ACTUAL CARRIER_1812371860599138325','ACTUAL CARRIER_1840836194653418378',
             'ACTUAL CARRIER_1896630384904438820','ACTUAL CARRIER_1992656996180753256','ACTUAL CARRIER_2137694019173416411','ACTUAL CARRIER_2235266994283596514','ACTUAL CARRIER_2239140168571598162','ACTUAL CARRIER_2282664550396472528','ACTUAL CARRIER_2328615639935842750','ACTUAL CARRIER_2336787424202757150','ACTUAL CARRIER_2394214085265090099','ACTUAL CARRIER_2491319798290551867','ACTUAL CARRIER_2515764005699930051','ACTUAL CARRIER_2516457524366802597','ACTUAL CARRIER_2619599452973704908','ACTUAL CARRIER_2636136953422826340','ACTUAL CARRIER_2639119519292987712','ACTUAL CARRIER_2687986817561434611','ACTUAL CARRIER_2689514744828594441','ACTUAL CARRIER_2752091217497244299','ACTUAL CARRIER_2799860752622732791','ACTUAL CARRIER_2817688474839256833','ACTUAL CARRIER_2839028930768726471','ACTUAL CARRIER_2848297393482927612','ACTUAL CARRIER_2973947785370081612','ACTUAL CARRIER_2996298379816604571','ACTUAL CARRIER_3042996887439097013','ACTUAL CARRIER_3093956354811887686','ACTUAL CARRIER_3161313781580036381','ACTUAL CARRIER_3338967183097163509','ACTUAL CARRIER_3369437310226587637','ACTUAL CARRIER_3448241910996348931','ACTUAL CARRIER_3485279221829682367','ACTUAL CARRIER_3491371160382064094','ACTUAL CARRIER_3526018404470248184','ACTUAL CARRIER_3529155843034776654','ACTUAL CARRIER_3554730261913487435','ACTUAL CARRIER_3606368351755010369','ACTUAL CARRIER_3918214702143936120','ACTUAL CARRIER_4107741553661004589','ACTUAL CARRIER_4120651981297905903','ACTUAL CARRIER_4222906504014004013','ACTUAL CARRIER_4316320795423936661','ACTUAL CARRIER_4337839914667252195','ACTUAL CARRIER_4345763104461969032','ACTUAL CARRIER_4364018878568062699','ACTUAL CARRIER_4420768141604581544','ACTUAL CARRIER_4430323104218801879','ACTUAL CARRIER_4532869605455601760','ACTUAL CARRIER_4616343029047361851','ACTUAL CARRIER_4653163374797542751','ACTUAL CARRIER_4653166576236138991','ACTUAL CARRIER_4775771877738855709','ACTUAL CARRIER_4807135686403412271','ACTUAL CARRIER_4818649119531991587','ACTUAL CARRIER_5025274402081265438','ACTUAL CARRIER_5025661985520042630','ACTUAL CARRIER_5073512493142632412','ACTUAL CARRIER_5074324453392728385','ACTUAL CARRIER_5119932344854677299','ACTUAL CARRIER_5212287128358006298','ACTUAL CARRIER_5219045461062122483','ACTUAL CARRIER_5266463211128490078','ACTUAL CARRIER_5286489544027453557','ACTUAL CARRIER_5349716118574854984','ACTUAL CARRIER_5352943694923179223','ACTUAL CARRIER_5365577002007598641','ACTUAL CARRIER_5449107145564959861','ACTUAL CARRIER_5472037180666771884','ACTUAL CARRIER_5566950593659878938','ACTUAL CARRIER_5576213910022895645','ACTUAL CARRIER_5613578782018461725','ACTUAL CARRIER_5631275026044946337','ACTUAL CARRIER_5672223945644987703','ACTUAL CARRIER_5826500001536478122','ACTUAL CARRIER_5860777404852369898','ACTUAL CARRIER_5903916240297969530','ACTUAL CARRIER_5904312124681493838','ACTUAL CARRIER_5923938882419567855','ACTUAL CARRIER_5932990539557884032','ACTUAL CARRIER_5959605862285908499','ACTUAL CARRIER_5991979590076326807','ACTUAL CARRIER_6045824833479685818','ACTUAL CARRIER_6106694748475860736','ACTUAL CARRIER_6108590758796752386','ACTUAL CARRIER_6201733358358936082','ACTUAL CARRIER_6273882146989164462','ACTUAL CARRIER_6290488854980792933','ACTUAL CARRIER_6385395136405395239','ACTUAL CARRIER_6394411864793688819','ACTUAL CARRIER_6446456201533463285','ACTUAL CARRIER_6518408014183118603','ACTUAL CARRIER_6578851111272543020','ACTUAL CARRIER_6589258965743719203','ACTUAL CARRIER_6602110040726556539','ACTUAL CARRIER_6731120826423266688','ACTUAL CARRIER_6752178823324025035','ACTUAL CARRIER_6799439698415359101','ACTUAL CARRIER_6826230257980741520','ACTUAL CARRIER_6886273989362063879','ACTUAL CARRIER_7035860709627471079','ACTUAL CARRIER_7222235055918937966','ACTUAL CARRIER_7348035433034685921','ACTUAL CARRIER_7374968629738504936','ACTUAL CARRIER_7425008351091776428','ACTUAL CARRIER_7455953921530758348','ACTUAL CARRIER_7492466490633419177','ACTUAL CARRIER_7502661219536617405','ACTUAL CARRIER_7543077221938801331','ACTUAL CARRIER_7555102410970701683','ACTUAL CARRIER_7572218506850480884','ACTUAL CARRIER_7662770697122124202','ACTUAL CARRIER_7718290150509819265','ACTUAL CARRIER_7718458745802428150','ACTUAL CARRIER_7750907041358647437','ACTUAL CARRIER_7774624532487150331','ACTUAL CARRIER_7814146810873680723','ACTUAL CARRIER_7829521301973152978','ACTUAL CARRIER_7845843523283851214','ACTUAL CARRIER_7906850182737607618','ACTUAL CARRIER_7966547263551291757','ACTUAL CARRIER_8149589042190358712','ACTUAL CARRIER_8180256758475220345','ACTUAL CARRIER_8212689186170666376','ACTUAL CARRIER_8345153872821796369','ACTUAL CARRIER_8540983874916498163','ACTUAL CARRIER_8596093010462176909','ACTUAL CARRIER_8689987903716344936','ACTUAL CARRIER_8710650212460098106','ACTUAL CARRIER_8736485323820759024','ACTUAL CARRIER_8879159275886726765','ACTUAL CARRIER_8928287337202294697','ACTUAL CARRIER_9026142584875498412','ACTUAL CARRIER_9030094597835863827','ACTUAL CARRIER_9084881302716223838','ACTUAL CARRIER_9197451511315311647','ACTUAL MODE_INTERMODAL','ACTUAL MODE_LTL','ACTUAL MODE_TL','ACTUAL EQUIP_01MM','ACTUAL EQUIP_12RF','ACTUAL EQUIP_16STRAIGHT TRUCK','ACTUAL EQUIP_20DV','ACTUAL EQUIP_20FT','ACTUAL EQUIP_20RF','ACTUAL EQUIP_40DV','ACTUAL EQUIP_40RF','ACTUAL EQUIP_45FB','ACTUAL EQUIP_48DV','ACTUAL EQUIP_48FB','ACTUAL EQUIP_53BK','ACTUAL EQUIP_53DV','ACTUAL EQUIP_53DVS','ACTUAL EQUIP_53FB','ACTUAL EQUIP_53HH','ACTUAL EQUIP_53IM','ACTUAL EQUIP_53IM_SONY','ACTUAL EQUIP_53RF','ACTUAL EQUIP_53XX','ACTUAL EQUIP_PICKUP_TRUCK','ACTUAL EQUIP_TEST_56PP','PU_APPT','DL_APPT']
def ltl_model():
    data_cleaning()
    df = ltl_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.detach().numpy())
    
    outputdf = copy.deepcopy(predict_data)
    outputdf['ACTUAL MODE'] = outputdf['ACTUAL MODE'].str.split('.').str[-1]
    outputdf = outputdf.loc[outputdf['ACTUAL MODE']=='LTL']
    df['Predicted Linehaul Cost'] = y_hat_df
    savename = filedialog.asksaveasfilename()
    savename = savename.split('.')[0]+'.csv'
    df.to_csv(savename,index = False)
    Callpredict()

    
    
def tl_model():
    data_cleaning()
    df = tl_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.detach().numpy())
    outputdf = copy.deepcopy(predict_data)
    outputdf['ACTUAL MODE'] = outputdf['ACTUAL MODE'].str.split('.').str[-1]
    outputdf = outputdf.loc[outputdf['ACTUAL MODE']=='TL']
    df['Predicted Linehaul Cost'] = y_hat_df
    savename = filedialog.asksaveasfilename()
    savename = savename.split('.')[0]+'.csv'
    df.to_csv(savename,index = False)
    Callpredict()
    
    
def intermodal_model():
    data_cleaning()
    df = intermodal_cleaned_data
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.detach().numpy())
    outputdf = copy.deepcopy(predict_data)
    outputdf['ACTUAL MODE'] = outputdf['ACTUAL MODE'].str.split('.').str[-1]
    outputdf = outputdf.loc[outputdf['ACTUAL MODE']=='INTERMODAL']
    df['Predicted Linehaul Cost'] = y_hat_df
    savename = filedialog.asksaveasfilename()
    savename = savename.split('.')[0]+'.csv'
    df.to_csv(savename,index = False)
    Callpredict()
    
def all_model():
    data_cleaning()
    df = cleaned_df
    global model_file
    model_file = askopenfilename()
    model = torch.load(model_file)
    X = torch.tensor(df[X_columns].values, dtype = torch.float, requires_grad = True)
    y_hat = model(X)
    y_hat_df = pd.DataFrame(y_hat.detach().numpy())
    #outputdf = copy.deepcopy(predict_data)
    df['Predicted Linehaul Cost'] = y_hat_df
    savename = filedialog.asksaveasfilename()
    savename = savename.split('.')[0]+'.csv'
    df.to_csv(savename,index = False)
    Callpredict()  
    
    
   
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

tk.Entry(root, textvariable = Upcoming_shipment_path,width= 45).place(x = 250, y = 50)
tk.Button(root, text='Browse',bg = '#FFCC01',command = import_Upcoming_shipment).place(x = 700, y = 50)
#tk.Button(root, text='Manual Input',command = manual_input_trans,bg = '#FFCC01').place(x = 650, y = 50)
tk.Entry(root, textvariable=Upcoming_Data_Data_path,width= 45).place(x = 250, y = 80)
tk.Button(root, text='Browse',bg = '#FFCC01',command = import_DAT_Data).place(x = 700, y = 80)
#tk.Button(root, text='Manual Input',bg = '#FFCC01',command = manual_input_DAT).place(x = 650, y = 80)
tk.Button(root, text='LTL Model',command = ltl_model).place(x = 50 ,y = 150)
tk.Button(root, text='TL Model',command = tl_model).place(x = 200, y = 150)
tk.Button(root, text='Railway Model',command = intermodal_model).place(x = 350, y = 150)
tk.Button(root, text='Predict All',command = all_model).place(x = 530, y = 150)
tk.Button(root, text='Update Training data', command = update_train_data).place(x = 650, y = 150)

root.mainloop()

