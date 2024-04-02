# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 01:57:46 2024

@author: Xabi
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

scaler = StandardScaler()

    st.title('Information about CKD')
