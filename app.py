# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:01:43 2025

@author: jordi
"""

import streamlit as st
import matplotlib.pyplot as plt

st.header("Hello, World!")
st.write("This is a basic Streamlit app with a sample matplotlib figure.")


fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16], marker='o')
ax.set_title("Sample Plot")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")


st.pyplot(fig)