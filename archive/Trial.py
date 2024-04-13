import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Example surprisal values for each word in a sentence
words = ["This", "is", "a", "simple", "example"]
surprisal_values = [0.2, 0.5, 0.1, 0.8, 0.4]  # Example values

# Normalize surprisal values for color mapping
normalized_vals = np.interp(surprisal_values, (min(surprisal_values), max(surprisal_values)), (0, 1))
colors = [cm.viridis(val) for val in normalized_vals]  # Use a colormap of your choice

# Convert RGBA colors to HEX for HTML
colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

# Generate HTML content with styled spans for each word
html_content = "".join([f'<span style="background-color: {color}; padding: 5px; margin: 2px; border-radius: 5px;">{word}</span>' 
                        for word, color in zip(words, colors_hex)])

# Display the custom heatmap in Streamlit
st.markdown(html_content, unsafe_allow_html=True)
