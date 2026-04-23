import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment import CancerSimulation

st.set_page_config(page_title="Peacekeeper AI", layout="wide")

st.title("🎗️ Peacekeeper: AI Adaptive Therapy")
st.sidebar.header("Patient Profile")

# Let users play with the settings
growth = st.sidebar.slider("Avg Growth Rate", 1.0, 15.0, 8.5)
res_a = st.sidebar.slider("Initial Resistance A", 0.0, 10.0, 2.0)

if st.button("Run AI Treatment Simulation"):
    model = PPO.load("peacekeeper_final_azure")
    env = CancerSimulation({'avg_growth': growth, 'max_res_a': 15.0, 'starting_res_a': res_a})
    
    obs, _ = env.reset()
    history = []

    for day in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        history.append({
            "Day": day,
            "Size": obs[0],
            "Toxicity": obs[3],
            "Action": ["Rest", "Drug A", "Drug B"][action]
        })
        if done: break

    df = pd.DataFrame(history)
    
    # Show the shrinking tumor!
    st.line_chart(df.set_index("Day")[["Size", "Toxicity"]])
    st.table(df)