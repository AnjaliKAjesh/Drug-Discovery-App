
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from environment import DrugMDP, fragments
from dqn_model import DQN
from utils import encode_smiles

# Page configuration
st.set_page_config(page_title='RL Molecule Builder', layout='centered')
st.title('🧪 Reinforcement Learning Molecule Builder')
st.markdown("""Build molecules step-by-step using a pretrained DQN model.
Select a fragment and view the updated molecule along with reward feedback.""")

# Load pretrained model
model = DQN(state_size=64, action_size=len(fragments))
try:
    model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cpu')))
    model.eval()
    st.success("✅ Pretrained model loaded successfully.")
    model_loaded = True
except:
    st.warning("⚠️ Pretrained model not found. Manual fragment selection enabled.")
    model_loaded = False

# Session state setup
if 'env' not in st.session_state:
    st.session_state.env = DrugMDP(max_steps=10)
    st.session_state.state = st.session_state.env.reset()

env = st.session_state.env
smiles = env.smiles
mol = Chem.MolFromSmiles(smiles)

# Display current molecule
st.subheader("Current Molecule")
st.image(Draw.MolToImage(mol, size=(300, 300)), caption=smiles)

# Select or predict next action
st.subheader("Select Fragment")
if model_loaded:
    state_tensor = torch.tensor([encode_smiles(smiles)], dtype=torch.float32)
    q_vals = model(state_tensor).detach().numpy().flatten()
    best_action = int(q_vals.argmax())
    st.code(f"Recommended by model: {fragments[best_action]}")
    action = best_action
else:
    selected = st.selectbox("Choose a fragment to attach:", fragments)
    action = fragments.index(selected)

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Attach Fragment"):
        _, reward, done = env.step(action)
        st.experimental_rerun()

with col2:
    if st.button("🔄 Reset Molecule"):
        st.session_state.env = DrugMDP(max_steps=10)
        st.session_state.state = st.session_state.env.reset()
        st.experimental_rerun()
