import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import numpy as np
import tensorflow as tf
import deepchem as dc
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Gemini configuration
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
genai.configure(api_key=GENAI_API_KEY)

# Step 1: Load Pretrained Model
model = dc.models.GraphConvModel(n_tasks=12, mode='classification')
print("Pretrained model loaded successfully.")


# Step 2: Predict Toxicity using SMILES
def predict_toxicity(smiles):
    try:
        # Step 3: Featurize SMILES using DeepChem's ConvMolFeaturizer
        featurizer = dc.feat.ConvMolFeaturizer()
        sample_molecules = featurizer.featurize([smiles])

        # Step 4: Wrap the features in a NumpyDataset
        dataset = dc.data.NumpyDataset(X=sample_molecules)

        # Step 5: Predict toxicity
        predictions = model.predict(dataset)

        # Define toxicity endpoints
        endpoints = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
            "SR-HSE", "SR-MMP", "SR-p53"
        ]

        # Process predictions with detailed explanations
        results = {}
        for i, endpoint in enumerate(endpoints):
            toxicity = "Toxic" if predictions[0, i, 1] > 0.5 else "Non-Toxic"
            confidence = predictions[0, i, 1]
            results[endpoint] = {
                "prediction": toxicity,
                "confidence": f"{confidence:.2f}"
            }

        return results, format_toxicity_explanation(results)

    except Exception as e:
        return {"Error": f"Prediction error: {str(e)}"}, ""


def format_toxicity_explanation(results):
    """Format toxicity results into a detailed explanation for the LLM"""
    explanation = "Toxicity Analysis Results:\n"

    # Group predictions by toxicity status
    toxic_endpoints = []
    non_toxic_endpoints = []

    for endpoint, data in results.items():
        if data["prediction"] == "Toxic":
            toxic_endpoints.append(f"{endpoint} (confidence: {data['confidence']})")
        else:
            non_toxic_endpoints.append(f"{endpoint} (confidence: {data['confidence']})")

    if toxic_endpoints:
        explanation += "\nPotentially Toxic Effects:\n"
        explanation += "\n".join(f"- {endpoint}" for endpoint in toxic_endpoints)

    if non_toxic_endpoints:
        explanation += "\nNon-Toxic Effects:\n"
        explanation += "\n".join(f"- {endpoint}" for endpoint in non_toxic_endpoints)

    return explanation


# Generate 2D molecule structure image
def generate_molecule_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))
        return img
    except Exception as e:
        st.error(f"Error generating molecule image: {str(e)}")
        return None


# Enhanced molecule properties calculation with explanations
def get_molecule_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, ""

        properties = {
            "Molecular Weight": f"{Descriptors.ExactMolWt(mol):.2f}",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "H-Bond Donors": str(Descriptors.NumHDonors(mol)),
            "H-Bond Acceptors": str(Descriptors.NumHAcceptors(mol)),
            "Rotatable Bonds": str(Descriptors.NumRotatableBonds(mol)),
            "TPSA": f"{Descriptors.TPSA(mol):.2f}",
            "Aromatic Rings": str(Descriptors.NumAromaticRings(mol))
        }

        # Create detailed explanation for LLM
        explanation = "Molecular Properties Analysis:\n"
        explanation += f"- The molecule has a molecular weight of {properties['Molecular Weight']} g/mol\n"
        explanation += f"- LogP value is {properties['LogP']}, indicating its lipophilicity\n"
        explanation += f"- Contains {properties['H-Bond Donors']} hydrogen bond donors and {properties['H-Bond Acceptors']} acceptors\n"
        explanation += f"- Has {properties['Rotatable Bonds']} rotatable bonds, affecting its flexibility\n"
        explanation += f"- Topological Polar Surface Area (TPSA) is {properties['TPSA']} Å²\n"
        explanation += f"- Contains {properties['Aromatic Rings']} aromatic rings\n"

        return properties, explanation
    except:
        return None, ""


# Enhanced Gemini query with comprehensive molecular context
def query_gemini(prompt, smiles, properties_explanation, toxicity_explanation):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Create comprehensive context
        context = f"""
Analysis for molecule with SMILES: {smiles}

{properties_explanation}

{toxicity_explanation}

Please consider all the above information when answering the following question.
        """

        response = model.generate_content(f"{context}\n\nQuestion: {prompt}")
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit UI
st.title("Molecular Analysis with AI Chat")

# Sidebar input
with st.sidebar:
    st.title("Input")
    smiles_input = st.text_input("Enter SMILES string:", "CC(=O)Oc1ccccc1C(=O)O")

    if st.button("Analyze Molecule"):
        st.session_state.analyze_clicked = True

    st.title("Chat")
    chat_input = st.text_area("Ask about the molecule:")
    if st.button("Send"):
        if chat_input:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "user", "content": chat_input})

            # Get comprehensive molecular information
            properties, properties_explanation = get_molecule_properties(smiles_input)
            predictions, toxicity_explanation = predict_toxicity(smiles_input)

            # Query LLM with enhanced context
            ai_response = query_gemini(chat_input, smiles_input, properties_explanation, toxicity_explanation)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# Molecule analysis section
if 'analyze_clicked' in st.session_state and st.session_state.analyze_clicked:
    with st.spinner('Analyzing molecule...'):
        tab1, tab2, tab3 = st.tabs(["Structure", "Properties", "Toxicity Predictions"])

        with tab1:
            mol_image = generate_molecule_image(smiles_input)
            if mol_image:
                st.image(mol_image, caption="2D Molecular Structure")
            else:
                st.error("Failed to generate molecular structure.")

        with tab2:
            properties, _ = get_molecule_properties(smiles_input)
            if properties:
                for k, v in properties.items():
                    st.metric(k, v)
            else:
                st.error("Failed to calculate molecular properties.")

        with tab3:
            predictions, _ = predict_toxicity(smiles_input)
            if "Error" in predictions:
                st.error(predictions["Error"])
            else:
                for endpoint, result in predictions.items():
                    st.write(f"**{endpoint}**")
                    st.write(f"- Prediction: {result['prediction']}")
                    st.write(f"- Confidence: {result['confidence']}")

# Chat history section
if 'chat_history' in st.session_state:
    st.header("Chat History")
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        st.write(f"**{role}:** {msg['content']}")