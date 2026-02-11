import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Système de Classification des Barrages",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne et professionnel
st.markdown("""
    <style>
    /* Palette de couleurs professionnelle */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
        --danger-color: #C73E1D;
        --background-light: #F8F9FA;
        --text-dark: #212529;
    }
    
    /* Style général */
    .main {
        background-color: #F8F9FA;
    }
    
    /* En-têtes personnalisés */
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%);
        padding: 2rem;
        border-radius: 10px;
        color: black;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cartes d'information */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
        color: black
    }
    
    .info-card h3 {
        color: #2E86AB;
        margin-top: 0;
        font-weight: 600;
    }
    
    .info-card p {
        color: #black;
        line-height: 1.6;
    }
    
    /* Style pour les résultats de classification */
    .result-critical {
        background-color: #FFE8E8;
        border-left: 5px solid #C73E1D;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: black
    }
    
    .result-normal {
        background-color: #E8F8F5;
        border-left: 5px solid #06A77D;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .result-low {
        background-color: #FFF4E6;
        border-left: 5px solid #F18F01;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Boutons personnalisés */
    .stButton>button {
        background-color: #2E86AB;
        color: black;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1A5F7A;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* Métriques personnalisées */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86AB;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #495057;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Animation de chargement */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_v2_s()
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        
        model_path = "efficientnet_v2_compressed.pth"
        state_dict = torch.load(model_path, map_location="cpu")
        float32_state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
        model.load_state_dict(float32_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class_names = ["Critical", "Low", "Normal"]
class_colors = {
    "Critical": "#C73E1D",
    "Low": "#F18F01",
    "Normal": "#06A77D"
}

class_descriptions = {
    "Critical": "⚠️ État critique - Intervention urgente requise",
    "Low": "⚡ État de vigilance - Surveillance recommandée",
    "Normal": "✅ État normal - Aucune action requise"
}

# Initialiser le session state pour l'historique
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'daily_stats' not in st.session_state:
    st.session_state.daily_stats = {
        'total': 0,
        'critical': 0,
        'low': 0,
        'normal': 0,
        'avg_confidence': 0
    }

# Sidebar - Navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">🏗️ Dam Monitor</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "📋 Navigation",
        ["🏠 Accueil", "🔍 Classification", "📊 Tableau de Bord", "📚 Documentation", "ℹ️ À propos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    
    confidence_threshold = st.slider(
        "Seuil de confiance (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Seuil minimum de confiance pour accepter une prédiction"
    )
    
    show_probabilities = st.checkbox("Afficher les probabilités", value=True)
    
    st.markdown("---")
    st.markdown("### 📈 Statistiques de session")
    if 'total_classifications' not in st.session_state:
        st.session_state.total_classifications = 0
    st.metric("Classifications", st.session_state.total_classifications)

# =======================
# PAGE: ACCUEIL
# =======================
if page == "🏠 Accueil":
    st.markdown("""
        <div class="main-header">
            <h1>🏗️ Système de Classification des Barrages</h1>
            <p>Intelligence Artificielle pour la surveillance et l'évaluation de l'état des barrages</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🎯 Précision</h3>
                <p>Modèle entraîné avec EfficientNet V2 offrant une précision élevée dans la détection des anomalies structurelles</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>⚡ Rapidité</h3>
                <p>Analyse en temps réel permettant une évaluation instantanée de multiples images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>🔒 Fiabilité</h3>
                <p>Système optimisé pour identifier les états critiques et garantir la sécurité</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🚀 Fonctionnalités principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Classification Multi-Images** 📸
        - Téléchargement simultané de plusieurs images
        - Analyse batch pour gain de temps
        - Résultats structurés sous forme de tableau
        
        **Visualisations Avancées** 📊
        - Graphiques de probabilités interactifs
        - Comparaison visuelle des résultats
        - Export des données
        """)
    
    with col2:
        st.markdown("""
        **Analyses Détaillées** 🔬
        - Niveau de confiance pour chaque prédiction
        - Distribution des probabilités par classe
        - Recommandations basées sur les résultats
        
        **Documentation Complète** 📚
        - Types de barrages expliqués
        - Guide d'utilisation détaillé
        - Informations techniques
        """)
    
    st.markdown("---")
    st.info("💡 **Conseil:** Commencez par la section 'Classification' pour analyser vos images de barrages.")

# =======================
# PAGE: CLASSIFICATION
# =======================
elif page == "🔍 Classification":
    st.markdown("""
        <div class="main-header">
            <h1>🔍 Classification des Barrages</h1>
            <p>Téléchargez une ou plusieurs images pour analyse</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier que le fichier 'efficientnet_v2_compressed.pth' est présent.")
        st.stop()
    
    # Upload multiple images
    uploaded_files = st.file_uploader(
        "📤 Sélectionnez une ou plusieurs images",
        type=["jpg", "jpeg", "png", "tiff", "tif"],
        accept_multiple_files=True,
        help="Formats acceptés: JPG, JPEG, PNG, TIFF"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) téléchargée(s)")
        
        if st.button("🚀 Lancer l'analyse", type="primary"):
            results = []
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Analyse en cours: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                
                try:
                    # Charger et prétraiter l'image
                    image = Image.open(uploaded_file).convert("RGB")
                    img_tensor = preprocess(image).unsqueeze(0)
                    
                    # Prédiction
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence_values = probs[0].tolist()
                        predicted_idx = torch.argmax(probs, dim=1).item()
                        predicted_class = class_names[predicted_idx]
                        confidence = confidence_values[predicted_idx] * 100
                    
                    # Stocker les résultats
                    results.append({
                        'image': image,
                        'filename': uploaded_file.name,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'probabilities': {
                            class_names[i]: confidence_values[i] * 100 
                            for i in range(len(class_names))
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse de {uploaded_file.name}: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Analyse terminée!")
            st.session_state.total_classifications += len(uploaded_files)
            
            # Mettre à jour les statistiques du dashboard
            for result in results:
                st.session_state.classification_history.append({
                    'filename': result['filename'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Mettre à jour les stats journalières
                st.session_state.daily_stats['total'] += 1
                if result['prediction'] == 'Critical':
                    st.session_state.daily_stats['critical'] += 1
                elif result['prediction'] == 'Low':
                    st.session_state.daily_stats['low'] += 1
                else:
                    st.session_state.daily_stats['normal'] += 1
            
            # Calculer la confiance moyenne
            if st.session_state.daily_stats['total'] > 0:
                total_confidence = sum(r['confidence'] for r in results)
                current_avg = st.session_state.daily_stats['avg_confidence']
                previous_total = st.session_state.daily_stats['total'] - len(results)
                
                if previous_total > 0:
                    st.session_state.daily_stats['avg_confidence'] = (
                        (current_avg * previous_total + total_confidence) / 
                        st.session_state.daily_stats['total']
                    )
                else:
                    st.session_state.daily_stats['avg_confidence'] = total_confidence / len(results)
            
            # Affichage des résultats sous forme de tableau
            st.markdown("---")
            st.subheader("📊 Résultats de l'analyse")
            
            # Créer des colonnes pour chaque image
            num_cols = min(3, len(results))  # Maximum 3 colonnes
            
            for i in range(0, len(results), num_cols):
                cols = st.columns(num_cols)
                
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        result = results[i + j]
                        
                        with col:
                            # Image
                            st.image(result['image'], use_container_width=True)
                            
                            # Nom du fichier
                            st.markdown(f"**📁 {result['filename']}**")
                            
                            # Résultat de prédiction
                            pred_class = result['prediction']
                            confidence = result['confidence']
                            
                            # Coloration selon la classe
                            if pred_class == "Critical":
                                st.markdown(f"""
                                    <div class="result-critical">
                                        <strong>⚠️ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """, unsafe_allow_html=True)
                            elif pred_class == "Normal":
                                st.markdown(f"""
                                    <div class="result-normal">
                                        <strong>✅ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="result-low">
                                        <strong>⚡ {pred_class}</strong><br>
                                        Confiance: {confidence:.1f}%<br>
                                        {class_descriptions[pred_class]}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Graphique des probabilités
                            if show_probabilities:
                                probs = result['probabilities']
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(probs.keys()),
                                        y=list(probs.values()),
                                        marker_color=[class_colors[k] for k in probs.keys()],
                                        text=[f"{v:.1f}%" for v in probs.values()],
                                        textposition='auto',
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Probabilités",
                                    xaxis_title="Classe",
                                    yaxis_title="Probabilité (%)",
                                    height=250,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif
            st.markdown("---")
            st.subheader("📋 Tableau récapitulatif")
            
            df_results = pd.DataFrame([
                {
                    'Fichier': r['filename'],
                    'Prédiction': r['prediction'],
                    'Confiance (%)': f"{r['confidence']:.2f}",
                    'Critical (%)': f"{r['probabilities']['Critical']:.2f}",
                    'Low (%)': f"{r['probabilities']['Low']:.2f}",
                    'Normal (%)': f"{r['probabilities']['Normal']:.2f}",
                }
                for r in results
            ])
            
            st.dataframe(df_results, use_container_width=True, height=300)
            
            # Statistiques globales
            st.markdown("---")
            st.subheader("📈 Statistiques globales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            critical_count = sum(1 for r in results if r['prediction'] == 'Critical')
            low_count = sum(1 for r in results if r['prediction'] == 'Low')
            normal_count = sum(1 for r in results if r['prediction'] == 'Normal')
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            with col1:
                st.metric("Total analysé", len(results))
            with col2:
                st.metric("Critical", critical_count, delta=None if critical_count == 0 else "⚠️")
            with col3:
                st.metric("Low", low_count)
            with col4:
                st.metric("Normal", normal_count, delta=None if normal_count == 0 else "✅")
            
            # Graphique de distribution
            dist_fig = go.Figure(data=[
                go.Pie(
                    labels=['Critical', 'Low', 'Normal'],
                    values=[critical_count, low_count, normal_count],
                    marker_colors=[class_colors['Critical'], class_colors['Low'], class_colors['Normal']],
                    hole=0.4
                )
            ])
            
            dist_fig.update_layout(
                title="Distribution des classifications",
                height=400
            )
            
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Export CSV
            st.markdown("---")
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    else:
        st.info("👆 Veuillez télécharger au moins une image pour commencer l'analyse.")

# =======================
# PAGE: TABLEAU DE BORD
# =======================
elif page == "📊 Tableau de Bord":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Tableau de Bord</h1>
            <p>Vue d'ensemble et statistiques</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Récupérer les statistiques
    stats = st.session_state.daily_stats
    history = st.session_state.classification_history
    
    if stats['total'] == 0:
        st.info("📌 Aucune classification effectuée pour le moment. Allez dans la section 'Classification' pour commencer à analyser des images.")
    else:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['total']}</div>
                    <div class="metric-label">Classifications totales</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #C73E1D;">{stats['critical']}</div>
                    <div class="metric-label">Barrages critiques</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #F18F01;">{stats['low']}</div>
                    <div class="metric-label">Barrages en vigilance</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['avg_confidence']:.1f}%</div>
                    <div class="metric-label">Confiance moyenne</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution des classifications")
            
            # Graphique en camembert
            dist_fig = go.Figure(data=[
                go.Pie(
                    labels=['Critical', 'Low', 'Normal'],
                    values=[stats['critical'], stats['low'], stats['normal']],
                    marker_colors=[class_colors['Critical'], class_colors['Low'], class_colors['Normal']],
                    hole=0.4,
                    textinfo='label+percent',
                    textfont_size=14
                )
            ])
            
            dist_fig.update_layout(
                height=350,
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Répartition par état")
            
            # Graphique en barres
            bar_fig = go.Figure(data=[
                go.Bar(
                    x=['Critical', 'Low', 'Normal'],
                    y=[stats['critical'], stats['low'], stats['normal']],
                    marker_color=[class_colors['Critical'], class_colors['Low'], class_colors['Normal']],
                    text=[stats['critical'], stats['low'], stats['normal']],
                    textposition='auto',
                    textfont_size=16
                )
            ])
            
            bar_fig.update_layout(
                height=350,
                yaxis_title="Nombre",
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(bar_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Historique des classifications
        st.subheader("📋 Historique des classifications récentes")
        
        if len(history) > 0:
            # Afficher les 10 dernières classifications
            recent_history = history[-10:][::-1]  # Inverser pour avoir les plus récentes en premier
            
            df_history = pd.DataFrame(recent_history)
            
            # Formater le dataframe
            df_display = df_history[['timestamp', 'filename', 'prediction', 'confidence']].copy()
            df_display.columns = ['Date/Heure', 'Fichier', 'Prédiction', 'Confiance (%)']
            df_display['Confiance (%)'] = df_display['Confiance (%)'].apply(lambda x: f"{x:.2f}")
            
            # Ajouter des emojis selon la prédiction
            df_display['Statut'] = df_display['Prédiction'].apply(
                lambda x: '⚠️ Critical' if x == 'Critical' else ('⚡ Low' if x == 'Low' else '✅ Normal')
            )
            
            st.dataframe(
                df_display[['Date/Heure', 'Fichier', 'Statut', 'Confiance (%)']],
                use_container_width=True,
                height=400
            )
            
            # Option de téléchargement de l'historique complet
            st.markdown("---")
            full_history_df = pd.DataFrame(history)
            csv = full_history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger l'historique complet (CSV)",
                data=csv,
                file_name=f"historique_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        # Alertes si présence de cas critiques
        if stats['critical'] > 0:
            st.markdown("---")
            st.error(f"⚠️ **Attention:** {stats['critical']} barrage(s) en état critique détecté(s). Une inspection urgente est recommandée.")
        
        if stats['low'] > 0:
            st.warning(f"⚡ **Vigilance:** {stats['low']} barrage(s) en état de surveillance. Une maintenance préventive est conseillée.")
        
        # Statistiques supplémentaires
        st.markdown("---")
        st.subheader("📈 Statistiques détaillées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if stats['total'] > 0:
                critical_pct = (stats['critical'] / stats['total']) * 100
                st.metric("Taux de criticité", f"{critical_pct:.1f}%")
        
        with col2:
            if stats['total'] > 0:
                normal_pct = (stats['normal'] / stats['total']) * 100
                st.metric("Taux de normalité", f"{normal_pct:.1f}%")
        
        with col3:
            st.metric("Total analysé", f"{stats['total']} images")

# =======================
# PAGE: DOCUMENTATION
# =======================
elif page == "📚 Documentation":
    st.markdown("""
        <div class="main-header">
            <h1>📚 Documentation</h1>
            <p>Guide complet sur les types de barrages et leur classification</p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Types de barrages", "🎯 Classes de risque", "💡 Guide d'utilisation", "🔧 Informations techniques"])
    
    with tab1:
        st.header("Types de barrages")
        
        st.markdown("""
        ### 1. Barrage-poids 🏔️
        
        **Description:** Structure massive qui résiste à la pression de l'eau par son propre poids.
        
        **Caractéristiques:**
        - Construction en béton ou en maçonnerie
        - Profil triangulaire
        - Très stable et durable
        - Nécessite des fondations rocheuses solides
        
        **Exemples célèbres:** Barrage Hoover (États-Unis), Barrage Grand Coulee (États-Unis)
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Hoover_Dam_aerial_view.jpg/1200px-Hoover_Dam_aerial_view.jpg", 
                 caption="Exemple de barrage-poids: Barrage Hoover", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 2. Barrage-voûte 🌉
        
        **Description:** Structure arquée qui transfère la pression de l'eau vers les flancs de la vallée.
        
        **Caractéristiques:**
        - Forme incurvée vers l'amont
        - Économie de matériaux
        - Convient aux vallées étroites
        - Nécessite des appuis rocheux de qualité
        
        **Exemples célèbres:** Barrage de Vajont (Italie), Barrage de Monteynard (France)
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Barrage_de_Monteynard_-_2.JPG/1200px-Barrage_de_Monteynard_-_2.JPG", 
                 caption="Exemple de barrage-voûte: Barrage de Monteynard", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 3. Barrage en remblai 🌊
        
        **Description:** Construction en terre ou enrochement avec un noyau imperméable.
        
        **Caractéristiques:**
        - Utilisation de matériaux locaux
        - Adapté à tout type de fondation
        - Plus économique pour les grandes hauteurs
        - Nécessite un entretien régulier
        
        **Exemples célèbres:** Barrage de Tarbela (Pakistan), Barrage des Trois Gorges (Chine)
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/TGD_Dam.jpg/1200px-TGD_Dam.jpg", 
                 caption="Exemple de barrage en remblai: Barrage des Trois Gorges", use_container_width=True)
    
    with tab2:
        st.header("Classes de risque")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div class="result-critical" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">⚠️ Critical</h3>
                    <p><strong>Niveau de risque:</strong> Élevé</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Caractéristiques identifiées:**
            - Fissures importantes visibles
            - Déformations structurelles
            - Infiltrations majeures
            - Signes de dégradation avancée
            
            **Actions requises:**
            - Inspection immédiate par des experts
            - Évaluation structurelle complète
            - Mise en place de mesures d'urgence
            - Surveillance renforcée
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div class="result-low" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">⚡ Low</h3>
                    <p><strong>Niveau de risque:</strong> Modéré</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Caractéristiques identifiées:**
            - Fissures mineures localisées
            - Début d'altération des matériaux
            - Infiltrations légères
            - Signes de vieillissement normal
            
            **Actions recommandées:**
            - Surveillance régulière
            - Maintenance préventive
            - Documentation photographique
            - Planification d'interventions
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div class="result-normal" style="margin: 1rem 0;">
                    <h3 style="margin-top: 0;">✅ Normal</h3>
                    <p><strong>Niveau de risque:</strong> Faible</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Caractéristiques identifiées:**
            - Structure intègre
            - Absence de fissures significatives
            - Bon état général
            - Fonctionnement nominal
            
            **Actions requises:**
            - Inspections de routine
            - Maintenance standard
            - Surveillance continue
            - Documentation régulière
            """)
    
    with tab3:
        st.header("Guide d'utilisation")
        
        st.markdown("""
        ### 🚀 Démarrage rapide
        
        1. **Accédez à la section Classification** 🔍
           - Cliquez sur "Classification" dans le menu latéral
        
        2. **Téléchargez vos images** 📤
           - Cliquez sur "Parcourir les fichiers"
           - Sélectionnez une ou plusieurs images
           - Formats acceptés: JPG, JPEG, PNG, TIFF
        
        3. **Lancez l'analyse** 🚀
           - Cliquez sur "Lancer l'analyse"
           - Attendez le traitement (quelques secondes par image)
        
        4. **Consultez les résultats** 📊
           - Visualisez les prédictions pour chaque image
           - Analysez les probabilités et la confiance
           - Téléchargez le rapport CSV si nécessaire
        
        ---
        
        ### 💡 Conseils pour de meilleurs résultats
        
        **Qualité des images:**
        - ✅ Utilisez des images nettes et bien éclairées
        - ✅ Privilégiez une résolution minimale de 800x600 pixels
        - ✅ Cadrez correctement la structure du barrage
        - ❌ Évitez les images floues ou sous-exposées
        - ❌ Évitez les obstructions (arbres, brouillard)
        
        **Angle de prise de vue:**
        - Vue frontale de la structure
        - Distance suffisante pour voir l'ensemble
        - Conditions météorologiques claires
        
        **Analyse des résultats:**
        - Vérifiez le niveau de confiance (>70% recommandé)
        - Comparez les probabilités entre classes
        - Considérez le contexte et l'historique du barrage
        
        ---
        
        ### ⚙️ Paramètres avancés
        
        Dans le menu latéral, vous pouvez ajuster:
        - **Seuil de confiance:** Niveau minimum pour valider une prédiction
        - **Affichage des probabilités:** Montrer/masquer les graphiques détaillés
        """)
    
    with tab4:
        st.header("Informations techniques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧠 Architecture du modèle
            
            **Réseau de base:** EfficientNet V2-S
            - Optimisé pour la performance
            - Architecture moderne et efficace
            - Pré-entraîné sur ImageNet
            
            **Couches personnalisées:**
            - Dropout (20%)
            - 4 couches denses (1024 neurones)
            - 1 couche dense (512 neurones)
            - Couche de sortie (3 classes)
            
            **Fonction d'activation:** ReLU
            **Optimisation:** Quantification FP16
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Spécifications
            
            **Taille d'entrée:** 224x224 pixels
            **Normalisation:** ImageNet standard
            - Mean: [0.485, 0.456, 0.406]
            - Std: [0.229, 0.224, 0.225]
            
            **Classes de sortie:**
            1. Critical (Critique)
            2. Low (Faible)
            3. Normal (Normal)
            
            **Sortie:** Probabilités softmax
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔬 Processus de traitement
        
        1. **Prétraitement de l'image**
           - Redimensionnement à 256x256
           - Recadrage central à 224x224
           - Conversion en tenseur
           - Normalisation
        
        2. **Inférence**
           - Propagation avant dans le réseau
           - Calcul des logits
           - Application du softmax
        
        3. **Post-traitement**
           - Extraction de la classe prédite
           - Calcul des probabilités
           - Génération des visualisations
        
        ### 📈 Performance
        
        - **Temps d'inférence:** ~0.5-1s par image (CPU)
        - **Précision:** Optimisée pour la détection de défauts structurels
        - **Compression:** Modèle optimisé avec quantification FP16
        """)

# =======================
# PAGE: À PROPOS
# =======================
elif page == "ℹ️ À propos":
    st.markdown("""
        <div class="main-header">
            <h1>ℹ️ À propos</h1>
            <p>Information sur le système et l'équipe</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        
        Le **Système de Classification des Barrages** utilise l'intelligence artificielle pour améliorer 
        la surveillance et l'évaluation de l'état des infrastructures hydrauliques. Notre objectif est 
        de fournir un outil rapide, précis et accessible pour aider les ingénieurs et les responsables 
        dans la gestion préventive des barrages.
        
        ### 🔬 Technologie
        
        Cette application utilise:
        - **PyTorch** pour le deep learning
        - **EfficientNet V2** comme architecture de réseau neuronal
        - **Streamlit** pour l'interface utilisateur
        - **Plotly** pour les visualisations interactives
        
        ### 🌟 Fonctionnalités clés
        
        - Classification multi-classe (Critical, Low, Normal)
        - Analyse batch de plusieurs images
        - Visualisations interactives des résultats
        - Export des données au format CSV
        - Interface intuitive et professionnelle
        
        ### 📞 Contact
        
        Pour toute question, suggestion ou collaboration:
        - 📧 Email: contact@dam-monitor.com
        - 🌐 Site web: www.dam-monitor.com
        - 📱 Téléphone: +237 XXX XXX XXX
        """)
    
    with col2:
        st.markdown("""
        ### 📦 Version
        
        **v2.0.0**
        
        *Dernière mise à jour:*  
        Février 2026
        
        ---
        
        ### 📝 Licence
        
        © 2026 Dam Monitor  
        Tous droits réservés
        
        ---
        
        ### 🙏 Remerciements
        
        Merci aux contributeurs  
        et à la communauté  
        open-source
        """)
        
        st.markdown("---")
        
        st.info("""
        **Note:** Ce système est un outil d'aide à la décision. 
        Les résultats doivent toujours être validés par des experts qualifiés.
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Développé avec ❤️ par l'équipe Dam Monitor | © 2026 Tous droits réservés</p>
        <p style="font-size: 0.8rem;">Version 2.0.0 | Propulsé par PyTorch & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
