import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
import mlflow
import shap
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io

# Configurer le chemin MLflow pour accéder au modèle enregistré
mlflow.set_tracking_uri(
    "file:///Users/zi/Documents/OC - Data Scientist/Projet 7/mlruns"
)


# Définir une fonction  pour récupérer les noms des variables après transformation par un ColumnTransformer
def get_feature_names_from_column_transformer(
    ct: ColumnTransformer, original_features: List[str]
) -> List[str]:
    """
    Récupère les noms des variables après transformation par un ColumnTransformer.

    Paramètres:
        ct: ColumnTransformer utilisé pour les transformations
        original_features: Liste des noms des variables originales

    Retourne:
        Liste des noms des variables transformées
    """
    feature_names = []
    for name, transformer, columns in ct.transformers_:
        if name == "remainder":
            # Gérer les colonnes du remainder (passthrough)
            if transformer == "passthrough":
                remainder_features = [
                    original_features[i] if isinstance(i, int) else i for i in columns
                ]
                feature_names.extend(remainder_features)
            continue
        if hasattr(transformer, "get_feature_names_out"):
            input_features = [
                original_features[i] if isinstance(i, int) else i for i in columns
            ]
            transformed_features = transformer.get_feature_names_out(input_features)
            feature_names.extend(transformed_features)
        else:
            # Pour les transformers sans get_feature_names_out
            feature_names.extend(
                [original_features[i] if isinstance(i, int) else i for i in columns]
            )
    return feature_names


# Définir une fonction qui mappe les colonnes transformées avec la colonne originale
def create_feature_mapping(
    original_features: List[str], transformed_features: List[str]
) -> Dict[str, List[int]]:
    """
    Crée un mapping entre variables originales et leurs indices dans les variables transformées.

    Paramètres:
        original_features: Liste des noms des variables originales
        transformed_features: Liste des noms des variables transformées

    Retourne:
        Dictionnaire mappant chaque variable originale à ses indices transformés
    """
    mapping = {}
    for orig_feature in original_features:
        # Trouver toutes les colonnes transformées qui correspondent à cette variable
        matching_cols = []
        for i, trans_feature in enumerate(transformed_features):
            # Convertir trans_feature en string pour éviter l'erreur TypeError
            trans_feature_str = str(trans_feature)
            # Adapter les patterns selon votre encodage
            if orig_feature in trans_feature_str:
                matching_cols.append(i)
        mapping[orig_feature] = matching_cols
    return mapping


# Définir une fonction qui renvoie les valeurs SHAP pour chaque colonne orginale
def group_shap_by_original_features(
    shap_values: np.ndarray,
    feature_mapping: Dict[str, List[int]],
    original_features: List[str],
) -> pd.DataFrame:
    """
    Groupe les valeurs SHAP par variables originales en sommant les contributions.
    Paramètres:
    shap_values: Array numpy des valeurs SHAP (n_samples, n_features)
    feature_mapping: Mapping entre variables originales et indices transformés
    original_features: Liste des noms des variables originales
    Retourne:
    DataFrame avec les valeurs SHAP groupées par variable originale
    """

    # Calculer les valeurs shap pour chaque variable
    shap_data = {}
    for orig_feature in original_features:
        if orig_feature in feature_mapping and len(feature_mapping[orig_feature]) > 0:
            col_indices = feature_mapping[orig_feature]
            if len(col_indices) == 1:
                shap_data[orig_feature] = shap_values[:, col_indices[0]]
            else:
                # Plusieurs colonnes : somme des valeurs SHAP
                shap_data[orig_feature] = shap_values[:, col_indices].sum(axis=1)

    return pd.DataFrame(shap_data)


# Définir une focntion qui effectue la prédiction
def predict(df, seuil):
    """
    Prédit le risque de défaut de crédit et calcule les valeurs SHAP explicatives
    pour chaque variable et chaque prédiction.
    Paramètres:
    df (pd.DataFrame): Données clients
    Retourne:
    pd.DataFrame: Prédictions (probabilité, classe) et valeurs SHAP par variable originale
    """
    # Mettre l'ID du demandeur de prêt en index
    df = df.set_index("SK_ID_CURR").copy()

    # Charger le pipeline contenant les étapes de preprocessing et le modèle depuis ML Flow
    pipe = mlflow.sklearn.load_model("models:/Defaut_Credit_LGBM_Pipeline_VF/latest")

    # Effectuer la prédiction en récupérant les probabilités
    y_pred_df = pd.DataFrame(index=df.index)
    y_pred_df["PROBA_DEFAUT"] = pipe.predict_proba(df)[:, 1]
    y_pred_df["PRED_DEFAUT"] = 0
    y_pred_df.loc[y_pred_df["PROBA_DEFAUT"] > seuil, "PRED_DEFAUT"] = 1

    # Récupérer le nom des variables transformées
    ct = pipe["columntransformer"]
    transformed_features = get_feature_names_from_column_transformer(ct, df.columns)

    # Récupérer le mapping entre variables originales et variables transformées
    feature_mapping = create_feature_mapping(df.columns, transformed_features)

    # Récupérer le modèle
    model = pipe.named_steps["lgbmclassifier"]

    # Transformer X
    X_transformed = pipe[:-1].transform(df)

    # Créer l'explainer avec le modèle
    explainer = shap.TreeExplainer(model)

    # Calculer les valeurs SHAP pour chaque prédiction
    shap_values = explainer.shap_values(X_transformed)

    # Récupérer les valeurs SHAP groupées par variables originales
    shap_df = group_shap_by_original_features(
        shap_values, feature_mapping, df.columns
    ).set_index(df.index)

    # Joindre les prédictions aux valeurs shap par variable
    resultat_df = y_pred_df.merge(
        shap_df, left_index=True, right_index=True, how="left"
    )

    return resultat_df


# Créer l'instance FastAPI avec métadonnées pour la documentation automatique
app = FastAPI(
    title="Prédiction Défaut Crédit API",
    description="API permettant de prédire le risque de défaut d'un client avec le poids de chaque variable",
    version="1.0.0",
)


# Définir le endpoint POST sur la route "/predict"
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint de prédiction pour le risque de défaut de crédit
    Paramètre: file: Fichier CSV contenant les données clients
    Retourne: List[Dict]: Résultats avec prédictions et valeurs SHAP par client
    """
    # Lire le fichier uploadé
    df = pd.read_csv(file.file)

    # Effectué la prédiction
    results_df = predict(df, 0.48)

    # Retourner les résultats sous forme de dictionnaire pour
    # conversion automatique par FastAPI en format JSON
    return results_df.reset_index().to_dict(orient="records")


# Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
