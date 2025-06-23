import pandas as pd
from utilities import predict
import mlflow
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Configurer le chemin MLflow pour accéder au modèle enregistré
mlflow.set_tracking_uri(
    "file:///Users/zi/Documents/OC - Data Scientist/Projet 7/mlruns"
)

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

    # Vérifier l'extension du fichier
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail=f"Format de fichier non supporté ({file.filename}). Veuillez fournir un fichier CSV."
        )

    # Lire le fichier uploadé
    df = pd.read_csv(file.file)

    # Vérifier que le fichier n'est pas vide
    if df.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail="Le fichier CSV est vide. Veuillez fournir un fichier contenant des données."
        )

    # Vérifier que le fichier n'est pas trop volumineux
    if df.shape[0] > 1000:
        raise HTTPException(
            status_code=413,
            detail=f"Le fichier est trop volumineux ({df.shape[0]} lignes). Maximum autorisé: 1 000 lignes."
        )
    
    # Vérifier que le fichier contient toutes les colonnes nécessaires
    needed_columns = {
        'SK_ID_CURR': 'int64', 'NAME_CONTRACT_TYPE': 'object', 'CODE_GENDER': 'object', 'FLAG_OWN_CAR': 'object', 'FLAG_OWN_REALTY': 'object',
        'CNT_CHILDREN': 'int64', 'AMT_INCOME_TOTAL': 'float64', 'AMT_CREDIT': 'float64', 'AMT_ANNUITY': 'float64', 'AMT_GOODS_PRICE': 'float64',
        'NAME_TYPE_SUITE': 'object', 'NAME_INCOME_TYPE': 'object', 'NAME_EDUCATION_TYPE': 'object', 'NAME_FAMILY_STATUS': 'object', 'NAME_HOUSING_TYPE': 'object',
        'REGION_POPULATION_RELATIVE': 'float64', 'DAYS_BIRTH': 'int64', 'DAYS_EMPLOYED': 'int64', 'DAYS_REGISTRATION': 'float64', 'DAYS_ID_PUBLISH': 'int64',
        'OWN_CAR_AGE': 'float64', 'FLAG_MOBIL': 'int64', 'FLAG_EMP_PHONE': 'int64', 'FLAG_WORK_PHONE': 'int64', 'FLAG_CONT_MOBILE': 'int64',
        'FLAG_PHONE': 'int64', 'FLAG_EMAIL': 'int64', 'OCCUPATION_TYPE': 'object', 'CNT_FAM_MEMBERS': 'float64', 'REGION_RATING_CLIENT': 'int64',
        'REGION_RATING_CLIENT_W_CITY': 'int64', 'WEEKDAY_APPR_PROCESS_START': 'object', 'HOUR_APPR_PROCESS_START': 'int64', 'REG_REGION_NOT_LIVE_REGION': 'int64', 'REG_REGION_NOT_WORK_REGION': 'int64',
        'LIVE_REGION_NOT_WORK_REGION': 'int64', 'REG_CITY_NOT_LIVE_CITY': 'int64', 'REG_CITY_NOT_WORK_CITY': 'int64', 'LIVE_CITY_NOT_WORK_CITY': 'int64', 'ORGANIZATION_TYPE': 'object',
        'EXT_SOURCE_1': 'float64', 'EXT_SOURCE_2': 'float64', 'EXT_SOURCE_3': 'float64', 'APARTMENTS_AVG': 'float64', 'BASEMENTAREA_AVG': 'float64',
        'YEARS_BEGINEXPLUATATION_AVG': 'float64', 'YEARS_BUILD_AVG': 'float64', 'COMMONAREA_AVG': 'float64', 'ELEVATORS_AVG': 'float64', 'ENTRANCES_AVG': 'float64',
        'FLOORSMAX_AVG': 'float64', 'FLOORSMIN_AVG': 'float64', 'LANDAREA_AVG': 'float64', 'LIVINGAPARTMENTS_AVG': 'float64', 'LIVINGAREA_AVG': 'float64',
        'NONLIVINGAPARTMENTS_AVG': 'float64', 'NONLIVINGAREA_AVG': 'float64', 'APARTMENTS_MODE': 'float64', 'BASEMENTAREA_MODE': 'float64', 'YEARS_BEGINEXPLUATATION_MODE': 'float64',
        'YEARS_BUILD_MODE': 'float64', 'COMMONAREA_MODE': 'float64', 'ELEVATORS_MODE': 'float64', 'ENTRANCES_MODE': 'float64', 'FLOORSMAX_MODE': 'float64',
        'FLOORSMIN_MODE': 'float64', 'LANDAREA_MODE': 'float64', 'LIVINGAPARTMENTS_MODE': 'float64', 'LIVINGAREA_MODE': 'float64', 'NONLIVINGAPARTMENTS_MODE': 'float64',
        'NONLIVINGAREA_MODE': 'float64', 'APARTMENTS_MEDI': 'float64', 'BASEMENTAREA_MEDI': 'float64', 'YEARS_BEGINEXPLUATATION_MEDI': 'float64', 'YEARS_BUILD_MEDI': 'float64',
        'COMMONAREA_MEDI': 'float64', 'ELEVATORS_MEDI': 'float64', 'ENTRANCES_MEDI': 'float64', 'FLOORSMAX_MEDI': 'float64', 'FLOORSMIN_MEDI': 'float64',
        'LANDAREA_MEDI': 'float64', 'LIVINGAPARTMENTS_MEDI': 'float64', 'LIVINGAREA_MEDI': 'float64', 'NONLIVINGAPARTMENTS_MEDI': 'float64', 'NONLIVINGAREA_MEDI': 'float64',
        'FONDKAPREMONT_MODE': 'float64', 'HOUSETYPE_MODE': 'object', 'TOTALAREA_MODE': 'float64', 'WALLSMATERIAL_MODE': 'object', 'EMERGENCYSTATE_MODE': 'object',
        'OBS_30_CNT_SOCIAL_CIRCLE': 'float64', 'DEF_30_CNT_SOCIAL_CIRCLE': 'float64', 'OBS_60_CNT_SOCIAL_CIRCLE': 'float64', 'DEF_60_CNT_SOCIAL_CIRCLE': 'float64', 'DAYS_LAST_PHONE_CHANGE': 'float64',
        'FLAG_DOCUMENT_2': 'int64', 'FLAG_DOCUMENT_3': 'int64', 'FLAG_DOCUMENT_4': 'int64', 'FLAG_DOCUMENT_5': 'int64', 'FLAG_DOCUMENT_6': 'int64',
        'FLAG_DOCUMENT_7': 'int64', 'FLAG_DOCUMENT_8': 'int64', 'FLAG_DOCUMENT_9': 'int64', 'FLAG_DOCUMENT_10': 'int64', 'FLAG_DOCUMENT_11': 'int64',
        'FLAG_DOCUMENT_12': 'int64', 'FLAG_DOCUMENT_13': 'int64', 'FLAG_DOCUMENT_14': 'int64', 'FLAG_DOCUMENT_15': 'int64', 'FLAG_DOCUMENT_16': 'int64',
        'FLAG_DOCUMENT_17': 'int64', 'FLAG_DOCUMENT_18': 'int64', 'FLAG_DOCUMENT_19': 'int64', 'FLAG_DOCUMENT_20': 'int64', 'FLAG_DOCUMENT_21': 'int64',
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'float64', 'AMT_REQ_CREDIT_BUREAU_DAY': 'float64', 'AMT_REQ_CREDIT_BUREAU_WEEK': 'float64', 'AMT_REQ_CREDIT_BUREAU_MON': 'float64', 'AMT_REQ_CREDIT_BUREAU_QRT': 'float64',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'float64'
    }
    
    col_manquantes = [col for col in needed_columns.keys() if col not in df.columns]
    if len(col_manquantes) > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier ne contient pas toutes les colonnes nécessaires. Colonnes manquantes: {col_manquantes}"
        )
    
    # Vérifier que le fichier contient au moins l'ID
    if df['SK_ID_CURR'].isna().sum() > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier contient {df['SK_ID_CURR'].isna().sum()} ligne(s) avec des valeurs manquantes dans la colonne SK_ID_CURR. Toutes les lignes doivent avoir un identifiant client."
        )
    
    # Vérifier les types des colonnes
    try:
        df = df.astype(needed_columns)
    except (ValueError, TypeError) as e:
        # Trouver les colonnes pour lesquelles la conversion a échouée
        failed_columns = []
        for col, dtype in needed_columns.items():
            if col in df.columns:
                try:
                    df[col].astype(dtype)
                except (ValueError, TypeError):
                    failed_columns.append({
                        'column': col,
                        'expected_type': dtype,
                        'current_type': str(df[col].dtype)
                    })
        
        error_msg = "Impossible de convertir les types de données pour les colonnes: "
        error_msg += ", ".join([f"{fc['column']} (attendu: {fc['expected_type']}, trouvé: {fc['current_type']})" 
                                for fc in failed_columns])
        
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    # Effectuer la prédiction
    results_df = predict(df, 0.48)

    # Retourner les résultats sous forme de dictionnaire pour
    # conversion automatique par FastAPI en format JSON
    return results_df.reset_index().to_dict(orient="records")

# Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)