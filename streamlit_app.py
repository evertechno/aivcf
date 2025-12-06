"""
Streamlit app: Financial correlation + ML explorer
- Connects to Supabase using st.secrets['SUPABASE_URL'] and st.secrets['SUPABASE_ANON_KEY']
- Fetches historical_prices + financial tables
- Does feature engineering: returns, rolling stats, volume z-score, merges nearest financial period
- Runs correlation analysis and trains simple scikit-learn models (RandomForest) to predict large next-day moves

Instructions:
1. Put this file in a project folder.
2. Create a requirements.txt with: streamlit, supabase, pandas, numpy, scikit-learn, plotly, matplotlib, joblib
3. Add your Supabase keys to Streamlit secrets (or set environment variables):
   [secrets]
   SUPABASE_URL = "https://..."  # your project url
   SUPABASE_ANON_KEY = "..."
4. Run: streamlit run streamlit_supabase_financial_ai.py

"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
import plotly.express as px
import joblib
import io
import base64

st.set_page_config(page_title="Financial Correlations & ML Explorer", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def get_supabase():
    # expects st.secrets to contain SUPABASE_URL and SUPABASE_ANON_KEY
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Please add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
        st.stop()
    return create_client(url, key)


def safe_numeric(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return s
    try:
        s = str(s).replace(',', '').replace('(', '-').replace(')', '')
        return float(s)
    except Exception:
        return np.nan


def fetch_table(supabase: Client, table_name: str, cols: str = '*', limit: int = None):
    # simple fetch; for large datasets, consider server-side filtering
    query = supabase.table(table_name).select(cols)
    if limit:
        query = query.limit(limit)
    res = query.execute()
    if res.error:
        st.error(f"Error fetching {table_name}: {res.error.message}")
        return pd.DataFrame()
    return pd.DataFrame(res.data)


# -----------------------------
# Feature engineering
# -----------------------------

def prepare_historic_df(h_df: pd.DataFrame):
    df = h_df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        st.error('No date column found in historical_prices table')
        return pd.DataFrame()

    for col in ['Open','High','Low','Close','volume','Volume','close','Close']:
        if col in df.columns:
            df.rename(columns={col: col.lower()}, inplace=True)
    # unify numeric cols
    for c in ['open','high','low','close','volume','dividends','stock splits','volume']:
        if c in df.columns:
            df[c] = df[c].apply(safe_numeric)

    df = df.sort_values('date').reset_index(drop=True)
    # compute returns
    df['return_1d'] = df['close'].pct_change()
    df['log_return_1d'] = np.log1p(df['return_1d'])
    # rolling features
    for w in [5, 10, 20, 60]:
        df[f'rolling_mean_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}'] = df['close'].rolling(window=w, min_periods=1).std()
        df[f'rolling_vol_{w}'] = df['log_return_1d'].rolling(window=w, min_periods=1).std()
    # volume zscore
    if 'volume' in df.columns:
        df['volume_med_20'] = df['volume'].rolling(20, min_periods=1).median()
        df['volume_std_20'] = df['volume'].rolling(20, min_periods=1).std()
        df['volume_z'] = (df['volume'] - df['volume_med_20']) / (df['volume_std_20'] + 1e-9)
    # future label (next day return)
    df['future_return_1d'] = df['return_1d'].shift(-1)
    return df


def merge_financials_by_nearest_date(h_df: pd.DataFrame, fin_df: pd.DataFrame, fin_date_col='period_end'):
    # Ensure fin_df has period_end as datetime
    df_fin = fin_df.copy()
    if fin_date_col not in df_fin.columns:
        # try common names
        matches = [c for c in df_fin.columns if 'period' in c or 'date' in c]
        if matches:
            fin_date_col = matches[0]
        else:
            return h_df
    df_fin[fin_date_col] = pd.to_datetime(df_fin[fin_date_col], errors='coerce')
    df_fin = df_fin.sort_values(fin_date_col)
    # merge_asof to attach latest financial period at each historical date
    merged = pd.merge_asof(h_df.sort_values('date'), df_fin.sort_values(fin_date_col), left_on='date', right_on=fin_date_col, direction='backward')
    return merged


# -----------------------------
# Analysis functions
# -----------------------------

def compute_correlations(df: pd.DataFrame, target_col='future_return_1d'):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric.columns:
        return pd.DataFrame()
    corr = numeric.corr(method='pearson')
    # focus on correlations with target
    corr_with_target = corr[[target_col]].sort_values(by=target_col, ascending=False)
    return corr, corr_with_target


def detect_volume_price_events(df: pd.DataFrame, vol_z_thresh=2.0, price_move_thresh=0.02):
    df = df.copy()
    df['volume_spike'] = df['volume_z'].abs() > vol_z_thresh
    df['large_price_move'] = df['return_1d'].abs() > price_move_thresh
    # events where both occur same day
    events = df[df['volume_spike'] & df['large_price_move']]
    return events


# -----------------------------
# Modeling (classification/regression)
# -----------------------------

def build_and_train_model(df: pd.DataFrame, feature_cols, label_col, model_type='classification'):
    df_model = df.dropna(subset=feature_cols + [label_col]).copy()
    if df_model.empty:
        st.warning('No data available to train the model after dropping NaNs.')
        return None
    X = df_model[feature_cols]
    y = df_model[label_col]

    # time-based split - keep order
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42)) if model_type=='classification' else ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    if model_type == 'classification':
        probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        report = classification_report(y_test, preds, output_dict=True)
        auc = roc_auc_score(y_test, probs) if probs is not None else None
        return {'pipeline': pipeline, 'report': report, 'auc': auc, 'X_test': X_test, 'y_test': y_test, 'preds': preds}
    else:
        mse = mean_squared_error(y_test, preds)
        return {'pipeline': pipeline, 'mse': mse, 'X_test': X_test, 'y_test': y_test, 'preds': preds}


# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("Financial correlations, volume deviations & ML Explorer")

supabase = get_supabase()

# Fetch metadata and symbols
with st.spinner('Fetching available symbols...'):
    meta = fetch_table(supabase, 'meta_data')

symbols = []
if not meta.empty and 'symbol' in meta.columns:
    symbols = sorted(meta['symbol'].dropna().unique().tolist())

col1, col2 = st.columns([1,3])
with col1:
    symbol = st.selectbox('Choose symbol', options=symbols if symbols else ['AAPL'])
    start_date = st.date_input('Start date', value=datetime(2019,1,1))
    end_date = st.date_input('End date', value=datetime.today())
    vol_z_thresh = st.slider('Volume z-score threshold for events', 1.0, 5.0, 2.0)
    price_move_thresh = st.slider('Price move threshold (abs) for events', 0.005, 0.1, 0.02)
    label_move_thresh = st.slider('Label: next-day move threshold (abs) to classify', 0.005, 0.2, 0.02)
    model_type = st.selectbox('Model type', ['classification', 'regression'])
    run_analysis = st.button('Run analysis & train model')

with col2:
    st.write('Instructions: choose a symbol and date range, then press Run')

if run_analysis:
    with st.spinner('Fetching data from Supabase...'):
        # fetch history for symbol
        hist = supabase.table('historical_prices').select('*').eq('symbol', symbol).execute().data
        hist_df = pd.DataFrame(hist)
        if hist_df.empty:
            st.error('No historical_prices rows returned for this symbol.')
        else:
            hist_df = prepare_historic_df(hist_df)
            # filter date range
            hist_df = hist_df[(hist_df['date'] >= pd.to_datetime(start_date)) & (hist_df['date'] <= pd.to_datetime(end_date))].reset_index(drop=True)

            # fetch latest income_statement for symbol
            income = supabase.table('income_statement').select('*').eq('symbol', symbol).execute().data
            income_df = pd.DataFrame(income)

            merged = merge_financials_by_nearest_date(hist_df, income_df, fin_date_col='period_end')

            # Correlations
            corr_matrix, corr_with_target = compute_correlations(merged, target_col='future_return_1d')

            st.subheader(f'Price chart for {symbol}')
            fig = px.line(merged, x='date', y='close', title=f'{symbol} Close Price')
            st.plotly_chart(fig, use_container_width=True)

            if 'volume' in merged.columns:
                st.subheader('Volume (with z-score)')
                fig2 = px.bar(merged, x='date', y='volume', title='Volume')
                fig2.add_scatter(x=merged['date'], y=merged['volume_z'], mode='lines', name='volume_z (scaled)')
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader('Top correlations with next-day return')
            st.dataframe(corr_with_target.head(20))

            st.subheader('Correlation heatmap (numeric features)')
            # small heatmap - use pandas
            st.write(corr_matrix.style.background_gradient(cmap='RdBu', axis=None))

            # detect events
            events = detect_volume_price_events(merged, vol_z_thresh=vol_z_thresh, price_move_thresh=price_move_thresh)
            st.subheader('Detected volume+price events')
            st.dataframe(events[['date','close','return_1d','volume','volume_z']].head(50))

            # Prepare model features
            feature_cols = [c for c in merged.columns if c.startswith('rolling_') or c.startswith('rolling_vol_') or c in ['volume','volume_z','return_1d']]
            # drop highly correlated duplicates
            feature_cols = [c for c in feature_cols if c in merged.columns]

            if model_type == 'classification':
                merged['label'] = (merged['future_return_1d'].abs() > label_move_thresh).astype(int)
                label_col = 'label'
            else:
                label_col = 'future_return_1d'

            st.write(f'Using features: {feature_cols[:30]}')

            model_result = build_and_train_model(merged, feature_cols, label_col, model_type=model_type)
            if model_result:
                st.success('Model trained')
                if model_type == 'classification':
                    st.subheader('Classification report')
                    st.text(str(pd.DataFrame(model_result['report']).transpose()))
                    if model_result['auc'] is not None:
                        st.write('AUC:', model_result['auc'])
                else:
                    st.subheader('Regression metrics')
                    st.write('MSE:', model_result['mse'])

                # feature importance
                pipeline = model_result['pipeline']
                # extract RF
                rf = None
                for step in pipeline.steps:
                    if isinstance(step[1], (RandomForestClassifier, RandomForestRegressor)):
                        rf = step[1]
                if rf is not None:
                    importances = rf.feature_importances_
                    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
                    st.subheader('Top feature importances')
                    st.dataframe(fi.head(20))

                # allow download of model
                buffer = io.BytesIO()
                joblib.dump(pipeline, buffer)
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="model_{symbol}.joblib">Download trained model</a>'
                st.markdown(href, unsafe_allow_html=True)


# -----------------------------
# Export helper
# -----------------------------

st.sidebar.markdown('---')
st.sidebar.write('Notes & next steps:')
st.sidebar.write('- This is a starter app. For production, you should:')
st.sidebar.write('  * Add pagination when fetching large tables from Supabase')
st.sidebar.write('  * Cache expensive queries (st.cache_data)')
st.sidebar.write('  * Use more robust label generation and avoid lookahead leakage')
st.sidebar.write('  * Add model monitoring and backtesting')


# End of file
