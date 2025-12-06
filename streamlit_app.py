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
    # using [] so we fail loudly if they are missing
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
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


def _extract_data_from_response(res):
    """
    Handle different supabase-py response types (dict, object with .data, pydantic model).
    """
    # supabase-py v2 often returns a pydantic model with .data attribute
    data = getattr(res, "data", None)

    if data is not None:
        return data

    # sometimes res might be a dict
    if isinstance(res, dict):
        return res.get("data", res)

    # fall back: try model_dump (pydantic)
    try:
        dumped = res.model_dump()
        return dumped.get("data", [])
    except Exception:
        return []

def execute_query(query, table_name: str = ""):
    """
    Wrapper around query.execute() that handles exceptions and extracts data safely.
    """
    try:
        res = query.execute()
    except Exception as e:
        if table_name:
            st.error(f"Error fetching {table_name}: {e}")
        else:
            st.error(f"Error executing Supabase query: {e}")
        return []
    return _extract_data_from_response(res)


def fetch_table(supabase: Client, table_name: str, cols: str = '*', limit: int = None) -> pd.DataFrame:
    # simple fetch; for large datasets, consider server-side filtering
    query = supabase.table(table_name).select(cols)
    if limit:
        query = query.limit(limit)
    data = execute_query(query, table_name=table_name)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


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

    # normalize price & volume column names
    for col in ['Open', 'High', 'Low', 'Close', 'volume', 'Volume', 'close', 'Close']:
        if col in df.columns:
            df.rename(columns={col: col.lower()}, inplace=True)

    # unify numeric cols
    for c in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits', 'volume']:
        if c in df.columns:
            df[c] = df[c].apply(safe_numeric)

    df = df.sort_values('date').reset_index(drop=True)

    # compute returns
    if 'close' not in df.columns:
        st.error("No 'close' column found after normalization.")
        return pd.DataFrame()

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
    if df_fin.empty:
        return h_df

    if fin_date_col not in df_fin.columns:
        # try common names
        matches = [c for c in df_fin.columns if 'period' in c.lower() or 'date' in c.lower()]
        if matches:
            fin_date_col = matches[0]
        else:
            return h_df

    df_fin[fin_date_col] = pd.to_datetime(df_fin[fin_date_col], errors='coerce')
    df_fin = df_fin.sort_values(fin_date_col)

    if df_fin[fin_date_col].isna().all():
        # nothing valid to merge
        return h_df

    # merge_asof to attach latest financial period at each historical date
    merged = pd.merge_asof(
        h_df.sort_values('date'),
        df_fin.sort_values(fin_date_col),
        left_on='date',
        right_on=fin_date_col,
        direction='backward'
    )
    return merged


# -----------------------------
# Analysis functions
# -----------------------------

def compute_correlations(df: pd.DataFrame, target_col='future_return_1d'):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric.columns:
        return pd.DataFrame(), pd.DataFrame()
    corr = numeric.corr(method='pearson')
    # focus on correlations with target
    corr_with_target = corr[[target_col]].sort_values(by=target_col, ascending=False)
    return corr, corr_with_target


def detect_volume_price_events(df: pd.DataFrame, vol_z_thresh=2.0, price_move_thresh=0.02):
    df = df.copy()
    if 'volume_z' not in df.columns:
        df['volume_spike'] = False
    else:
        df['volume_spike'] = df['volume_z'].abs() > vol_z_thresh

    if 'return_1d' not in df.columns:
        df['large_price_move'] = False
    else:
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
    if split_idx == 0 or split_idx == len(df_model):
        st.warning("Not enough data to create a train/test split.")
        return None

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # model step
    if model_type == 'classification':
        final_step = ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    else:
        final_step = ('reg', RandomForestRegressor(n_estimators=100, random_state=42))

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        final_step
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    if model_type == 'classification':
        # Pipeline proxies predict_proba to the final estimator if available
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

col1, col2 = st.columns([1, 3])
with col1:
    symbol = st.selectbox('Choose symbol', options=symbols if symbols else ['AAPL'])
    start_date = st.date_input('Start date', value=datetime(2019, 1, 1))
    end_date = st.date_input('End date', value=datetime.today())
    vol_z_thresh = st.slider('Volume z-score threshold for events', 1.0, 5.0, 2.0)
    price_move_thresh = st.slider('Price move threshold (abs) for events', 0.005, 0.1, 0.02)
    label_move_thresh = st.slider('Label: next-day move threshold (abs) to classify', 0.005, 0.2, 0.02)
    model_type = st.selectbox('Model type', ['classification', 'regression'])
    run_analysis = st.button('Run analysis & train model')

with col2:
    st.write('Instructions: choose a symbol and date range, then press **Run analysis & train model**.')

if run_analysis:
    with st.spinner('Fetching data from Supabase...'):
        # fetch history for symbol
        hist_query = supabase.table('historical_prices').select('*').eq('symbol', symbol)
        hist_data = execute_query(hist_query, table_name='historical_prices')
        hist_df = pd.DataFrame(hist_data)

        if hist_df.empty:
            st.error('No historical_prices rows returned for this symbol.')
        else:
            hist_df = prepare_historic_df(hist_df)
            if hist_df.empty or 'date' not in hist_df.columns:
                st.error("Historical data could not be prepared correctly (missing 'date' or 'close').")
            else:
                # filter date range
                hist_df = hist_df[
                    (hist_df['date'] >= pd.to_datetime(start_date)) &
                    (hist_df['date'] <= pd.to_datetime(end_date))
                ].reset_index(drop=True)

                if hist_df.empty:
                    st.error("No data in the selected date range.")
                else:
                    # fetch latest income_statement for symbol
                    income_query = supabase.table('income_statement').select('*').eq('symbol', symbol)
                    income_data = execute_query(income_query, table_name='income_statement')
                    income_df = pd.DataFrame(income_data)

                    merged = merge_financials_by_nearest_date(hist_df, income_df, fin_date_col='period_end')

                    # Correlations
                    corr_matrix, corr_with_target = compute_correlations(merged, target_col='future_return_1d')

                    st.subheader(f'Price chart for {symbol}')
                    fig = px.line(merged, x='date', y='close', title=f'{symbol} Close Price')
                    st.plotly_chart(fig, use_container_width=True)

                    if 'volume' in merged.columns:
                        st.subheader('Volume (with z-score)')
                        fig2 = px.bar(merged, x='date', y='volume', title='Volume')
                        if 'volume_z' in merged.columns:
                            fig2.add_scatter(
                                x=merged['date'],
                                y=merged['volume_z'],
                                mode='lines',
                                name='volume_z (scaled)'
                            )
                        st.plotly_chart(fig2, use_container_width=True)

                    if not corr_with_target.empty:
                        st.subheader('Top correlations with next-day return')
                        st.dataframe(corr_with_target.head(20))

                    if not corr_matrix.empty:
                        st.subheader('Correlation heatmap (numeric features)')
                        # small heatmap - use pandas styling
                        st.write(corr_matrix.style.background_gradient(cmap='RdBu', axis=None))

                    # detect events
                    events = detect_volume_price_events(
                        merged,
                        vol_z_thresh=vol_z_thresh,
                        price_move_thresh=price_move_thresh
                    )
                    st.subheader('Detected volume+price events')
                    if not events.empty:
                        cols_to_show = [c for c in ['date', 'close', 'return_1d', 'volume', 'volume_z'] if c in events.columns]
                        st.dataframe(events[cols_to_show].head(50))
                    else:
                        st.write("No events matching the selected thresholds.")

                    # Prepare model features
                    base_feature_cols = [c for c in merged.columns
                                         if c.startswith('rolling_')
                                         or c.startswith('rolling_vol_')
                                         or c in ['volume', 'volume_z', 'return_1d']]
                    # ensure they exist
                    feature_cols = [c for c in base_feature_cols if c in merged.columns]

                    if not feature_cols:
                        st.warning("No feature columns available for modeling.")
                    else:
                        if model_type == 'classification':
                            if 'future_return_1d' not in merged.columns:
                                st.error("Missing 'future_return_1d' for classification label.")
                            else:
                                merged['label'] = (merged['future_return_1d'].abs() > label_move_thresh).astype(int)
                                label_col = 'label'
                        else:
                            label_col = 'future_return_1d'

                        if label_col not in merged.columns:
                            st.error(f"Label column '{label_col}' not available for modeling.")
                        else:
                            st.write(f'Using features (first 30 shown): {feature_cols[:30]}')

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
                                rf = None
                                for _, step in pipeline.steps:
                                    if isinstance(step, (RandomForestClassifier, RandomForestRegressor)):
                                        rf = step
                                        break

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
# Export helper / Sidebar
# -----------------------------

st.sidebar.markdown('---')
st.sidebar.write('Notes & next steps:')
st.sidebar.write('- This is a starter app. For production, you should:')
st.sidebar.write('  * Add pagination when fetching large tables from Supabase')
st.sidebar.write('  * Cache expensive queries (st.cache_data)')
st.sidebar.write('  * Use more robust label generation and avoid lookahead leakage')
st.sidebar.write('  * Add model monitoring and backtesting')

# End of file
