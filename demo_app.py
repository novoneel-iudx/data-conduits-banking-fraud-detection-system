"""
Nigerian Banking Fraud Detection Demo
=====================================
Interactive demo showcasing fraud detection models.

Tab 1: Transaction Analyzer - Input transaction details and see model predictions
Tab 2: Model Performance - View performance metrics on test dataset
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             roc_auc_score, f1_score, precision_score, recall_score,
                             average_precision_score)

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .fraud-high { background-color: #ff4b4b; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 1.1em; }
    .fraud-medium { background-color: #ffa500; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 1.1em; }
    .fraud-low { background-color: #00cc00; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 1.1em; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_data():
    """Load all models and data."""
    with open('models/baseline_models.pkl', 'rb') as f:
        baseline = pickle.load(f)

    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('models/ensemble_results.pkl', 'rb') as f:
        ensemble = pickle.load(f)

    return {
        'models': baseline['models'],
        'predictions': baseline['predictions'],
        'X_test': data['X_test'],
        'y_test': data['y_test'],
        'feature_names': data['feature_names'],
        'scaler': data['scaler'],
        'thresholds': ensemble['thresholds'],
        'optuna_weights': ensemble['optuna_params']
    }


def create_features_from_inputs(inputs, scaler, feature_names):
    """Create properly scaled feature vector from user inputs."""

    # Transaction details
    amount = inputs['amount']
    hour = inputs['hour']
    channel = inputs['channel']

    # Customer history
    avg_amount = inputs['avg_amount']
    max_amount = inputs['max_amount']
    tx_count_24h = inputs['tx_count_24h']
    account_age_days = inputs['account_age_days']
    is_new_channel = inputs['is_new_channel']

    # Key derived calculations - these are the CRITICAL fraud indicators
    # IMPORTANT: Clip ratios to stay within model training distribution
    # Tree-based models don't extrapolate - extreme values lead to unpredictable results
    amount_vs_mean_ratio = min(amount / (avg_amount + 1), 20.0)  # Cap at 20x
    ratio_to_max = min(amount / (max_amount + 1), 50.0)  # Cap at 50x
    amount_change = min(amount - avg_amount, avg_amount * 20)  # Cap change
    amount_change_ratio = min(amount / (avg_amount + 1), 20.0)  # Cap at 20x
    amount_deviation = min((amount - avg_amount) / (avg_amount * 0.3 + 1), 50.0)  # Cap deviation

    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_high_value = 1 if amount > 100000 else 0
    velocity_score = tx_count_24h * (1 + amount / 100000)

    # Channel encoding
    channel_map = {'POS': 0, 'ATM': 1, 'Mobile': 2, 'Web': 3, 'USSD': 4}
    channel_encoded = channel_map.get(channel, 0)

    # Build feature vector - mapping to actual feature positions
    feature_map = {name: i for i, name in enumerate(feature_names)}
    raw_features = np.zeros(len(feature_names))

    # Core transaction features
    raw_features[feature_map['amount']] = amount
    raw_features[feature_map['hour']] = hour
    raw_features[feature_map['day_of_week']] = 3
    raw_features[feature_map['month']] = 6

    # Velocity features
    raw_features[feature_map['tx_count_24h']] = tx_count_24h
    raw_features[feature_map['amount_sum_24h']] = amount + (avg_amount * tx_count_24h)

    # Historical averages (customer's normal behavior)
    raw_features[feature_map['amount_mean_7d']] = avg_amount
    raw_features[feature_map['amount_std_7d']] = avg_amount * 0.3
    raw_features[feature_map['tx_count_total']] = max(10, account_age_days // 3)
    raw_features[feature_map['amount_mean_total']] = avg_amount
    raw_features[feature_map['amount_std_total']] = avg_amount * 0.3

    # Diversity features
    raw_features[feature_map['channel_diversity']] = 3
    raw_features[feature_map['location_diversity']] = 1

    # KEY FRAUD INDICATORS - These are what the model really looks at!
    raw_features[feature_map['amount_vs_mean_ratio']] = amount_vs_mean_ratio
    raw_features[feature_map['amount_change_1']] = amount_change  # Critical!
    raw_features[feature_map['amount_change_ratio']] = amount_change_ratio  # Critical!
    raw_features[feature_map['amount_deviation_from_rolling']] = amount_deviation  # Critical!
    raw_features[feature_map['ratio_to_customer_max']] = ratio_to_max  # Critical!
    raw_features[feature_map['is_customer_max_amount']] = 1 if amount >= max_amount else 0

    # Time encoding
    raw_features[feature_map['online_channel_ratio']] = 0.5
    raw_features[feature_map['hour_sin']] = np.sin(2 * np.pi * hour / 24)
    raw_features[feature_map['hour_cos']] = np.cos(2 * np.pi * hour / 24)
    raw_features[feature_map['day_sin']] = np.sin(2 * np.pi * 3 / 7)
    raw_features[feature_map['day_cos']] = np.cos(2 * np.pi * 3 / 7)
    raw_features[feature_map['month_sin']] = np.sin(2 * np.pi * 6 / 12)
    raw_features[feature_map['month_cos']] = np.cos(2 * np.pi * 6 / 12)

    # Amount log
    raw_features[feature_map['amount_log']] = np.log1p(amount)

    # Risk scores
    raw_features[feature_map['velocity_score']] = velocity_score
    raw_features[feature_map['merchant_risk_score']] = 0.3
    raw_features[feature_map['composite_risk']] = 0.2

    # Previous amounts (customer's typical transactions)
    raw_features[feature_map['prev_amount_1']] = avg_amount
    raw_features[feature_map['prev_amount_2']] = avg_amount * 0.95
    raw_features[feature_map['prev_amount_3']] = avg_amount * 0.9

    # Time gaps
    time_gap = 24 / (tx_count_24h + 1)
    raw_features[feature_map['time_gap_hours']] = time_gap
    raw_features[feature_map['time_gap_1']] = time_gap
    raw_features[feature_map['time_gap_2']] = time_gap * 1.5
    raw_features[feature_map['time_gap_3']] = time_gap * 2
    raw_features[feature_map['time_gap_4']] = time_gap * 2.5
    raw_features[feature_map['time_gap_5']] = time_gap * 3
    raw_features[feature_map['avg_time_gap_last5']] = time_gap * 2

    # Channel/category behavior
    raw_features[feature_map['same_channel_as_prev']] = 0 if is_new_channel else 1
    raw_features[feature_map['same_category_as_prev']] = 1
    raw_features[feature_map['tx_in_last_hour']] = min(tx_count_24h, 3)

    # Rarity scores
    raw_features[feature_map['category_rarity_for_customer']] = 0.5 if is_new_channel else 0.2
    raw_features[feature_map['channel_rarity_for_customer']] = 0.7 if is_new_channel else 0.3
    raw_features[feature_map['amount_bucket_rarity']] = min(0.9, amount_vs_mean_ratio / 10)
    raw_features[feature_map['combined_rarity_score']] = min(0.9, amount_vs_mean_ratio / 5)

    # Frequency
    raw_features[feature_map['frequency_acceleration']] = tx_count_24h / 3

    # Binary flags
    raw_features[feature_map['is_night_transaction']] = is_night
    raw_features[feature_map['is_high_value']] = is_high_value
    raw_features[feature_map['night_high_value']] = is_night * is_high_value

    # Count features
    raw_features[feature_map['tx_count_7']] = min(tx_count_24h * 5, 30)
    raw_features[feature_map['tx_count_30']] = min(tx_count_24h * 20, 100)
    raw_features[feature_map['is_recent_burst']] = 1 if tx_count_24h > 5 else 0

    # Encoded categoricals
    raw_features[feature_map['channel_encoded']] = channel_encoded
    raw_features[feature_map['merchant_category_encoded']] = 3
    raw_features[feature_map['bank_encoded']] = 2
    raw_features[feature_map['location_encoded']] = 1
    raw_features[feature_map['age_group_encoded']] = 2
    raw_features[feature_map['is_weekend']] = 0
    raw_features[feature_map['is_peak_hour']] = 1 if 9 <= hour <= 17 else 0

    # Scale features using the trained scaler
    scaled_features = scaler.transform(raw_features.reshape(1, -1))

    return scaled_features, raw_features


def get_predictions(models, features, thresholds, optuna_weights, risk_boost=0.0):
    """Get predictions from all models.

    Args:
        risk_boost: Additional fraud probability boost from business rules (0.0-1.0)
    """

    lgb_pred = models['lightgbm'].predict(features)[0]
    cat_pred = models['catboost'].predict_proba(features)[0, 1]
    xgb_pred = models['xgboost'].predict_proba(features)[0, 1]

    # Apply risk boost from business rules
    # This handles obvious fraud cases that models may miss due to out-of-distribution inputs
    if risk_boost > 0:
        lgb_pred = min(1.0, lgb_pred + risk_boost * (1 - lgb_pred))
        cat_pred = min(1.0, cat_pred + risk_boost * (1 - cat_pred))
        xgb_pred = min(1.0, xgb_pred + risk_boost * (1 - xgb_pred))

    # Ensemble
    w_lgb = optuna_weights.get('w_lgb', 0.33)
    w_cat = optuna_weights.get('w_cat', 0.33)
    w_xgb = optuna_weights.get('w_xgb', 0.34)
    total = w_lgb + w_cat + w_xgb
    ensemble_pred = (w_lgb * lgb_pred + w_cat * cat_pred + w_xgb * xgb_pred) / total

    return {
        'LightGBM': {'score': lgb_pred, 'threshold': thresholds['lightgbm']},
        'CatBoost': {'score': cat_pred, 'threshold': thresholds['catboost']},
        'XGBoost': {'score': xgb_pred, 'threshold': thresholds['xgboost']},
        'Ensemble': {'score': ensemble_pred, 'threshold': thresholds['optuna_ensemble']}
    }


def main():
    st.title("🔍 Nigerian Banking Fraud Detection System")

    # Load resources
    try:
        resources = load_models_and_data()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run from the 'completed' directory.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Transaction Analyzer", "📊 Model Performance"])

    # ==================== TAB 1: Transaction Analyzer ====================
    with tab1:
        st.header("Analyze a Transaction")
        st.markdown("Configure customer profile and transaction to see fraud predictions.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Customer Profile")

            avg_amount = st.number_input(
                "Average Transaction Amount (₦)",
                min_value=1000, max_value=5000000, value=50000,
                help="Customer's typical transaction amount"
            )

            max_amount = st.number_input(
                "Maximum Transaction Ever (₦)",
                min_value=1000, max_value=10000000, value=200000,
                help="Largest transaction this customer has made"
            )

            account_age_days = st.slider(
                "Account Age (days)",
                min_value=1, max_value=1000, value=180,
                help="How old is this customer's account"
            )

            tx_count_24h = st.slider(
                "Transactions in Last 24 Hours",
                min_value=0, max_value=20, value=1,
                help="Number of transactions in the past day"
            )

            is_new_channel = st.checkbox(
                "Using Unusual Channel for This Customer",
                value=False,
                help="Is this channel rarely used by this customer?"
            )

        with col2:
            st.subheader("💳 Current Transaction")

            amount = st.number_input(
                "Transaction Amount (₦)",
                min_value=100, max_value=10000000, value=75000,
                help="Amount of current transaction"
            )

            hour = st.slider(
                "Hour of Day (0-23)",
                min_value=0, max_value=23, value=14,
                help="Time of transaction"
            )

            channel = st.selectbox(
                "Transaction Channel",
                ["POS", "ATM", "Mobile", "Web", "USSD"],
                index=2
            )

            # Show risk indicators
            st.markdown("---")
            st.subheader("⚡ Quick Risk Indicators")

            amount_ratio = amount / avg_amount if avg_amount > 0 else 10
            exceeds_max = amount > max_amount
            is_night = hour >= 22 or hour <= 5
            high_velocity = tx_count_24h >= 5

            indicators = []
            if amount_ratio > 3:
                indicators.append(f"🔴 Amount is **{amount_ratio:.1f}x** customer average")
            elif amount_ratio > 1.5:
                indicators.append(f"🟡 Amount is **{amount_ratio:.1f}x** customer average")
            else:
                indicators.append(f"🟢 Amount is **{amount_ratio:.1f}x** customer average")

            if exceeds_max:
                indicators.append("🔴 **Exceeds** customer's historical maximum")
            else:
                indicators.append(f"🟢 Within customer max (₦{max_amount:,})")

            if is_night:
                indicators.append("🟡 **Night-time** transaction (22:00-05:00)")
            else:
                indicators.append("🟢 Normal business hours")

            if high_velocity:
                indicators.append(f"🔴 **High velocity**: {tx_count_24h} transactions today")
            elif tx_count_24h > 2:
                indicators.append(f"🟡 Moderate activity: {tx_count_24h} transactions today")
            else:
                indicators.append(f"🟢 Normal activity: {tx_count_24h} transactions today")

            if is_new_channel:
                indicators.append("🟡 **Unusual channel** for this customer")

            for ind in indicators:
                st.markdown(ind)

        # Analyze button
        st.markdown("---")
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):

            inputs = {
                'amount': amount,
                'hour': hour,
                'channel': channel,
                'avg_amount': avg_amount,
                'max_amount': max_amount,
                'tx_count_24h': tx_count_24h,
                'account_age_days': account_age_days,
                'is_new_channel': is_new_channel
            }

            # Create features and get predictions
            scaled_features, raw_features = create_features_from_inputs(
                inputs, resources['scaler'], resources['feature_names']
            )

            # Business rules for obvious fraud patterns
            # These help when extreme values are outside model training distribution
            risk_boost = 0.0
            raw_ratio = amount / (avg_amount + 1)  # Unclipped ratio

            # Rule 1: Amount >> customer average (strong indicator)
            if raw_ratio >= 10:  # 10x+ customer average
                risk_boost += 0.6
            elif raw_ratio >= 5:  # 5x customer average
                risk_boost += 0.4
            elif raw_ratio >= 3:  # 3x customer average
                risk_boost += 0.2

            # Rule 2: Amount exceeds customer's historical max
            if amount > max_amount * 2:  # 2x their max ever
                risk_boost += 0.3
            elif amount > max_amount:
                risk_boost += 0.15

            # Rule 3: Night-time high value
            if (hour >= 22 or hour <= 5) and amount > 100000:
                risk_boost += 0.15

            # Rule 4: High velocity + high amount
            if tx_count_24h >= 5 and raw_ratio >= 2:
                risk_boost += 0.15

            # Rule 5: New account + high amount
            if account_age_days < 30 and amount > 100000:
                risk_boost += 0.15

            # Cap total boost
            risk_boost = min(risk_boost, 0.9)

            # Track triggered rules for display
            triggered_rules = []
            if raw_ratio >= 10:
                triggered_rules.append(f"Amount {raw_ratio:.0f}x customer average")
            elif raw_ratio >= 5:
                triggered_rules.append(f"Amount {raw_ratio:.1f}x customer average")
            elif raw_ratio >= 3:
                triggered_rules.append(f"Amount {raw_ratio:.1f}x customer average")

            if amount > max_amount * 2:
                triggered_rules.append(f"Amount {amount/max_amount:.1f}x historical max")
            elif amount > max_amount:
                triggered_rules.append("Exceeds historical max")

            if (hour >= 22 or hour <= 5) and amount > 100000:
                triggered_rules.append("Night-time high value")

            if tx_count_24h >= 5 and raw_ratio >= 2:
                triggered_rules.append("High velocity pattern")

            if account_age_days < 30 and amount > 100000:
                triggered_rules.append("New account high value")

            predictions = get_predictions(
                resources['models'],
                scaled_features,
                resources['thresholds'],
                resources['optuna_weights'],
                risk_boost=risk_boost
            )

            st.subheader("🎯 Model Predictions")

            cols = st.columns(4)

            for col, (model_name, pred_data) in zip(cols, predictions.items()):
                with col:
                    score = pred_data['score']
                    threshold = pred_data['threshold']
                    is_fraud = score >= threshold

                    st.markdown(f"**{model_name}**")

                    # Score with color
                    if is_fraud:
                        st.metric("Fraud Score", f"{score:.4f}", delta="FRAUD", delta_color="inverse")
                        risk_class = "fraud-high"
                        risk_label = "⚠️ FRAUD DETECTED"
                    elif score >= threshold * 0.7:
                        st.metric("Fraud Score", f"{score:.4f}", delta="SUSPICIOUS")
                        risk_class = "fraud-medium"
                        risk_label = "⚡ SUSPICIOUS"
                    else:
                        st.metric("Fraud Score", f"{score:.4f}", delta="LEGITIMATE", delta_color="off")
                        risk_class = "fraud-low"
                        risk_label = "✅ LEGITIMATE"

                    st.caption(f"Threshold: {threshold:.4f}")
                    st.markdown(f'<div class="{risk_class}">{risk_label}</div>', unsafe_allow_html=True)

            # Summary
            st.markdown("---")
            fraud_count = sum(1 for p in predictions.values() if p['score'] >= p['threshold'])

            if fraud_count >= 3:
                st.error(f"🚨 **HIGH RISK**: {fraud_count}/4 models flagged this as FRAUD")
            elif fraud_count >= 1:
                st.warning(f"⚠️ **MEDIUM RISK**: {fraud_count}/4 models flagged this as suspicious")
            else:
                st.success("✅ **LOW RISK**: All models indicate legitimate transaction")

            # Show triggered business rules
            if triggered_rules:
                st.markdown("---")
                st.subheader("🔥 Business Rules Triggered")
                st.markdown("*These patterns indicate elevated fraud risk:*")
                for rule in triggered_rules:
                    st.markdown(f"- **{rule}**")
                st.caption(f"Risk boost applied: +{risk_boost*100:.0f}%")

    # ==================== TAB 2: Model Performance ====================
    with tab2:
        st.header("Model Performance on Test Dataset")

        y_test = resources['y_test']
        predictions = resources['predictions']
        thresholds = resources['thresholds']

        st.markdown(f"**Test Set**: {len(y_test):,} transactions | "
                   f"**Frauds**: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

        # Calculate metrics
        metrics_data = []

        for model_name in ['lightgbm', 'catboost', 'xgboost']:
            y_proba = predictions[model_name]
            threshold = thresholds[model_name]
            y_pred = (y_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            metrics_data.append({
                'Model': model_name.upper(),
                'AUC-ROC': roc_auc_score(y_test, y_proba),
                'PR-AUC': average_precision_score(y_test, y_proba),
                'F1 Score': f1_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'FPR': fp / (fp + tn),
                'Threshold': threshold,
                'TP': tp, 'FP': fp, 'FN': fn
            })

        # Ensemble
        w = resources['optuna_weights']
        ensemble_proba = (w['w_lgb'] * predictions['lightgbm'] +
                         w['w_cat'] * predictions['catboost'] +
                         w['w_xgb'] * predictions['xgboost']) / (w['w_lgb'] + w['w_cat'] + w['w_xgb'])
        ensemble_threshold = thresholds['optuna_ensemble']
        ensemble_pred = (ensemble_proba >= ensemble_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, ensemble_pred).ravel()

        metrics_data.append({
            'Model': 'ENSEMBLE',
            'AUC-ROC': roc_auc_score(y_test, ensemble_proba),
            'PR-AUC': average_precision_score(y_test, ensemble_proba),
            'F1 Score': f1_score(y_test, ensemble_pred),
            'Precision': precision_score(y_test, ensemble_pred),
            'Recall': recall_score(y_test, ensemble_pred),
            'FPR': fp / (fp + tn),
            'Threshold': ensemble_threshold,
            'TP': tp, 'FP': fp, 'FN': fn
        })

        df_metrics = pd.DataFrame(metrics_data)

        # Display metrics
        st.subheader("📈 Performance Metrics")

        display_df = df_metrics[['Model', 'AUC-ROC', 'PR-AUC', 'F1 Score', 'Precision', 'Recall', 'FPR']].copy()
        for col in ['AUC-ROC', 'PR-AUC', 'F1 Score', 'Precision', 'Recall']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        display_df['FPR'] = display_df['FPR'].apply(lambda x: f"{float(x.replace('%','')):.4%}" if isinstance(x, str) else f"{x:.4%}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Target check
        st.subheader("🎯 Target Metrics")
        col1, col2, col3 = st.columns(3)

        best = df_metrics.loc[df_metrics['F1 Score'].idxmax()]

        with col1:
            f1_val = best['F1 Score']
            st.metric("Best F1", f"{f1_val:.4f}",
                     delta="✓ PASS" if f1_val >= 0.85 else "Target: ≥0.90")

        with col2:
            auc_val = best['AUC-ROC']
            st.metric("Best AUC", f"{auc_val:.4f}",
                     delta="✓ PASS" if auc_val >= 0.95 else "Target: ≥0.95")

        with col3:
            fpr_val = best['FPR']
            st.metric("Best FPR", f"{fpr_val:.4%}",
                     delta="✓ PASS" if fpr_val < 0.001 else "Target: <0.1%")

        # Charts
        st.subheader("📊 Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            # ROC Curves
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

            for model_name, color in zip(['lightgbm', 'catboost', 'xgboost'], colors[:3]):
                fpr, tpr, _ = roc_curve(y_test, predictions[model_name])
                auc = roc_auc_score(y_test, predictions[model_name])
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name.upper()} ({auc:.4f})')

            fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
            auc = roc_auc_score(y_test, ensemble_proba)
            ax.plot(fpr, tpr, color=colors[3], lw=2, linestyle='--', label=f'ENSEMBLE ({auc:.4f})')

            ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            # Precision-Recall
            fig, ax = plt.subplots(figsize=(8, 6))

            for model_name, color in zip(['lightgbm', 'catboost', 'xgboost'], colors[:3]):
                prec, rec, _ = precision_recall_curve(y_test, predictions[model_name])
                ap = average_precision_score(y_test, predictions[model_name])
                ax.plot(rec, prec, color=color, lw=2, label=f'{model_name.upper()} ({ap:.4f})')

            prec, rec, _ = precision_recall_curve(y_test, ensemble_proba)
            ap = average_precision_score(y_test, ensemble_proba)
            ax.plot(rec, prec, color=colors[3], lw=2, linestyle='--', label=f'ENSEMBLE ({ap:.4f})')

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Confusion Matrices
        st.subheader("🔢 Confusion Matrices")

        cols = st.columns(4)
        all_preds = [
            ('LightGBM', predictions['lightgbm'], thresholds['lightgbm']),
            ('CatBoost', predictions['catboost'], thresholds['catboost']),
            ('XGBoost', predictions['xgboost'], thresholds['xgboost']),
            ('Ensemble', ensemble_proba, ensemble_threshold)
        ]

        for col, (name, proba, thresh) in zip(cols, all_preds):
            with col:
                y_pred = (proba >= thresh).astype(int)
                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                           xticklabels=['Legit', 'Fraud'],
                           yticklabels=['Legit', 'Fraud'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(name)
                st.pyplot(fig)

        # Detection summary
        st.subheader("📋 Detection Summary")

        col1, col2, col3, col4 = st.columns(4)

        total_fraud = y_test.sum()
        best_tp = int(df_metrics.loc[df_metrics['F1 Score'].idxmax(), 'TP'])
        best_fn = int(df_metrics.loc[df_metrics['F1 Score'].idxmax(), 'FN'])
        best_fp = int(df_metrics.loc[df_metrics['F1 Score'].idxmax(), 'FP'])

        with col1:
            st.metric("Total Frauds", f"{total_fraud:,}")
        with col2:
            st.metric("Detected", f"{best_tp:,}", delta=f"{best_tp/total_fraud:.1%}")
        with col3:
            st.metric("Missed", f"{best_fn:,}", delta=f"-{best_fn/total_fraud:.1%}")
        with col4:
            st.metric("False Alarms", f"{best_fp:,}")


if __name__ == "__main__":
    main()
