# src/features/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

LOW_CORR_COLS = [
    'cost_of_living_index',
    'regional_unemployment_rate',
    'housing_price_index',
    'regional_median_income',
    'regional_median_rent',
    'application_id',
    'num_customer_service_calls',
    'marketing_campaign',
    'num_inquiries_6mo',
    'recent_inquiry_count',
    'account_open_year',
    'account_status_code',
    'employment_type',
    'total_debt_amount',
    'education',
    'marital_status',
    'credit_utilization',
    'loan_term',
    'annual_debt_payment',
    'total_monthly_debt_payment',
    'num_delinquencies_2yrs',
    'loan_type',
    'has_mobile_app',
    'previous_zip_code',
    'state',
    'origination_channel',
    'num_dependents',
    'interest_rate',
    'loan_purpose',
    'paperless_billing',
    'employment_length',
    'num_credit_accounts',
    'credit_usage_amount',
    'preferred_contact',
    'referral_code',
    'num_public_records',
    'random_noise_1',
    'application_hour',
    'application_day_of_week',
    'revolving_balance',
    'loan_to_value_ratio',
    'loan_officer_id'
]


class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def remove_low_corr(self, df: pd.DataFrame):
        return df.drop(columns=LOW_CORR_COLS, errors="ignore")

    def scale(self, X_train, X_test):
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def save_scaler(self, path="scaler.pkl"):
        import joblib
        joblib.dump(self.scaler, path)

    def load_scaler(self, path="scaler.pkl"):
        import joblib
        self.scaler = joblib.load(path)
        return self.scaler
    
    def fill_missing_values(self, df: pd.DataFrame):
        if 'employment_length' in df.columns:
            mode_val = df['employment_length'].mode()[0]
            df['employment_length'].fillna(mode_val, inplace=True)
        return df

