#!/usr/bin/env python3
"""
実データからのマクロ経済パラメータ推定（外れ値・健全性強化版 statsmodels使用）
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    print("✓ statsmodels読み込み成功")
except ImportError as e:
    print(f"エラー: statsmodelsが必要です。pip install statsmodels でインストールしてください")
    print(f"詳細: {e}")
    exit(1)

def remove_outliers(df, columns, z=3):
    """指定カラムについてzσ以上の外れ値を除去"""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            m = df_clean[col].mean()
            s = df_clean[col].std()
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] - m).abs() < z * s]
            after = len(df_clean)
            if before != after:
                print(f"  外れ値除去: {col}: {before-after}行除去")
    return df_clean

class RealDataParameterEstimator:
    def __init__(self):
        self.data = None
        self.parameters = {}
        self.estimation_results = {}
        success = self.load_dummy_data()
        if not success:
            raise ValueError("ダミーデータの読み込みに失敗しました")

    def load_dummy_data(self):
        try:
            self.data = pd.read_csv("initial_values_dummy.csv", index_col=0)
            self.data.index = pd.date_range('2020Q1', periods=len(self.data), freq='Q')
            print(f"✓ ダミーデータ読み込み完了: {len(self.data)}行 × {len(self.data.columns)}列")
            print(f"  期間: {self.data.index[0].strftime('%Y-%m')} から {self.data.index[-1].strftime('%Y-%m')}")
            if len(self.data) < 8:
                print(f"警告: データが少なすぎます（{len(self.data)}行）")
                return False
            missing_data = self.data.isnull().sum()
            if missing_data.any():
                print(f"警告: 欠損値が存在します")
                key_vars = ['GDP', 'CP', 'IFP', 'XGS', 'YDV', 'NWCV', 'PGDP', 'PCP', 'RCD', 'UR']
                for var in key_vars:
                    if var in self.data.columns and self.data[var].isnull().any():
                        print(f"  {var}: {self.data[var].isnull().sum()}個の欠損値")
            self.prepare_variables()
            return True
        except Exception as e:
            print(f"✗ データ読み込みエラー: {e}")
            return False

    def prepare_variables(self):
        print("変数変換を開始...")
        log_vars = ['GDP', 'CP', 'IFP', 'IHP', 'XGS', 'MGS', 'YWV', 'YDV', 'NWCV']
        for var in log_vars:
            if var in self.data.columns:
                if (self.data[var] > 0).all():
                    self.data[f'log_{var}'] = np.log(self.data[var])
                    print(f"  ✓ {var} → log_{var}")
                else:
                    print(f"  ✗ {var}: 非正の値があるため対数変換をスキップ")
        diff_vars = ['GDP', 'CP', 'IFP', 'IHP', 'XGS', 'MGS', 'PGDP', 'PCP']
        for var in diff_vars:
            if f'log_{var}' in self.data.columns:
                self.data[f'dlog_{var}'] = self.data[f'log_{var}'].diff()
                print(f"  ✓ dlog_{var}")
            elif var in self.data.columns and (self.data[var] > 0).all():
                self.data[f'dlog_{var}'] = np.log(self.data[var]).diff()
                print(f"  ✓ dlog_{var} (直接)")
        if all(var in self.data.columns for var in ['YDV', 'PCP', 'NWCV']):
            self.data['YDV_real'] = self.data['YDV'] / self.data['PCP']
            self.data['NWCV_real'] = self.data['NWCV'] / self.data['PCP']
            print("  ✓ 実質所得・実質資産")
        lag_vars = ['GDP', 'CP', 'IFP', 'YDV', 'NWCV', 'PGDP', 'RCD', 'RGB', 'UR', 'PSHARE', 'WD_YVI', 'FXS']
        for var in lag_vars:
            if var in self.data.columns:
                self.data[f'{var}_lag1'] = self.data[var].shift(1)
                self.data[f'{var}_lag2'] = self.data[var].shift(2)
        growth_vars = ['PSHARE', 'WD_YVI', 'FXS']
        for var in growth_vars:
            if var in self.data.columns:
                self.data[f'{var}_growth'] = self.data[var].pct_change()
                print(f"  ✓ {var}_growth")
        diff_simple_vars = ['RCD', 'RGB', 'UR']
        for var in diff_simple_vars:
            if var in self.data.columns:
                self.data[f'd_{var}'] = self.data[var].diff()
        print("✓ 変数変換完了\n")
        # 主要成長率変数の分布可視化
        print("主要成長率変数の分布:")
        for var in ['WD_YVI_growth', 'FXS_growth', 'PSHARE_growth']:
            if var in self.data.columns:
                print(f"{var}:")
                print(self.data[var].describe())

    def safe_regression(self, y, X, model_name=""):
        try:
            if len(y) < 5:
                print(f"  ✗ {model_name}: データ不足（{len(y)}行）")
                return None
            if hasattr(y, 'index') and hasattr(X, 'index'):
                common_idx = y.dropna().index.intersection(X.dropna().index)
                y_clean = y.loc[common_idx]
                X_clean = X.loc[common_idx]
            else:
                mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
                y_clean = y[mask]
                X_clean = X[mask]
            if len(y_clean) < 3:
                print(f"  ✗ {model_name}: 有効データ不足（{len(y_clean)}行）")
                return None
            X_with_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_with_const).fit()
            print(f"  ✓ {model_name}: R² = {model.rsquared:.4f}, obs = {len(y_clean)}")
            return model
        except Exception as e:
            print(f"  ✗ {model_name} 回帰エラー: {e}")
            return None

    def estimate_consumption_function(self):
        print("=" * 60)
        print("1. 消費関数の推定")
        print("=" * 60)
        required_vars = ['log_CP', 'log_YDV', 'log_NWCV']
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            print(f"✗ 必要な変数が不足: {missing_vars}")
            return False
        try:
            print("\n【長期均衡関係】")
            consumption_data = self.data[required_vars].dropna()
            print("長期データの分布:")
            print(consumption_data.describe())
            if len(consumption_data) < 5:
                print(f"✗ 有効データ不足: {len(consumption_data)}行")
                return False
            y_long = consumption_data['log_CP']
            X_long = consumption_data[['log_YDV', 'log_NWCV']]
            # 外れ値除去
            consumption_data = remove_outliers(consumption_data, ['log_YDV', 'log_NWCV'], z=3)
            y_long = consumption_data['log_CP']
            X_long = consumption_data[['log_YDV', 'log_NWCV']]
            model_long = self.safe_regression(y_long, X_long, "長期均衡")
            if model_long is None:
                return False
            income_elasticity = model_long.params.get('log_YDV', np.nan)
            wealth_elasticity = model_long.params.get('log_NWCV', np.nan)
            print(f"  所得弾力性: {income_elasticity:.4f}")
            print(f"  資産弾力性: {wealth_elasticity:.4f}")
            print("\n【短期動学】")
            print("dlog_CP欠損数:", self.data['dlog_CP'].isnull().sum() if 'dlog_CP' in self.data.columns else 'N/A')
            print("dlog_YDV欠損数:", self.data['dlog_YDV'].isnull().sum() if 'dlog_YDV' in self.data.columns else 'N/A')
            print("d_RGB欠損数:", self.data['d_RGB'].isnull().sum() if 'd_RGB' in self.data.columns else 'N/A')
            if 'dlog_CP' in self.data.columns:
                ecm_resid = model_long.resid
                short_data = pd.DataFrame({
                    'dlog_CP': self.data['dlog_CP'],
                    'ecm_lag1': ecm_resid.shift(1),
                    'dlog_YDV': self.data.get('dlog_YDV', np.nan),
                    'd_RGB': self.data.get('d_RGB', np.nan)
                }).dropna()
                print("短期データ行数:", len(short_data))
                print("短期データ分布:")
                print(short_data.describe())
                if len(short_data) >= 3:
                    y_short = short_data['dlog_CP']
                    X_vars = ['ecm_lag1']
                    if 'dlog_YDV' in short_data.columns and short_data['dlog_YDV'].notna().sum() > 3:
                        X_vars.append('dlog_YDV')
                    if 'd_RGB' in short_data.columns and short_data['d_RGB'].notna().sum() > 3:
                        X_vars.append('d_RGB')
                    X_short = short_data[X_vars]
                    # 外れ値除去
                    short_data = remove_outliers(short_data, X_vars, z=3)
                    y_short = short_data['dlog_CP']
                    X_short = short_data[X_vars]
                    model_short = self.safe_regression(y_short, X_short, "短期動学")
                    if model_short is not None:
                        adjustment_speed = model_short.params.get('ecm_lag1', np.nan)
                        income_short = model_short.params.get('dlog_YDV', np.nan)
                        interest_effect = model_short.params.get('d_RGB', np.nan)
                        print(f"  調整速度: {adjustment_speed:.4f}")
                        if not np.isnan(income_short):
                            print(f"  短期所得効果: {income_short:.4f}")
                        if not np.isnan(interest_effect):
                            print(f"  金利効果: {interest_effect:.4f}")
                        self.parameters['consumption'] = {
                            'income_elasticity': float(income_elasticity) if not np.isnan(income_elasticity) else None,
                            'wealth_elasticity': float(wealth_elasticity) if not np.isnan(wealth_elasticity) else None,
                            'adjustment_speed': float(adjustment_speed) if not np.isnan(adjustment_speed) else None,
                            'income_short1': float(income_short) if not np.isnan(income_short) else None,
                            'interest_rate': float(interest_effect) if not np.isnan(interest_effect) else None
                        }
                        self.estimation_results['consumption'] = {
                            'sample_size_long': len(consumption_data),
                            'sample_size_short': len(short_data),
                            'r_squared_long': model_long.rsquared,
                            'r_squared_short': model_short.rsquared
                        }
                        print("✓ 消費関数推定完了")
                        return True
            # 長期関係のみの場合
            self.parameters['consumption'] = {
                'income_elasticity': float(income_elasticity) if not np.isnan(income_elasticity) else None,
                'wealth_elasticity': float(wealth_elasticity) if not np.isnan(wealth_elasticity) else None,
                'adjustment_speed': None,
                'income_short1': None,
                'interest_rate': None
            }
            self.estimation_results['consumption'] = {
                'sample_size_long': len(consumption_data),
                'sample_size_short': 0,
                'r_squared_long': model_long.rsquared,
                'r_squared_short': None
            }
            print("✓ 消費関数推定完了（長期のみ）")
            return True
        except Exception as e:
            print(f"✗ 消費関数推定エラー: {e}")
            return False

    def estimate_investment_function(self):
        print("\n" + "=" * 60)
        print("2. 設備投資関数の推定")
        print("=" * 60)
        if 'dlog_IFP' not in self.data.columns:
            print("✗ dlog_IFP（投資成長率）が見つかりません")
            return False
        try:
            explanatory_vars = []
            if 'PSHARE_growth' in self.data.columns:
                explanatory_vars.append('PSHARE_growth')
            elif 'PSHARE' in self.data.columns:
                self.data['PSHARE_growth_4q'] = self.data['PSHARE'].pct_change(4)
                explanatory_vars.append('PSHARE_growth_4q')
            if 'd_RCD' in self.data.columns:
                explanatory_vars.append('d_RCD')
            if not explanatory_vars:
                print("✗ 有効な説明変数が見つかりません")
                return False
            model_vars = ['dlog_IFP'] + explanatory_vars
            investment_data = self.data[model_vars].dropna()
            print("投資関数データ分布(外れ値除去前):")
            print(investment_data.describe())
            investment_data = remove_outliers(investment_data, explanatory_vars, z=3)
            print("投資関数データ分布(外れ値除去後):")
            print(investment_data.describe())
            if len(investment_data) < 3:
                print(f"✗ 有効データ不足: {len(investment_data)}行")
                return False
            y = investment_data['dlog_IFP']
            X = investment_data[explanatory_vars]
            model = self.safe_regression(y, X, "設備投資")
            if model is None:
                return False
            q_elasticity = model.params.get('PSHARE_growth', model.params.get('PSHARE_growth_4q', np.nan))
            interest_rate = model.params.get('d_RCD', np.nan)
            print(f"  トービンのq効果: {q_elasticity:.4f}" if not np.isnan(q_elasticity) else "  トービンのq効果: 推定不可")
            print(f"  金利効果: {interest_rate:.4f}" if not np.isnan(interest_rate) else "  金利効果: 推定不可")
            self.parameters['investment'] = {
                'q_elasticity': float(q_elasticity) if not np.isnan(q_elasticity) else None,
                'interest_rate': float(interest_rate) if not np.isnan(interest_rate) else None,
                'interest_rate_lag': None
            }
            self.estimation_results['investment'] = {
                'sample_size': len(investment_data),
                'r_squared': model.rsquared,
                'explanatory_vars': explanatory_vars
            }
            print("✓ 設備投資関数推定完了")
            return True
        except Exception as e:
            print(f"✗ 設備投資関数推定エラー: {e}")
            return False

    def estimate_export_function(self):
        print("\n" + "=" * 60)
        print("3. 輸出関数の推定")
        print("=" * 60)
        if 'dlog_XGS' not in self.data.columns:
            print("✗ dlog_XGS（輸出成長率）が見つかりません")
            return False
        try:
            explanatory_vars = []
            if 'WD_YVI_growth' in self.data.columns:
                explanatory_vars.append('WD_YVI_growth')
            if 'FXS_growth' in self.data.columns:
                explanatory_vars.append('FXS_growth')
            if not explanatory_vars:
                print("✗ 有効な説明変数が見つかりません")
                return False
            model_vars = ['dlog_XGS'] + explanatory_vars
            export_data = self.data[model_vars].dropna()
            print("輸出関数データ分布(外れ値除去前):")
            print(export_data.describe())
            export_data = remove_outliers(export_data, explanatory_vars, z=3)
            print("輸出関数データ分布(外れ値除去後):")
            print(export_data.describe())
            if len(export_data) < 3:
                print(f"✗ 有効データ不足: {len(export_data)}行")
                return False
            y = export_data['dlog_XGS']
            X = export_data[explanatory_vars]
            model = self.safe_regression(y, X, "輸出")
            if model is None:
                return False
            world_demand_elasticity = model.params.get('WD_YVI_growth', np.nan)
            exchange_rate_elasticity = model.params.get('FXS_growth', np.nan)
            print(f"  世界需要弾力性: {world_demand_elasticity:.4f}" if not np.isnan(world_demand_elasticity) else "  世界需要弾力性: 推定不可")
            print(f"  為替レート弾力性: {exchange_rate_elasticity:.4f}" if not np.isnan(exchange_rate_elasticity) else "  為替レート弾力性: 推定不可")
            self.parameters['export'] = {
                'world_demand_elasticity': float(world_demand_elasticity) if not np.isnan(world_demand_elasticity) else None,
                'exchange_rate_elasticity': float(exchange_rate_elasticity) if not np.isnan(exchange_rate_elasticity) else None
            }
            self.estimation_results['export'] = {
                'sample_size': len(export_data),
                'r_squared': model.rsquared
            }
            print("✓ 輸出関数推定完了")
            return True
        except Exception as e:
            print(f"✗ 輸出関数推定エラー: {e}")
            return False

    # 以下、他の推定関数（フィリップス曲線・テイラールール・労働市場）は必要に応じて同様に追加

    # ...（省略。元コードのままでも大きな問題はありませんが、分布printや外れ値除去を同様に追加可）...

    # estimate_all, print_estimated_parameters, get_model_diagnostics, save_parameters等は元のままでOK

# ...（残りのmainなどは元コードのままでOK）...
