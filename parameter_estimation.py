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
    
    def estimate_phillips_curve(self):
        """フィリップス曲線のパラメータ推定"""
        print("\n" + "=" * 60)
        print("4. フィリップス曲線の推定")
        print("=" * 60)
        
        if 'dlog_PGDP' not in self.data.columns:
            print("✗ dlog_PGDP（インフレ率）が見つかりません")
            return False
        
        try:
            # 説明変数の準備
            explanatory_vars = []
            
            # ラグ付きインフレ率（慣性項）
            if 'dlog_PGDP' in self.data.columns:
                self.data['dlog_PGDP_lag1'] = self.data['dlog_PGDP'].shift(1)
                explanatory_vars.append('dlog_PGDP_lag1')
            
            # GDPギャップ
            if 'GDPGAP' in self.data.columns:
                explanatory_vars.append('GDPGAP')
            
            if not explanatory_vars:
                print("✗ 有効な説明変数が見つかりません")
                return False
            
            # データの準備
            model_vars = ['dlog_PGDP'] + explanatory_vars
            phillips_data = self.data[model_vars].dropna()
            
            if len(phillips_data) < 3:
                print(f"✗ 有効データ不足: {len(phillips_data)}行")
                return False
            
            print(f"説明変数: {explanatory_vars}")
            
            y = phillips_data['dlog_PGDP']
            X = phillips_data[explanatory_vars]
            
            model = self.safe_regression(y, X, "フィリップス曲線")
            if model is None:
                return False
            
            # パラメータ取得
            inertia = model.params.get('dlog_PGDP_lag1', np.nan)
            gap_sensitivity = model.params.get('GDPGAP', np.nan)
            
            print(f"  慣性項: {inertia:.4f}" if not np.isnan(inertia) else "  慣性項: 推定不可")
            print(f"  GDPギャップ感応度: {gap_sensitivity:.4f}" if not np.isnan(gap_sensitivity) else "  GDPギャップ感応度: 推定不可")
            
            self.parameters['phillips_curve'] = {
                'inertia': float(inertia) if not np.isnan(inertia) else None,
                'gap_sensitivity': float(gap_sensitivity) if not np.isnan(gap_sensitivity) else None,
                'money_sensitivity': None
            }
            
            self.estimation_results['phillips_curve'] = {
                'sample_size': len(phillips_data),
                'r_squared': model.rsquared
            }
            
            print("✓ フィリップス曲線推定完了")
            return True
            
        except Exception as e:
            print(f"✗ フィリップス曲線推定エラー: {e}")
            return False
    
    def estimate_taylor_rule(self):
        """テイラールールのパラメータ推定"""
        print("\n" + "=" * 60)
        print("5. テイラールールの推定")
        print("=" * 60)
        
        if 'RCD' not in self.data.columns:
            print("✗ RCD（政策金利）が見つかりません")
            return False
        
        try:
            # インフレ率の計算
            target_inflation = 2.0
            
            if 'inflation_rate' not in self.data.columns:
                if 'PGDP' in self.data.columns:
                    self.data['inflation_rate'] = self.data['PGDP'].pct_change(4) * 100
                else:
                    print("✗ インフレ率データが見つかりません")
                    return False
            
            # 説明変数の準備
            explanatory_vars = []
            
            # 金利の慣性項
            self.data['RCD_lag1'] = self.data['RCD'].shift(1)
            explanatory_vars.append('RCD_lag1')
            
            # インフレギャップ
            self.data['inflation_gap'] = self.data['inflation_rate'] - target_inflation
            explanatory_vars.append('inflation_gap')
            
            # GDPギャップ
            if 'GDPGAP' in self.data.columns:
                explanatory_vars.append('GDPGAP')
            
            # ゼロ金利制約を考慮（正の金利期間のみ）
            valid_periods = self.data['RCD'] > 0.001
            
            model_vars = ['RCD'] + explanatory_vars
            taylor_data = self.data.loc[valid_periods, model_vars].dropna()
            
            if len(taylor_data) < 3:
                print(f"✗ 有効データ不足: {len(taylor_data)}行（ゼロ金利期間除く）")
                return False
            
            print(f"説明変数: {explanatory_vars}")
            print(f"使用期間: {len(taylor_data)}行（ゼロ金利期間除く）")
            
            y = taylor_data['RCD']
            X = taylor_data[explanatory_vars]
            
            model = self.safe_regression(y, X, "テイラールール")
            if model is None:
                return False
            
            # パラメータの解釈
            smoothing = model.params.get('RCD_lag1', 0)
            raw_inflation_coef = model.params.get('inflation_gap', 0)
            raw_gap_coef = model.params.get('GDPGAP', 0)
            
            # 長期係数に変換
            if smoothing < 0.99 and smoothing != 0:
                inflation_coefficient = raw_inflation_coef / (1 - smoothing)
                gap_coefficient = raw_gap_coef / (1 - smoothing)
            else:
                inflation_coefficient = raw_inflation_coef
                gap_coefficient = raw_gap_coef
            
            print(f"  金利慣性: {smoothing:.4f}")
            print(f"  インフレ反応係数: {inflation_coefficient:.4f}")
            print(f"  産出ギャップ反応係数: {gap_coefficient:.4f}")
            
            self.parameters['taylor_rule'] = {
                'smoothing': float(smoothing) if not np.isnan(smoothing) else None,
                'inflation_coefficient': float(inflation_coefficient) if not np.isnan(inflation_coefficient) else None,
                'gap_coefficient': float(gap_coefficient) if not np.isnan(gap_coefficient) else None
            }
            
            self.estimation_results['taylor_rule'] = {
                'sample_size': len(taylor_data),
                'r_squared': model.rsquared
            }
            
            print("✓ テイラールール推定完了")
            return True
            
        except Exception as e:
            print(f"✗ テイラールール推定エラー: {e}")
            return False
    
    def estimate_labor_market(self):
        """労働市場のパラメータ推定"""
        print("\n" + "=" * 60)
        print("6. 労働市場の推定")
        print("=" * 60)
        
        if 'UR' not in self.data.columns:
            print("✗ UR（失業率）が見つかりません")
            return False
        
        try:
            # 失業率の差分
            self.data['d_UR'] = self.data['UR'].diff()
            
            # 説明変数の準備
            explanatory_vars = []
            
            # 稼働率の変化
            if 'CUX' in self.data.columns:
                self.data['dlog_CUX'] = self.data['CUX'].pct_change()
                explanatory_vars.append('dlog_CUX')
            
            # GDPギャップの影響
            if 'GDPGAP' in self.data.columns:
                explanatory_vars.append('GDPGAP')
            
            if not explanatory_vars:
                print("✗ 有効な説明変数が見つかりません")
                return False
            
            model_vars = ['d_UR'] + explanatory_vars
            labor_data = self.data[model_vars].dropna()
            
            if len(labor_data) < 3:
                print(f"✗ 有効データ不足: {len(labor_data)}行")
                return False
            
            print(f"説明変数: {explanatory_vars}")
            
            y = labor_data['d_UR']
            X = labor_data[explanatory_vars]
            
            model = self.safe_regression(y, X, "労働市場")
            if model is None:
                return False
            
            # パラメータ取得
            capacity_sensitivity = model.params.get('dlog_CUX', np.nan)
            gdp_effect = model.params.get('GDPGAP', np.nan)
            
            print(f"  稼働率感応度: {capacity_sensitivity:.4f}" if not np.isnan(capacity_sensitivity) else "  稼働率感応度: 推定不可")
            print(f"  GDPギャップ効果: {gdp_effect:.4f}" if not np.isnan(gdp_effect) else "  GDPギャップ効果: 推定不可")
            
            self.parameters['labor_market'] = {
                'capacity_sensitivity': float(capacity_sensitivity) if not np.isnan(capacity_sensitivity) else None,
                'adjustment_speed': None,
                'gdp_gap_effect': float(gdp_effect) if not np.isnan(gdp_effect) else None
            }
            
            self.estimation_results['labor_market'] = {
                'sample_size': len(labor_data),
                'r_squared': model.rsquared
            }
            
            print("✓ 労働市場推定完了")
            return True
            
        except Exception as e:
            print(f"✗ 労働市場推定エラー: {e}")
            return False
    
    def estimate_all(self):
        """全てのパラメータを推定"""
        print("=" * 80)
        print("ダミーデータからのマクロ経済パラメータ推定（statsmodels使用）")
        print("=" * 80)
        
        if self.data is None:
            print("✗ データが読み込まれていません")
            return
        
        # 推定の実行
        results = {}
        results['consumption'] = self.estimate_consumption_function()
        results['investment'] = self.estimate_investment_function()
        results['export'] = self.estimate_export_function()
        results['phillips_curve'] = self.estimate_phillips_curve()
        results['taylor_rule'] = self.estimate_taylor_rule()
        results['labor_market'] = self.estimate_labor_market()
        
        # 推定結果のサマリー
        print("\n" + "=" * 80)
        print("推定結果サマリー")
        print("=" * 80)
        
        successful_estimations = sum(results.values())
        total_estimations = len(results)
        
        print(f"推定成功: {successful_estimations}/{total_estimations} 関数")
        
        for function_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            print(f"  {function_name}: {status}")
        
        if successful_estimations > 0:
            print(f"\n推定されたパラメータ:")
            self.print_estimated_parameters()
        else:
            print("\n✗ 推定に成功した関数がありません")
    
    def print_estimated_parameters(self):
        """推定されたパラメータのみを表示"""
        print("\n" + "=" * 60)
        print("推定パラメータ（ダミーデータベース）")
        print("=" * 60)
        
        for category, params in self.parameters.items():
            if params and any(v is not None for v in params.values()):
                print(f"\n{category.upper()}:")
                for param_name, value in params.items():
                    if value is not None:
                        print(f"  {param_name}: {value:.4f}")
                    else:
                        print(f"  {param_name}: 推定不可")
    
    def get_model_diagnostics(self):
        """推定されたモデルの診断情報を取得"""
        diagnostics = {}
        
        for function_name, result in self.estimation_results.items():
            if result:
                r_squared = result.get('r_squared', result.get('r_squared_long', 0))
                sample_size = result.get('sample_size', result.get('sample_size_long', 0))
                
                diagnostics[function_name] = {
                    'sample_size': sample_size,
                    'r_squared': r_squared,
                    'successful': True
                }
            else:
                diagnostics[function_name] = {
                    'sample_size': 0,
                    'r_squared': 0,
                    'successful': False
                }
        
        return diagnostics
    
    def save_parameters(self, filename='estimated_parameters_from_dummy_statsmodels.json'):
        """推定されたパラメータを保存"""
        # Noneを除去したクリーンなパラメータ
        clean_parameters = {}
        
        for category, params in self.parameters.items():
            clean_params = {}
            for key, value in params.items():
                if value is not None:
                    clean_params[key] = float(value)
            
            if clean_params:
                clean_parameters[category] = clean_params
        
        # 推定の詳細情報も追加
        metadata = {
            'estimation_method': 'dummy_data_statsmodels_ols',
            'data_period': f"{self.data.index[0].strftime('%Y-%m')} to {self.data.index[-1].strftime('%Y-%m')}",
            'total_observations': len(self.data),
            'diagnostics': self.get_model_diagnostics(),
            'statsmodels_version': 'available'
        }
        
        output_data = {
            'parameters': clean_parameters,
            'metadata': metadata
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n✓ 推定パラメータを {filename} に保存しました")
            return clean_parameters
            
        except Exception as e:
            print(f"\n✗ パラメータ保存エラー: {e}")
            return clean_parameters


def test_basic_functionality():
    """基本機能のテスト"""
    print("=" * 80)
    print("基本機能テスト")
    print("=" * 80)
    
    try:
        # 基本的なデータフレーム操作
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10]
        })
        
        # 対数変換
        test_data['log_A'] = np.log(test_data['A'])
        test_data['log_B'] = np.log(test_data['B'])
        
        # 簡単な回帰
        y = test_data['log_B']
        X = test_data[['log_A']]
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_with_const).fit()
        
        print(f"✓ 基本回帰テスト成功: R² = {model.rsquared:.4f}")
        print(f"  係数: {model.params.get('log_A', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本機能テストエラー: {e}")
        return False


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("実データからのマクロ経済パラメータ推定（最終版）")
    print("=" * 80)
    print("statsmodels使用、実際にテスト済み")
    print()
    
    # 基本機能テスト
    if not test_basic_functionality():
        print("✗ 基本機能テストに失敗しました")
        return None, None
    
    try:
        # パラメータ推定器のインスタンス化
        print("\n" + "=" * 80)
        print("メイン推定開始")
        print("=" * 80)
        
        estimator = RealDataParameterEstimator()
        
        # 全パラメータの推定
        estimator.estimate_all()
        
        # パラメータの保存
        saved_params = estimator.save_parameters()
        
        print("\n" + "=" * 80)
        print("推定完了")
        print("=" * 80)
        
        # 最終的な推定パラメータの表示
        if saved_params:
            print("\n最終推定パラメータ:")
            for category, params in saved_params.items():
                print(f"\n{category}:")
                for key, value in params.items():
                    print(f"  {key}: {value:.4f}")
            
            # 診断情報
            diagnostics = estimator.get_model_diagnostics()
            print(f"\n診断情報:")
            for func_name, diag in diagnostics.items():
                if diag['successful']:
                    print(f"  {func_name}: R² = {diag['r_squared']:.3f}, obs = {diag['sample_size']}")
                else:
                    print(f"  {func_name}: 推定失敗")
        else:
            print("\n推定されたパラメータがありません")
        
        return estimator, saved_params
        
    except Exception as e:
        print(f"\n✗ 実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    estimator, parameters = main()
    
    if estimator is not None and parameters:
        print(f"\n✅ プログラム正常終了")
        print(f"   推定成功関数数: {len(parameters)}")
        print(f"   出力ファイル: estimated_parameters_from_dummy_statsmodels.json")
    else:
        print(f"\n❌ プログラム異常終了")


# テスト実行用の関数
def run_quick_test():
    """クイックテスト実行"""
    print("クイックテスト開始...")
    
    try:
        # データ読み込みテスト
        estimator = RealDataParameterEstimator()
        print(f"データ形状: {estimator.data.shape}")
        
        # 1つの関数だけテスト
        result = estimator.estimate_consumption_function()
        
        if result:
            print("✅ クイックテスト成功")
        else:
            print("❌ クイックテスト失敗")
            
    except Exception as e:
        print(f"❌ クイックテストエラー: {e}")


# デバッグ用
def debug_data_info():
    """データの詳細情報を表示"""
    try:
        estimator = RealDataParameterEstimator()
        
        print("データ情報:")
        print(f"  形状: {estimator.data.shape}")
        print(f"  列数: {len(estimator.data.columns)}")
        print(f"  期間: {estimator.data.index[0]} - {estimator.data.index[-1]}")
        
        # 主要変数の存在確認
        key_vars = ['GDP', 'CP', 'IFP', 'XGS', 'YDV', 'NWCV', 'PGDP', 'RCD', 'UR', 'GDPGAP']
        print(f"\n主要変数:")
        for var in key_vars:
            exists = var in estimator.data.columns
            if exists:
                missing = estimator.data[var].isnull().sum()
                print(f"  {var}: ✓ (欠損: {missing})")
            else:
                print(f"  {var}: ✗")
        
        # 変換後変数の確認
        log_vars = [col for col in estimator.data.columns if col.startswith('log_')]
        print(f"\n対数変換変数 ({len(log_vars)}個): {log_vars[:5]}...")
        
        dlog_vars = [col for col in estimator.data.columns if col.startswith('dlog_')]
        print(f"対数差分変数 ({len(dlog_vars)}個): {dlog_vars[:5]}...")
        
    except Exception as e:
        print(f"デバッグ情報取得エラー: {e}")


# 実行可能なテストコマンド
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_quick_test()
        elif sys.argv[1] == "debug":
            debug_data_info()
        else:
            main()
    else:
        main()
