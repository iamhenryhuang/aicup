import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def LoadCSV(dir_path):
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    Args:
        dir_path (str): 資料夾，請把上述3個檔案放在同一個資料夾
    
    Returns:
        df_txn: 交易資料 DataFrame
        df_alert: 警示帳戶註記 DataFrame
        df_test: 待預測帳戶清單 DataFrame
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    
    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test


def PreProcessing(df, df_alert=None):
    """
    改進的資料處理，加入圖網絡特徵和更多進階特徵
    """    
    print("開始特徵工程...")
    
    # === 基本金額統計特徵 ===
    send = df.groupby('from_acct')['txn_amt'].sum().rename('total_send_amt')
    recv = df.groupby('to_acct')['txn_amt'].sum().rename('total_recv_amt')
    
    max_send = df.groupby('from_acct')['txn_amt'].max().rename('max_send_amt')
    min_send = df.groupby('from_acct')['txn_amt'].min().rename('min_send_amt')
    avg_send = df.groupby('from_acct')['txn_amt'].mean().rename('avg_send_amt')
    std_send = df.groupby('from_acct')['txn_amt'].std().rename('std_send_amt')
    median_send = df.groupby('from_acct')['txn_amt'].median().rename('median_send_amt')
    
    max_recv = df.groupby('to_acct')['txn_amt'].max().rename('max_recv_amt')
    min_recv = df.groupby('to_acct')['txn_amt'].min().rename('min_recv_amt')
    avg_recv = df.groupby('to_acct')['txn_amt'].mean().rename('avg_recv_amt')
    std_recv = df.groupby('to_acct')['txn_amt'].std().rename('std_recv_amt')
    median_recv = df.groupby('to_acct')['txn_amt'].median().rename('median_recv_amt')
    
    # === 交易次數特徵 ===
    send_count = df.groupby('from_acct').size().rename('send_count')
    recv_count = df.groupby('to_acct').size().rename('recv_count')
    
    # === 交易對象數量特徵 ===
    unique_to = df.groupby('from_acct')['to_acct'].nunique().rename('unique_to_accts')
    unique_from = df.groupby('to_acct')['from_acct'].nunique().rename('unique_from_accts')
    
    # === 特殊交易特徵 ===
    self_txn_count = df[df['is_self_txn']=='Y'].groupby('from_acct').size().rename('self_txn_count')
    self_txn_amt = df[df['is_self_txn']=='Y'].groupby('from_acct')['txn_amt'].sum().rename('self_txn_amt')
    
    # === 通道類型特徵 ===
    channel_diversity_send = df.groupby('from_acct')['channel_type'].nunique().rename('channel_diversity_send')
    channel_diversity_recv = df.groupby('to_acct')['channel_type'].nunique().rename('channel_diversity_recv')
    
    # 各通道的交易次數
    for channel in df['channel_type'].unique():
        if pd.notna(channel):
            ch_send = df[df['channel_type']==channel].groupby('from_acct').size().rename(f'send_channel_{channel}')
            ch_recv = df[df['channel_type']==channel].groupby('to_acct').size().rename(f'recv_channel_{channel}')
    
    # === 幣別特徵 ===
    currency_diversity_send = df.groupby('from_acct')['currency_type'].nunique().rename('currency_diversity_send')
    currency_diversity_recv = df.groupby('to_acct')['currency_type'].nunique().rename('currency_diversity_recv')
    
    # 外幣交易
    foreign_curr_send = df[df['currency_type']!='TWD'].groupby('from_acct').size().rename('foreign_curr_send_count')
    foreign_curr_recv = df[df['currency_type']!='TWD'].groupby('to_acct').size().rename('foreign_curr_recv_count')
    foreign_curr_send_amt = df[df['currency_type']!='TWD'].groupby('from_acct')['txn_amt'].sum().rename('foreign_curr_send_amt')
    foreign_curr_recv_amt = df[df['currency_type']!='TWD'].groupby('to_acct')['txn_amt'].sum().rename('foreign_curr_recv_amt')
    
    # === 時間特徵 ===
    date_span_send = df.groupby('from_acct')['txn_date'].agg(lambda x: x.max() - x.min()).rename('date_span_send')
    date_span_recv = df.groupby('to_acct')['txn_date'].agg(lambda x: x.max() - x.min()).rename('date_span_recv')
    
    first_date_send = df.groupby('from_acct')['txn_date'].min().rename('first_date_send')
    last_date_send = df.groupby('from_acct')['txn_date'].max().rename('last_date_send')
    first_date_recv = df.groupby('to_acct')['txn_date'].min().rename('first_date_recv')
    last_date_recv = df.groupby('to_acct')['txn_date'].max().rename('last_date_recv')
    
    # 交易頻率（平均每天交易次數）
    avg_daily_send = (df.groupby('from_acct').size() / (date_span_send + 1)).rename('avg_daily_send_freq')
    avg_daily_recv = (df.groupby('to_acct').size() / (date_span_recv + 1)).rename('avg_daily_recv_freq')
    
    # 晚期交易特徵（最後30天）
    max_date = df['txn_date'].max()
    recent_df = df[df['txn_date'] > max_date - 30]
    recent_send_count = recent_df.groupby('from_acct').size().rename('recent_send_count')
    recent_recv_count = recent_df.groupby('to_acct').size().rename('recent_recv_count')
    recent_send_amt = recent_df.groupby('from_acct')['txn_amt'].sum().rename('recent_send_amt')
    recent_recv_amt = recent_df.groupby('to_acct')['txn_amt'].sum().rename('recent_recv_amt')
    
    # === 圖網絡特徵 - 與警示帳戶的關聯 ===
    if df_alert is not None:
        alert_accts = set(df_alert['acct'].values)
        
        # 一階關聯：直接與警示帳戶交易
        alert_send_count = df[df['to_acct'].isin(alert_accts)].groupby('from_acct').size().rename('alert_send_count')
        alert_send_amt = df[df['to_acct'].isin(alert_accts)].groupby('from_acct')['txn_amt'].sum().rename('alert_send_amt')
        alert_send_avg = df[df['to_acct'].isin(alert_accts)].groupby('from_acct')['txn_amt'].mean().rename('alert_send_avg')
        alert_send_max = df[df['to_acct'].isin(alert_accts)].groupby('from_acct')['txn_amt'].max().rename('alert_send_max')
        
        alert_recv_count = df[df['from_acct'].isin(alert_accts)].groupby('to_acct').size().rename('alert_recv_count')
        alert_recv_amt = df[df['from_acct'].isin(alert_accts)].groupby('to_acct')['txn_amt'].sum().rename('alert_recv_amt')
        alert_recv_avg = df[df['from_acct'].isin(alert_accts)].groupby('to_acct')['txn_amt'].mean().rename('alert_recv_avg')
        alert_recv_max = df[df['from_acct'].isin(alert_accts)].groupby('to_acct')['txn_amt'].max().rename('alert_recv_max')
        
        unique_alert_to = df[df['to_acct'].isin(alert_accts)].groupby('from_acct')['to_acct'].nunique().rename('unique_alert_to')
        unique_alert_from = df[df['from_acct'].isin(alert_accts)].groupby('to_acct')['from_acct'].nunique().rename('unique_alert_from')
        
        # 二階關聯：與警示帳戶有交易的帳戶（間接關聯）
        # 找出所有與警示帳戶有交易的帳戶
        first_hop_accts = set()
        first_hop_accts.update(df[df['from_acct'].isin(alert_accts)]['to_acct'].unique())
        first_hop_accts.update(df[df['to_acct'].isin(alert_accts)]['from_acct'].unique())
        first_hop_accts -= alert_accts  # 排除警示帳戶本身
        
        if len(first_hop_accts) > 0:
            indirect_send_count = df[df['to_acct'].isin(first_hop_accts)].groupby('from_acct').size().rename('indirect_alert_send_count')
            indirect_send_amt = df[df['to_acct'].isin(first_hop_accts)].groupby('from_acct')['txn_amt'].sum().rename('indirect_alert_send_amt')
            indirect_recv_count = df[df['from_acct'].isin(first_hop_accts)].groupby('to_acct').size().rename('indirect_alert_recv_count')
            indirect_recv_amt = df[df['from_acct'].isin(first_hop_accts)].groupby('to_acct')['txn_amt'].sum().rename('indirect_alert_recv_amt')
        else:
            indirect_send_count = pd.Series(dtype=float).rename('indirect_alert_send_count')
            indirect_send_amt = pd.Series(dtype=float).rename('indirect_alert_send_amt')
            indirect_recv_count = pd.Series(dtype=float).rename('indirect_alert_recv_count')
            indirect_recv_amt = pd.Series(dtype=float).rename('indirect_alert_recv_amt')
    
    # === 統計特徵 - 變異係數和偏度、峰度 ===
    cv_send = (std_send / (avg_send + 1)).rename('cv_send_amt')
    cv_recv = (std_recv / (avg_recv + 1)).rename('cv_recv_amt')
    
    # 偏度（skewness）- 衡量分布的不對稱性
    skew_send = df.groupby('from_acct')['txn_amt'].skew().rename('skew_send_amt')
    skew_recv = df.groupby('to_acct')['txn_amt'].skew().rename('skew_recv_amt')
    
    # 四分位數
    q25_send = df.groupby('from_acct')['txn_amt'].quantile(0.25).rename('q25_send_amt')
    q75_send = df.groupby('from_acct')['txn_amt'].quantile(0.75).rename('q75_send_amt')
    q25_recv = df.groupby('to_acct')['txn_amt'].quantile(0.25).rename('q25_recv_amt')
    q75_recv = df.groupby('to_acct')['txn_amt'].quantile(0.75).rename('q75_recv_amt')
    
    # 時間窗口特徵 - 早期vs晚期活動對比
    mid_date = df['txn_date'].min() + (df['txn_date'].max() - df['txn_date'].min()) / 2
    early_df = df[df['txn_date'] <= mid_date]
    late_df = df[df['txn_date'] > mid_date]
    
    early_send_count = early_df.groupby('from_acct').size().rename('early_send_count')
    late_send_count = late_df.groupby('from_acct').size().rename('late_send_count')
    early_recv_count = early_df.groupby('to_acct').size().rename('early_recv_count')
    late_recv_count = late_df.groupby('to_acct').size().rename('late_recv_count')
    
    # === 合併所有特徵 ===
    feature_list = [
        max_send, min_send, avg_send, std_send, median_send, send, send_count, unique_to,
        max_recv, min_recv, avg_recv, std_recv, median_recv, recv, recv_count, unique_from,
        self_txn_count, self_txn_amt,
        channel_diversity_send, channel_diversity_recv,
        currency_diversity_send, currency_diversity_recv,
        foreign_curr_send, foreign_curr_recv, foreign_curr_send_amt, foreign_curr_recv_amt,
        date_span_send, date_span_recv,
        first_date_send, last_date_send, first_date_recv, last_date_recv,
        avg_daily_send, avg_daily_recv,
        recent_send_count, recent_recv_count, recent_send_amt, recent_recv_amt,
        cv_send, cv_recv, skew_send, skew_recv,
        q25_send, q75_send, q25_recv, q75_recv,
        early_send_count, late_send_count, early_recv_count, late_recv_count
    ]
    
    if df_alert is not None:
        feature_list.extend([
            alert_send_count, alert_send_amt, alert_send_avg, alert_send_max,
            alert_recv_count, alert_recv_amt, alert_recv_avg, alert_recv_max,
            unique_alert_to, unique_alert_from,
            indirect_send_count, indirect_send_amt, indirect_recv_count, indirect_recv_amt
        ])
    
    df_result = pd.concat(feature_list, axis=1).fillna(0).reset_index()
    df_result.rename(columns={'index': 'acct'}, inplace=True)
    
    # === 帳戶類型特徵 ===
    df_from = df[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')
    
    # === 衍生特徵 ===
    df_result['send_recv_ratio'] = df_result['total_send_amt'] / (df_result['total_recv_amt'] + 1)
    df_result['recv_send_ratio'] = df_result['total_recv_amt'] / (df_result['total_send_amt'] + 1)
    df_result['avg_txn_amt'] = (df_result['total_send_amt'] + df_result['total_recv_amt']) / (df_result['send_count'] + df_result['recv_count'] + 1)
    df_result['total_txn_count'] = df_result['send_count'] + df_result['recv_count']
    df_result['total_txn_amt'] = df_result['total_send_amt'] + df_result['total_recv_amt']
    df_result['unique_counterparties'] = df_result['unique_to_accts'] + df_result['unique_from_accts']
    df_result['counterparty_concentration'] = df_result['total_txn_count'] / (df_result['unique_counterparties'] + 1)
    df_result['net_flow'] = df_result['total_recv_amt'] - df_result['total_send_amt']
    df_result['send_recv_count_ratio'] = df_result['send_count'] / (df_result['recv_count'] + 1)
    df_result['activity_trend'] = (df_result['late_send_count'] + df_result['late_recv_count']) / (df_result['early_send_count'] + df_result['early_recv_count'] + 1)
    df_result['iqr_send'] = df_result['q75_send_amt'] - df_result['q25_send_amt']
    df_result['iqr_recv'] = df_result['q75_recv_amt'] - df_result['q25_recv_amt']
    df_result['range_send'] = df_result['max_send_amt'] - df_result['min_send_amt']
    df_result['range_recv'] = df_result['max_recv_amt'] - df_result['min_recv_amt']
    
    # 警示帳戶關聯比例和交互特徵
    if df_alert is not None:
        df_result['alert_send_ratio'] = df_result['alert_send_count'] / (df_result['send_count'] + 1)
        df_result['alert_recv_ratio'] = df_result['alert_recv_count'] / (df_result['recv_count'] + 1)
        df_result['alert_total_ratio'] = (df_result['alert_send_count'] + df_result['alert_recv_count']) / (df_result['total_txn_count'] + 1)
        df_result['alert_amt_ratio'] = (df_result['alert_send_amt'] + df_result['alert_recv_amt']) / (df_result['total_txn_amt'] + 1)
        df_result['indirect_alert_ratio'] = (df_result['indirect_alert_send_count'] + df_result['indirect_alert_recv_count']) / (df_result['total_txn_count'] + 1)
        
        # 複合警示特徵
        df_result['alert_concentration'] = (df_result['alert_send_count'] + df_result['alert_recv_count']) / (df_result['unique_alert_to'] + df_result['unique_alert_from'] + 1)
        df_result['alert_avg_amt'] = (df_result['alert_send_amt'] + df_result['alert_recv_amt']) / (df_result['alert_send_count'] + df_result['alert_recv_count'] + 1)
    
    print(f"特徵數量: {len(df_result.columns) - 1}")
    print("(Finish) PreProcessing.")
    return df_result

def TrainTestSplit(df, df_alert, df_test):
    """
    切分訓練集及測試集，並為訓練集的帳戶標上警示label (0為非警示、1為警示)
    """  
    X_train = df[(~df['acct'].isin(df_test['acct'])) & (df['is_esun']==1)].drop(columns=['is_esun']).copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)
    X_test = df[df['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    
    print(f"訓練集大小: {len(X_train)}, 警示帳戶數: {y_train.sum()}")
    print(f"測試集大小: {len(X_test)}")
    print(f"警示帳戶比例: {y_train.sum() / len(y_train):.4f}")
    print(f"(Finish) Train-Test-Split")
    return X_train, X_test, y_train

def Modeling(X_train, y_train, X_test):
    """
    改用Random Forest，通常比單一決策樹效果更好
    """
    # 使用Random Forest，調整參數以平衡性能和過擬合
    model = RandomForestClassifier(
        n_estimators=300,            # 樹的數量
        max_depth=20,                # 適度增加深度
        min_samples_split=10,        # 降低分裂門檻以捕捉更多模式
        min_samples_leaf=5,          # 降低葉節點門檻
        max_features='sqrt',         # 每次分裂考慮的特徵數
        class_weight='balanced',     # 平衡類別權重
        random_state=42,
        n_jobs=-1,                   # 使用所有CPU核心
        max_samples=0.8              # bootstrap樣本比例
    )
    
    print("開始訓練模型...")
    model.fit(X_train.drop(columns=['acct']), y_train)
    
    # 使用概率預測，可以調整閾值
    y_pred_proba = model.predict_proba(X_test.drop(columns=['acct']))[:, 1]
    
    # 使用較低的閾值來提高召回率（因為警示帳戶較少，要盡量找出來）
    threshold = 0.3
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 動態閾值：根據訓練集警示比例調整
    alert_ratio = y_train.sum() / len(y_train)
    threshold = 0.3
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 輸出預測統計
    print(f"\n預測為警示帳戶數: {y_pred.sum()} / {len(y_pred)}")
    print(f"預測警示比例: {y_pred.sum() / len(y_pred):.4f}")
    
    # 輸出特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.drop(columns=['acct']).columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n前15個重要特徵:")
    print(feature_importance.head(15).to_string())
    print(f"\n(Finish) Modeling")
    return y_pred

def OutputCSV(path, df_test, X_test, y_pred):
    """
    根據測試資料集及預測結果，產出預測結果之CSV，該CSV可直接上傳於TBrain    
    """
    df_pred = pd.DataFrame({
        'acct': X_test['acct'].values,
        'label': y_pred
    })
    
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(path, index=False)    
    
    print(f"(Finish) Output saved to {path}")

if __name__ == "__main__":
    dir_path = "./raw_data/"
    df_txn, df_alert, df_test = LoadCSV(dir_path)
    
    # 傳入df_alert以計算圖網絡特徵
    df_X = PreProcessing(df_txn, df_alert)
    
    X_train, X_test, y_train = TrainTestSplit(df_X, df_alert, df_test)
    y_pred = Modeling(X_train, y_train, X_test)
    out_path = "result_improved.csv"
    OutputCSV(out_path, df_test, X_test, y_pred)
    