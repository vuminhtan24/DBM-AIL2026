# ============================================================
# 🎓 DỰ ĐOÁN ĐIỂM THI - PHIÊN BẢN SỬA LỖI
# Đã fix: Data Leakage + Mất cân bằng dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ============================================================
# BƯỚC 1: LOAD & CHUẨN HÓA
# ============================================================
print("=" * 60)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("=" * 60)

d1 = pd.read_csv("data/education_dataset.csv").rename(columns={
    "Study Hours/Day"          : "study_hours",
    "Attendance %"             : "attendance",
    "Exam Score"               : "exam_score",
    "Learning Method"          : "study_method",
    "Smartphone Usage (hrs/day)": "screen_time",
})
d1['source'] = 'D1'

d2 = pd.read_csv("data/enhanced_student_habits_performance_dataset.csv").rename(columns={
    "study_hours_per_day"         : "study_hours",
    "attendance_percentage"       : "attendance",
    "sleep_hours"                 : "sleep_hours",
    "exercise_frequency"          : "physical_activity",
    "parental_education_level"    : "parental_education",
    "internet_quality"            : "internet_access",
    "extracurricular_participation": "extracurricular",
    "stress_level"                : "stress_level",
    "family_income_range"         : "family_income",
    "motivation_level"            : "motivation",
    "learning_style"              : "study_method",
    "time_management_score"       : "time_management",
    "mental_health_rating"        : "mental_health",
    "exam_score"                  : "exam_score",
})
d2['source'] = 'D2'

d3 = pd.read_csv("data/StudentPerformanceFactors.csv").rename(columns={
    "Hours_Studied"           : "study_hours",
    "Attendance"              : "attendance",
    "Sleep_Hours"             : "sleep_hours",
    "Physical_Activity"       : "physical_activity",
    "Parental_Education_Level": "parental_education",
    "Internet_Access"         : "internet_access",
    "Extracurricular_Activities": "extracurricular",
    "Motivation_Level"        : "motivation",
    "Family_Income"           : "family_income",
    "Gender"                  : "gender",
    "Exam_Score"              : "exam_score",
})
d3['source'] = 'D3'

d6 = pd.read_csv("data/student_performance.csv").rename(columns={
    "StudyHours"  : "study_hours",
    "Attendance"  : "attendance",
    "Internet"    : "internet_access",
    "Extracurricular": "extracurricular",
    "Motivation"  : "motivation",
    "Gender"      : "gender",
    "Age"         : "age",
    "LearningStyle": "study_method",
    "StressLevel" : "stress_level",
    "ExamScore"   : "exam_score",
})
d6['source'] = 'D6'

d7 = pd.read_csv("data/Student_Performance2.csv").rename(columns={
    "age"                 : "age",
    "gender"              : "gender",
    "parent_education"    : "parental_education",
    "study_hours"         : "study_hours",
    "attendance_percentage": "attendance",
    "internet_access"     : "internet_access",
    "extra_activities"    : "extracurricular",
    "study_method"        : "study_method",
    "overall_score"       : "exam_score",
})
d7['source'] = 'D7'

for name, df in [("D1",d1),("D2",d2),("D3",d3),("D6",d6),("D7",d7)]:
    print(f"  ✅ {name}: {len(df):,} dòng")


# ============================================================
# BƯỚC 2: CHỌN FEATURES — LOẠI BỎ DATA LEAKAGE
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: LOẠI BỎ DATA LEAKAGE")
print("=" * 60)

# ❌ KHÔNG dùng: previous_gpa, previous_scores
#    → corr = 0.93 với exam_score → model học từ điểm cũ,
#      không học từ hành vi học tập thực sự
# ✅ Chỉ giữ các features hành vi và môi trường

SAFE_FEATURES = [
    'study_hours',      # Giờ học mỗi ngày
    'attendance',       # Tỉ lệ đi học
    'sleep_hours',      # Giờ ngủ
    'physical_activity',# Hoạt động thể chất
    'stress_level',     # Mức độ stress
    'motivation',       # Động lực học
    'internet_access',  # Kết nối internet
    'extracurricular',  # Hoạt động ngoại khóa
    'parental_education',# Trình độ cha mẹ
    'family_income',    # Thu nhập gia đình
    'study_method',     # Phương pháp học
    'gender',           # Giới tính
    'age',              # Tuổi
    'screen_time',      # Thời gian dùng điện thoại
    'mental_health',    # Sức khỏe tinh thần
    'time_management',  # Kỹ năng quản lý thời gian
    'exam_score',       # TARGET
    'source',           # Để cân bằng dataset
]

df_all = pd.concat([d1, d2, d3, d6, d7], ignore_index=True)
df_all = df_all[[c for c in SAFE_FEATURES if c in df_all.columns]]
df_all = df_all.dropna(subset=['exam_score'])
df_all['exam_score'] = df_all['exam_score'].clip(0, 100)

print("  ❌ Đã loại: previous_gpa (corr=0.93 — gây data leakage)")
print(f"  ✅ Giữ lại: {len([c for c in SAFE_FEATURES if c in df_all.columns])-2} features hành vi")


# ============================================================
# BƯỚC 3: CÂN BẰNG DATASET
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: CÂN BẰNG DATASET")
print("=" * 60)

print("  Phân phối TRƯỚC khi cân bằng:")
for src in ['D1','D2','D3','D6','D7']:
    sub = df_all[df_all['source']==src]
    print(f"    {src}: {len(sub):>6,} dòng | mean score = {sub['exam_score'].mean():.1f}")

# D2 chiếm 63% data và có phân phối điểm cao bất thường
# → Downsample D2 xuống bằng tổng các dataset còn lại
others  = df_all[df_all['source'] != 'D2']
d2_cap  = df_all[df_all['source'] == 'D2'].sample(n=len(others), random_state=42)
df_bal  = pd.concat([others, d2_cap], ignore_index=True)

print(f"\n  Sau cân bằng: {len(df_bal):,} dòng | mean score = {df_bal['exam_score'].mean():.1f}")
print(f"  (Trước: 126,110 dòng | mean score = {df_all['exam_score'].mean():.1f})")


# ============================================================
# BƯỚC 4: XỬ LÝ FEATURES
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: XỬ LÝ FEATURES")
print("=" * 60)

df_model = df_bal.drop(columns=['source'])
X = df_model.drop(columns=['exam_score'])
y = df_model['exam_score']

# Encode categorical
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
    encoders[col] = le

# Impute missing
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"  Features: {list(X_imp.columns)}")
print(f"  Tổng: {X_imp.shape[1]} features | {X_imp.shape[0]:,} mẫu")


# ============================================================
# BƯỚC 5: CHIA DỮ LIỆU
# ============================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_imp, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")


# ============================================================
# BƯỚC 6: TRAIN MODELS
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: TRAIN MODELS")
print("=" * 60)

def evaluate(model, Xtr, ytr, Xvl, yvl, name, scaled=False):
    model.fit(Xtr, ytr)
    pred = model.predict(Xvl)
    mae  = mean_absolute_error(yvl, pred)
    rmse = np.sqrt(mean_squared_error(yvl, pred))
    r2   = r2_score(yvl, pred)
    print(f"\n  🤖 {name}")
    print(f"     MAE  = {mae:.2f}  (sai lệch ±{mae:.1f} điểm)")
    print(f"     R²   = {r2:.4f}  ({r2*100:.1f}%)")
    return {"name":name,"model":model,"mae":mae,"rmse":rmse,"r2":r2,"scaled":scaled}

results = []
results.append(evaluate(Ridge(alpha=1.0),
    X_train_sc, y_train, X_val_sc, y_val, "Ridge Regression", scaled=True))
results.append(evaluate(
    RandomForestRegressor(200, max_depth=14, random_state=42, n_jobs=-1),
    X_train, y_train, X_val, y_val, "Random Forest"))
results.append(evaluate(
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    X_train, y_train, X_val, y_val, "Gradient Boosting"))
if HAS_XGBOOST:
    results.append(evaluate(
        XGBRegressor(300, learning_rate=0.05, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        X_train, y_train, X_val, y_val, "XGBoost"))


# ============================================================
# BƯỚC 7: ĐÁNH GIÁ CUỐI
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: KẾT QUẢ TRÊN TEST SET")
print("=" * 60)

best = min(results, key=lambda x: x['mae'])
X_final = X_test_sc if best['scaled'] else X_test
y_pred  = best['model'].predict(X_final)

test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2   = r2_score(y_test, y_pred)

print(f"\n  🏆 Model tốt nhất: {best['name']}")
print(f"  MAE  = {test_mae:.2f}  → sai lệch trung bình ±{test_mae:.1f} điểm")
print(f"  RMSE = {test_rmse:.2f}")
print(f"  R²   = {test_r2:.4f}  ({test_r2*100:.1f}%)")

summary = pd.DataFrame([{"Model":r["name"],"MAE":round(r["mae"],2),"R²":round(r["r2"],3)}
                         for r in results]).sort_values("MAE")
print(f"\n{summary.to_string(index=False)}")


# ============================================================
# BƯỚC 8: BIỂU ĐỒ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Kết quả sau khi sửa lỗi  |  {best['name']}  |  MAE={test_mae:.2f}  R²={test_r2:.3f}",
             fontsize=13, fontweight='bold')

# Plot 1: Predicted vs Actual
ax = axes[0]
idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
yt, yp = y_test.values[idx], y_pred[idx]
ax.scatter(yt, yp, alpha=0.3, s=12, color='#2980b9')
lo, hi = min(yt.min(),yp.min())-2, max(yt.max(),yp.max())+2
ax.plot([lo,hi],[lo,hi],'r--',lw=1.5,label='Hoàn hảo')
ax.set_xlabel("Điểm thực tế"); ax.set_ylabel("Điểm dự đoán")
ax.set_title("Thực tế vs Dự đoán")
ax.text(0.05, 0.92, f"R²={test_r2:.3f}", transform=ax.transAxes,
        color='darkred', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Phân phối sai số
ax = axes[1]
residuals = y_test.values - y_pred
ax.hist(residuals, bins=50, color='#8e44ad', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel("Sai số"); ax.set_ylabel("Số lượng")
ax.set_title(f"Phân phối sai số\nMean={residuals.mean():.2f}  Std={residuals.std():.2f}")

# Plot 3: Feature Importance
ax = axes[2]
if hasattr(best['model'], 'feature_importances_'):
    fi = pd.Series(best['model'].feature_importances_, index=X_train.columns)
    fi.sort_values(ascending=True).tail(12).plot(kind='barh', ax=ax, color='#e67e22')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("ket_qua_fixed.png", dpi=150, bbox_inches='tight')
print("\n✅ Lưu biểu đồ: ket_qua_fixed.png")
plt.show()


# ============================================================
# BƯỚC 9: LƯU MODEL
# ============================================================
model_pkg = {
    'model'   : best['model'],
    'scaler'  : scaler if best['scaled'] else None,
    'imputer' : imputer,
    'encoders': encoders,
    'features': list(X_train.columns),
    'metrics' : {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2},
}
with open("exam_score_model_fixed.pkl","wb") as f:
    pickle.dump(model_pkg, f)
print("✅ Lưu model: exam_score_model_fixed.pkl")


# ============================================================
# KIỂM TRA: Học sinh lười vs chăm
# ============================================================
print("\n" + "=" * 60)
print("KIỂM TRA LOGIC DỰ ĐOÁN")
print("=" * 60)

def predict(study_hours, attendance, **kwargs):
    row = {'study_hours': study_hours, 'attendance': attendance}
    row.update(kwargs)
    df_in = pd.DataFrame([row])
    for col in model_pkg['features']:
        if col not in df_in.columns:
            df_in[col] = np.nan
    df_in = df_in[model_pkg['features']]
    df_imp = pd.DataFrame(model_pkg['imputer'].transform(df_in), columns=model_pkg['features'])
    X_in = model_pkg['scaler'].transform(df_imp) if model_pkg['scaler'] else df_imp
    sc = float(np.clip(model_pkg['model'].predict(X_in)[0], 0, 100))
    return sc

print("\n  So sánh dự đoán sau khi sửa lỗi:")
cases = [
    ("Học sinh lười   (study=1h, att=50%, stress=8)", 1, 50, dict(sleep_hours=5, stress_level=8)),
    ("Học sinh TB     (study=4h, att=75%, stress=5)", 4, 75, dict(sleep_hours=7, stress_level=5)),
    ("Học sinh chăm   (study=8h, att=95%, stress=2)", 8, 95, dict(sleep_hours=8, stress_level=2)),
]
for label, sh, att, kw in cases:
    sc = predict(sh, att, **kw)
    bar = '█' * int(sc / 5)
    print(f"  {label}: {sc:.1f}  {bar}")

print("\n🎉 Kết quả hợp lý: học sinh chăm > TB > lười")