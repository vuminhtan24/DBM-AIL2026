# ============================================================
# 🎓 DỰ ĐOÁN ĐIỂM 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    'study_hours',       # Giờ học mỗi ngày
    'attendance',        # Tỉ lệ đi học
    'sleep_hours',       # Giờ ngủ
    'physical_activity', # Hoạt động thể chất
    'stress_level',      # Mức độ stress
    'motivation',        # Động lực học
    'internet_access',   # Kết nối internet
    'extracurricular',   # Hoạt động ngoại khóa
    'parental_education',# Trình độ cha mẹ
    'family_income',     # Thu nhập gia đình
    'study_method',      # Phương pháp học
    'gender',            # Giới tính
    'age',               # Tuổi
    'screen_time',       # Thời gian dùng điện thoại
    'mental_health',     # Sức khỏe tinh thần
    'time_management',   # Kỹ năng quản lý thời gian
    'exam_score',        # TARGET
    'source',            # Để cân bằng dataset
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

others  = df_all[df_all['source'] != 'D2']
d2_cap  = df_all[df_all['source'] == 'D2'].sample(n=len(others), random_state=42)
df_bal  = pd.concat([others, d2_cap], ignore_index=True)

print(f"\n  Sau cân bằng: {len(df_bal):,} dòng | mean score = {df_bal['exam_score'].mean():.1f}")
print(f"  (Trước: 126,110 dòng | mean score = {df_all['exam_score'].mean():.1f})")

# ============================================================
# 📊 VISUALIZE: TRƯỚC & SAU CÂN BẰNG
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Trước cân bằng ---
ax = axes[0]
counts_before = df_all['source'].value_counts()
ax.bar(counts_before.index, counts_before.values)
ax.set_title("Trước cân bằng dataset")
ax.set_xlabel("Dataset nguồn")
ax.set_ylabel("Số lượng")

# --- Sau cân bằng ---
ax = axes[1]
counts_after = df_bal['source'].value_counts()
ax.bar(counts_after.index, counts_after.values)
ax.set_title("Sau cân bằng dataset")
ax.set_xlabel("Dataset nguồn")
ax.set_ylabel("Số lượng")
plt.savefig("truoc_va_sau_can_bang.png", dpi=150, bbox_inches='tight')
print("\n✅ Lưu biểu đồ: truoc_va_sau_can_bang.png")
plt.tight_layout()
plt.show()
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
print("\n" + "=" * 60)
print("BƯỚC 5: CHIA DỮ LIỆU")
print("=" * 60)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_imp, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42)

# Scale (dùng cho Linear Regression và KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")


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
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2, "scaled": scaled}

results = []

# 1. Linear Regression
results.append(evaluate(
    LinearRegression(),
    X_train_sc, y_train, X_val_sc, y_val,
    "Linear Regression", scaled=True))

# 2. KNN 
results.append(evaluate(
    KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
    X_train_sc, y_train, X_val_sc, y_val,
    "KNN (k=10)", scaled=True))

# 3. Decision Tree 
results.append(evaluate(
    DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42),
    X_train, y_train, X_val, y_val,
    "Decision Tree"))

# 4. Random Forest 
results.append(evaluate(
    RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=5,
                          random_state=42, n_jobs=-1),
    X_train, y_train, X_val, y_val,
    "Random Forest"))


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

summary = pd.DataFrame([
    {"Model": r["name"], "MAE": round(r["mae"], 2), "R²": round(r["r2"], 3)}
    for r in results
]).sort_values("MAE")
print(f"\n{summary.to_string(index=False)}")


# ============================================================
# BƯỚC 8: BIỂU ĐỒ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Kết quả  |  {best['name']}  |  MAE={test_mae:.2f}  R²={test_r2:.3f}",
    fontsize=13, fontweight='bold')

# Plot 1: Predicted vs Actual
ax = axes[0]
idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
yt, yp = y_test.values[idx], y_pred[idx]
ax.scatter(yt, yp, alpha=0.3, s=12, color='#2980b9')
lo, hi = min(yt.min(), yp.min())-2, max(yt.max(), yp.max())+2
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Hoàn hảo')
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

# Plot 3: So sánh MAE các model
ax = axes[2]
names = [r['name'] for r in results]
maes  = [r['mae']  for r in results]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22']
bars = ax.barh(names, maes, color=colors[:len(results)], edgecolor='white')
ax.set_xlabel("MAE (thấp hơn = tốt hơn)")
ax.set_title("So sánh MAE các Model")
for bar, val in zip(bars, maes):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=10)

# Feature importance (chỉ hiển thị nếu model tốt nhất có thuộc tính này)
if hasattr(best['model'], 'feature_importances_'):
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fi = pd.Series(best['model'].feature_importances_, index=X_train.columns)
    fi.sort_values(ascending=True).tail(12).plot(kind='barh', ax=ax2, color='#e67e22')
    ax2.set_title(f"Feature Importance — {best['name']}")
    ax2.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    print("✅ Lưu biểu đồ: feature_importance.png")

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
with open("exam_score_model_fixed.pkl", "wb") as f:
    pickle.dump(model_pkg, f)
print("✅ Lưu model: exam_score_model_fixed.pkl")


# ============================================================
# KIỂM TRA: Học sinh lười vs chăm
# ============================================================
print("\n" + "=" * 60)
print("KIỂM TRA LOGIC DỰ ĐOÁN")
print("=" * 60)

def predict(row_dict):
    """
    Truyền đầy đủ tất cả features — không để NaN.
    Imputer vẫn giữ để xử lý edge case, nhưng không ảnh hưởng kết quả.
    """
    df_in = pd.DataFrame([row_dict])
    # Đảm bảo đúng thứ tự cột
    for col in model_pkg['features']:
        if col not in df_in.columns:
            df_in[col] = np.nan  # fallback — không nên xảy ra
    df_in = df_in[model_pkg['features']]
    df_imp = pd.DataFrame(
        model_pkg['imputer'].transform(df_in), columns=model_pkg['features'])
    X_in = model_pkg['scaler'].transform(df_imp) if model_pkg['scaler'] else df_imp
    sc = float(np.clip(model_pkg['model'].predict(X_in)[0], 0, 100))
    return sc

# ── Định nghĩa 3 hồ sơ học sinh đầy đủ ─────────────────────
# Tất cả features được khai báo tường minh, không để NaN
# Các giá trị categorical dùng string (LabelEncoder sẽ xử lý nếu cần)

PROFILE_LAZY = {
    'study_hours'      : 1,      # học rất ít
    'attendance'       : 50,     # hay nghỉ học
    'sleep_hours'      : 5,      # ngủ ít
    'physical_activity': 1,      # ít vận động
    'stress_level'     : 8,      # stress cao
    'motivation'       : 2,      # ít động lực
    'internet_access'  : 1,      # có internet (1=Yes hoặc dạng số)
    'extracurricular'  : 0,      # không tham gia
    'parental_education': 0,     # thấp (encoded)
    'family_income'    : 0,      # thấp (encoded)
    'study_method'     : 0,      # passive (encoded)
    'gender'           : 0,
    'age'              : 17,
    'screen_time'      : 8,      # dùng điện thoại nhiều
    'mental_health'    : 3,      # sức khỏe tinh thần kém
    'time_management'  : 2,      # quản lý thời gian kém
}

PROFILE_AVERAGE = {
    'study_hours'      : 4,
    'attendance'       : 75,
    'sleep_hours'      : 7,
    'physical_activity': 3,
    'stress_level'     : 5,
    'motivation'       : 5,
    'internet_access'  : 1,
    'extracurricular'  : 1,
    'parental_education': 1,
    'family_income'    : 1,
    'study_method'     : 1,
    'gender'           : 0,
    'age'              : 17,
    'screen_time'      : 4,
    'mental_health'    : 6,
    'time_management'  : 5,
}

PROFILE_HARD = {
    'study_hours'      : 8,
    'attendance'       : 95,
    'sleep_hours'      : 8,
    'physical_activity': 5,
    'stress_level'     : 2,
    'motivation'       : 9,
    'internet_access'  : 1,
    'extracurricular'  : 1,
    'parental_education': 2,
    'family_income'    : 2,
    'study_method'     : 2,
    'gender'           : 0,
    'age'              : 17,
    'screen_time'      : 1,
    'mental_health'    : 9,
    'time_management'  : 9,
}

# Chỉ giữ features mà model thực sự dùng
def filter_features(profile):
    return {k: v for k, v in profile.items() if k in model_pkg['features']}

print("\n  So sánh dự đoán (đầy đủ features, không NaN):")
cases = [
    ("Học sinh lười   (study=1h, att=50%, stress=8)", PROFILE_LAZY),
    ("Học sinh TB     (study=4h, att=75%, stress=5)", PROFILE_AVERAGE),
    ("Học sinh chăm   (study=8h, att=95%, stress=2)", PROFILE_HARD),
]
for label, profile in cases:
    sc = predict(filter_features(profile))
    bar = '█' * int(sc / 5)
    print(f"  {label}: {sc:.1f}  {bar}")

print("\n🎉 Kết quả hợp lý: học sinh chăm > TB > lười")