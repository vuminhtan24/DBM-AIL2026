import pickle
import pandas as pd
import numpy as np

print("Loading model...")

# Load model package
with open("exam_score_model.pkl", "rb") as f:
    pkg = pickle.load(f)

model = pkg["model"]
scaler = pkg["scaler"]
imputer = pkg["imputer"]
features = pkg["features"]

print("Model loaded!")


def predict_student(study_hours, attendance, **kwargs):

    row = {
        "study_hours": study_hours,
        "attendance": attendance
    }

    row.update(kwargs)

    df = pd.DataFrame([row])

    # thêm các feature còn thiếu
    for col in features:
        if col not in df.columns:
            df[col] = np.nan

    df = df[features]

    # impute missing
    df_imp = pd.DataFrame(imputer.transform(df), columns=features)

    # scale nếu cần
    if scaler:
        X = scaler.transform(df_imp)
    else:
        X = df_imp

    pred = model.predict(X)[0]

    return float(np.clip(pred, 0, 100))


# =========================
# INPUT USER
# =========================

print("\n===== DỰ ĐOÁN ĐIỂM THI =====")

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance (%): "))
sleep_hours = float(input("Sleep hours: "))
stress_level = float(input("Stress level (1-10): "))

score = predict_student(
    study_hours=study_hours,
    attendance=attendance,
    sleep_hours=sleep_hours,
    stress_level=stress_level
)

print(f"\n🎯 Predicted Exam Score: {score:.1f}/100")