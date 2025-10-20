from django.shortcuts import render, redirect
from django.contrib import messages
import os
import pandas as pd
import joblib
import numpy as np

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'credit_card_fraud_xgb_model.pkl')
model = joblib.load(MODEL_PATH)


def home(request):
    files = os.listdir(UPLOAD_FOLDER)
    context = {"files": files}
    return render(request, "predictor/upload.html", context)


def upload_file(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb+") as dest:
                for chunk in file.chunks():
                    dest.write(chunk)
            messages.success(request, f"File '{file.name}' uploaded successfully!")
        else:
            messages.error(request, "No file selected!")
    return redirect("home")


def view_file(request):
    file_name = request.GET.get("file")
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        table_html = df.to_html(classes="table table-striped", index=False)
        context = {"files": os.listdir(UPLOAD_FOLDER), "table_html": table_html}
        return render(request, "predictor/upload.html", context)
    return redirect("home")


def delete_file(request):
    file_name = request.GET.get("file")
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        messages.success(request, f"File '{file_name}' deleted successfully!")
    else:
        messages.error(request, "File not found!")
    return redirect("home")


def predict(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb+") as dest:
                for chunk in file.chunks():
                    dest.write(chunk)

            df = pd.read_csv(file_path)
            feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            X = df[feature_cols].values

            probs = model.predict_proba(X)[:, 1]
            df["Fraud_Probability"] = np.round(probs, 4)

            # ✅ Lower threshold to show frauds
            df["Fraud_Prediction"] = (df["Fraud_Probability"] > 0.0003).astype(int)

            df.to_csv(file_path, index=False)
            table_html = df.to_html(classes="table table-striped", index=False)
            return render(request, "predictor/upload.html", {
                "table_html": table_html,
                "files": os.listdir(UPLOAD_FOLDER)
            })

    elif request.method == "GET":
        file_name = request.GET.get("file")
        if file_name:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
                X = df[feature_cols].values

                probs = model.predict_proba(X)[:, 1]
                df["Fraud_Probability"] = np.round(probs, 4)
                df["Fraud_Prediction"] = (df["Fraud_Probability"] > 0.0003).astype(int)

                table_html = df.to_html(classes="table table-striped", index=False)
                return render(request, "predictor/upload.html", {
                    "table_html": table_html,
                    "files": os.listdir(UPLOAD_FOLDER)
                })
    return redirect("home")


def analyze_file(request):
    file_name = request.GET.get("file")
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # ✅ Ensure Fraud_Prediction exists and uses the same threshold logic
        if "Fraud_Prediction" not in df.columns:
            feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            probs = model.predict_proba(df[feature_cols].values)[:, 1]
            df["Fraud_Probability"] = np.round(probs, 4)
            df["Fraud_Prediction"] = (df["Fraud_Probability"] > 0.0003).astype(int)

        stats = {
            "num_transactions": len(df),
            "avg_amount": round(df["Amount"].mean(), 2),
            "min_amount": df["Amount"].min(),
            "max_amount": df["Amount"].max(),
            "fraud_count": int(df["Fraud_Prediction"].sum()),
            "fraud_percentage": round(df["Fraud_Prediction"].mean() * 100, 2)
        }

        chart_data = {
            "non_fraud": stats["num_transactions"] - stats["fraud_count"],
            "fraud": stats["fraud_count"],
            "amounts": {
                "min": stats["min_amount"],
                "avg": stats["avg_amount"],
                "max": stats["max_amount"]
            }
        }

        return render(request, "predictor/analysis.html", {
            "stats": stats,
            "chart_data": chart_data,
            "file_name": file_name
        })

    messages.error(request, "File not found!")
    return redirect("home")
