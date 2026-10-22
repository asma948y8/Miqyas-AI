import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "miqyas.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-local-development-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
REQUIRED_COLUMNS = ["اسم المورد", "السعر", "مدة التسليم", "الضمان"]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                budget REAL,
                delivery_limit REAL,
                warranty_min REAL,
                offers_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                ai_summary TEXT
            )
        """)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_number(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(str(value).replace(",", "").replace(" ريال", "").strip())
    except (ValueError, TypeError):
        return None


def normalize_columns(df):
    aliases = {
        "اسم المورد": ["اسم المورد", "المورد", "اسم الشركة", "Supplier", "Supplier Name", "Vendor"],
        "السعر": ["السعر", "السعر (ريال)", "السعر بالريال", "Price", "Cost", "Amount"],
        "مدة التسليم": ["مدة التسليم", "مدة التسليم (يوم)", "التسليم بالأيام", "Delivery Days", "Delivery Time"],
        "الضمان": ["الضمان", "الضمان (شهر)", "مدة الضمان", "Warranty", "Warranty Months"],
        "مطابقة المواصفات": ["مطابقة المواصفات", "المواصفات", "Compliance", "Specification Match"],
        "ملاحظات": ["ملاحظات", "ملاحظات إضافية", "Notes", "Comments"],
    }
    rename = {}
    for standard, names in aliases.items():
        for col in df.columns:
            if str(col).strip().lower() in [n.lower() for n in names]:
                rename[col] = standard
                break
    return df.rename(columns=rename)


def score_offers(offers, budget, delivery_limit, warranty_min):
    valid_prices = [o["السعر"] for o in offers if o["السعر"] is not None and o["السعر"] > 0]
    max_delivery = max([o["مدة التسليم"] for o in offers if o["مدة التسليم"] is not None] or [1])
    max_warranty = max([o["الضمان"] for o in offers if o["الضمان"] is not None] or [1])

    for offer in offers:
        issues = []
        price = offer["السعر"]
        delivery = offer["مدة التسليم"]
        warranty = offer["الضمان"]
        compliance = str(offer.get("مطابقة المواصفات") or "").strip().lower()
        score = 0.0

        if price is None or price <= 0:
            issues.append("السعر مفقود أو غير صالح")
        else:
            cheapest = min(valid_prices) if valid_prices else price
            score += 40 * (cheapest / price)

        if delivery is None:
            issues.append("مدة التسليم غير مذكورة")
        elif delivery_limit and delivery > delivery_limit:
            issues.append("مدة التسليم تتجاوز الحد المحدد")
        else:
            score += 25 * (1 - (delivery / max_delivery)) if max_delivery else 25

        if warranty is None:
            issues.append("الضمان غير مذكور")
        elif warranty_min and warranty < warranty_min:
            issues.append("الضمان أقل من الحد المطلوب")
        else:
            score += 20 * (warranty / max_warranty) if max_warranty else 20

        if compliance in {"مطابق", "مطابقة", "نعم", "yes", "compliant", "مطابق للمواصفات"}:
            score += 15
        elif compliance in {"غير مطابق", "لا", "no", "non-compliant"}:
            issues.append("المواصفات غير مطابقة")
        else:
            issues.append("مطابقة المواصفات تحتاج إلى مراجعة")

        if budget and price is not None and price > budget:
            issues.append("السعر يتجاوز الميزانية")

        offer["الدرجة"] = round(max(0, min(100, score)), 1)
        offer["الملاحظات_والتحذيرات"] = issues
        offer["ضمن_الميزانية"] = (price is not None and (not budget or price <= budget))
    offers.sort(key=lambda x: x["الدرجة"], reverse=True)
    for i, offer in enumerate(offers, 1):
        offer["الترتيب"] = i
    return offers


def gemini_summary(offers, budget, delivery_limit, warranty_min):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "لم يُضف مفتاح Gemini بعد. المقارنة الرقمية تعمل، ويمكن تفعيل الملخص الذكي بإضافة GEMINI_API_KEY إلى ملف .env."

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        payload = {
            "budget_sar": budget,
            "max_delivery_days": delivery_limit,
            "minimum_warranty_months": warranty_min,
            "offers": offers,
        }
        prompt = (
            "أنت مساعد تحليل مشتريات. حلّل بيانات عروض الموردين التالية باللغة العربية. "
            "اعتمد على البيانات المقدمة فقط، ولا تخترع معلومات أو أرقامًا. "
            "اشرح أفضل عرض بحسب الدرجة الرقمية، واذكر المفاضلات والتحذيرات والبيانات الناقصة. "
            "لا تعتبر توصيتك قرار شراء نهائيًا، ونبّه إلى ضرورة التحقق من المواصفات والشروط. "
            "اكتب ملخصًا واضحًا بعناوين ونقاط قصيرة.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            contents=prompt,
        )
        return response.text or "لم يُرجع Gemini نصًا. راجعي إعدادات المفتاح والنموذج."
    except ImportError as exc:
        return (
            "تعذر تحميل مكتبة Gemini في بيئة Python الحالية. "
            "ثبّتي متطلبات requirements.txt باستخدام مفسر Python نفسه الذي يشغّل التطبيق. "
            f"التفاصيل: {exc}"
        )
    except Exception as exc:
        return f"تعذر الحصول على تحليل Gemini: {exc}. تحققي من مفتاح API والاتصال بالإنترنت."


@app.route("/")
def index():
    with get_db() as conn:
        recent = conn.execute(
            "SELECT id, title, created_at FROM comparisons ORDER BY id DESC LIMIT 5"
        ).fetchall()
        count = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
    return render_template("index.html", recent=recent, count=count)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return render_template("compare.html")

    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        flash("اختاري ملف Excel أو CSV أولًا.", "error")
        return redirect(url_for("compare"))
    if not allowed_file(uploaded.filename):
        flash("الملفات المدعومة حاليًا هي .xlsx و.csv فقط.", "error")
        return redirect(url_for("compare"))

    safe_name = secure_filename(uploaded.filename)
    saved_path = UPLOAD_DIR / safe_name
    uploaded.save(saved_path)

    try:
        df = pd.read_excel(saved_path) if saved_path.suffix.lower() == ".xlsx" else pd.read_csv(saved_path)
        df = normalize_columns(df)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            flash("الأعمدة المطلوبة غير موجودة: " + "، ".join(missing), "error")
            return redirect(url_for("compare"))

        offers = []
        for _, row in df.iterrows():
            supplier = str(row.get("اسم المورد", "")).strip()
            if supplier.lower() in {"", "nan", "none"}:
                supplier = "مورد بدون اسم"
            offers.append({
                "اسم المورد": supplier,
                "السعر": to_number(row.get("السعر")),
                "مدة التسليم": to_number(row.get("مدة التسليم")),
                "الضمان": to_number(row.get("الضمان")),
                "مطابقة المواصفات": str(row.get("مطابقة المواصفات", "")).strip(),
                "ملاحظات": str(row.get("ملاحظات", "")).strip(),
            })
        if not offers:
            flash("الملف لا يحتوي على عروض.", "error")
            return redirect(url_for("compare"))

        budget = to_number(request.form.get("budget"))
        delivery_limit = to_number(request.form.get("delivery_limit"))
        warranty_min = to_number(request.form.get("warranty_min"))
        offers = score_offers(offers, budget, delivery_limit, warranty_min)
        ai_summary = gemini_summary(offers, budget, delivery_limit, warranty_min)
        title = request.form.get("title", "").strip() or f"مقارنة {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO comparisons
                (title, created_at, budget, delivery_limit, warranty_min, offers_json, result_json, ai_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (title, created_at, budget, delivery_limit, warranty_min,
                 json.dumps(offers, ensure_ascii=False), json.dumps(offers, ensure_ascii=False), ai_summary)
            )
            comparison_id = cur.lastrowid

        return render_template(
            "results.html", offers=offers, ai_summary=ai_summary,
            title=title, comparison_id=comparison_id,
            budget=budget, delivery_limit=delivery_limit, warranty_min=warranty_min
        )
    except Exception as exc:
        flash(f"تعذر قراءة الملف: {exc}", "error")
        return redirect(url_for("compare"))


@app.route("/history")
def history():
    with get_db() as conn:
        records = conn.execute(
            "SELECT id, title, created_at FROM comparisons ORDER BY id DESC"
        ).fetchall()
    return render_template("history.html", records=records)


@app.route("/history/<int:comparison_id>")
def history_detail(comparison_id):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
    if record is None:
        flash("لم يتم العثور على المقارنة.", "error")
        return redirect(url_for("history"))
    return render_template(
        "results.html",
        offers=json.loads(record["result_json"]),
        ai_summary=record["ai_summary"] or "لا يوجد ملخص ذكي محفوظ.",
        title=record["title"], comparison_id=record["id"],
        budget=record["budget"], delivery_limit=record["delivery_limit"],
        warranty_min=record["warranty_min"]
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "Miqyas AI"})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
