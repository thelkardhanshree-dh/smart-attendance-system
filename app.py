from __future__ import annotations

import csv
import io
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, Response, flash, redirect, render_template, request,
                   session, url_for)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "attendance.db"
APP_PASSWORD = os.environ.get("SMART_ATTENDANCE_ADMIN_PASSWORD", "admin123")
app = Flask(__name__)
app.secret_key = os.environ.get("SMART_ATTENDANCE_SECRET_KEY", "change-this-local-secret-key")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def initialize_db():
    # Compatible with the schema used by the existing Tkinter application.
    with connect_db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, name TEXT,
            date TEXT, time TEXT, status TEXT
        )""")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def safe_month(value):
    try:
        datetime.strptime(value, "%Y-%m")
        return value
    except (ValueError, TypeError):
        return date.today().strftime("%Y-%m")


def month_dates(month):
    year, mon = map(int, month.split("-"))
    import calendar
    return [f"{year:04d}-{mon:02d}-{d:02d}" for d in range(1, calendar.monthrange(year, mon)[1] + 1)]


def get_analytics(month):
    dates = month_dates(month)
    with connect_db() as con:
        total_students = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        rows = con.execute("""SELECT date,
                COUNT(DISTINCT CASE WHEN status='Present' THEN student_id END) AS present
            FROM attendance WHERE date LIKE ? GROUP BY date ORDER BY date""", (month + "%",)).fetchall()
        by_date = {r["date"]: int(r["present"] or 0) for r in rows}
        # Absence is an estimate only on dates that have at least one attendance record.
        daily = []
        for day in dates:
            if day in by_date:
                present = min(by_date[day], total_students)
                daily.append({"date": day, "present": present,
                              "absent": max(total_students - present, 0),
                              "rate": round((present / total_students) * 100, 1) if total_students else 0,
                              "recorded": True})
            else:
                daily.append({"date": day, "present": 0, "absent": 0, "rate": None, "recorded": False})
        student_rows = con.execute("""SELECT s.student_id, s.name,
                COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.date END) AS present_days
            FROM students s LEFT JOIN attendance a ON a.student_id=s.student_id AND a.date LIKE ?
            GROUP BY s.student_id, s.name ORDER BY s.name COLLATE NOCASE""", (month + "%",)).fetchall()
        # A month-wide percentage is based on days where at least one attendance record exists.
        active_days = len(by_date)
        student_stats = []
        for row in student_rows:
            count = int(row["present_days"] or 0)
            pct = round(count / active_days * 100, 1) if active_days else None
            student_stats.append({"student_id": row["student_id"], "name": row["name"],
                                  "present_days": count, "percentage": pct})
        today = date.today().isoformat()
        today_row = con.execute("""SELECT COUNT(DISTINCT student_id) FROM attendance
                                  WHERE date=? AND status='Present'""", (today,)).fetchone()[0]
        today_has_records = con.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0] > 0
        attendance_count = con.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        latest_date = con.execute("SELECT MAX(date) FROM attendance").fetchone()[0]
    return {
        "total_students": total_students, "today_present": int(today_row),
        "today_has_records": today_has_records,
        "today_absent": max(total_students - int(today_row), 0) if today_has_records else None,
        "today_rate": round(int(today_row) / total_students * 100, 1) if total_students and today_has_records else None,
        "attendance_count": attendance_count, "latest_date": latest_date,
        "daily": daily, "student_stats": student_stats,
        "active_days": active_days,
        "month_present_total": sum(by_date.values()),
        "month_average": round(sum(by_date.values()) / (active_days * total_students) * 100, 1)
            if active_days and total_students else None,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == APP_PASSWORD:
            session["logged_in"] = True
            flash("Welcome to the Smart Attendance admin portal.", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Incorrect admin password.", "danger")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    month = safe_month(request.args.get("month", date.today().strftime("%Y-%m")))
    analytics = get_analytics(month)
    return render_template("dashboard.html", month=month, analytics=analytics,
                           daily_json=analytics["daily"], current_date=date.today().isoformat())


@app.route("/students", methods=["GET", "POST"])
@login_required
def students():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if not student_id or not name:
            flash("Student ID and name are required.", "danger")
        else:
            try:
                with connect_db() as con:
                    con.execute("INSERT INTO students (student_id, name, email) VALUES (?, ?, ?)",
                                (student_id, name, email))
                flash("Student added. Use the desktop app to capture face samples for recognition.", "success")
                return redirect(url_for("students"))
            except sqlite3.IntegrityError:
                flash("That student ID already exists.", "danger")
    query = request.args.get("q", "").strip()
    with connect_db() as con:
        if query:
            pattern = f"%{query}%"
            records = con.execute("""SELECT s.student_id, s.name, s.email,
                COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.date END) AS present_days
                FROM students s LEFT JOIN attendance a ON a.student_id=s.student_id
                WHERE s.student_id LIKE ? OR s.name LIKE ? OR s.email LIKE ?
                GROUP BY s.student_id, s.name, s.email ORDER BY s.name COLLATE NOCASE""",
                (pattern, pattern, pattern)).fetchall()
        else:
            records = con.execute("""SELECT s.student_id, s.name, s.email,
                COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.date END) AS present_days
                FROM students s LEFT JOIN attendance a ON a.student_id=s.student_id
                GROUP BY s.student_id, s.name, s.email ORDER BY s.name COLLATE NOCASE""").fetchall()
    return render_template("students.html", students=records, q=query)


@app.route("/students/<student_id>/edit", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    with connect_db() as con:
        student = con.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
        if student is None:
            flash("Student not found.", "danger")
            return redirect(url_for("students"))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip()
            if not name:
                flash("Name is required.", "danger")
            else:
                con.execute("UPDATE students SET name=?, email=? WHERE student_id=?", (name, email, student_id))
                con.execute("UPDATE attendance SET name=? WHERE student_id=?", (name, student_id))
                flash("Student details updated.", "success")
                return redirect(url_for("students"))
    return render_template("edit_student.html", student=student)


@app.route("/attendance")
@login_required
def attendance():
    start = request.args.get("start", date.today().replace(day=1).isoformat())
    end = request.args.get("end", date.today().isoformat())
    student_query = request.args.get("q", "").strip()
    # Validate date strings to avoid malformed query values.
    try:
        datetime.strptime(start, "%Y-%m-%d"); datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        start, end = date.today().replace(day=1).isoformat(), date.today().isoformat()
    sql = "SELECT id, student_id, name, date, time, status FROM attendance WHERE date BETWEEN ? AND ?"
    params = [start, end]
    if student_query:
        sql += " AND (student_id LIKE ? OR name LIKE ?)"
        params.extend([f"%{student_query}%", f"%{student_query}%"])
    sql += " ORDER BY date DESC, time DESC"
    with connect_db() as con:
        records = con.execute(sql, params).fetchall()
    return render_template("attendance.html", records=records, start=start, end=end, q=student_query)


@app.route("/export/attendance.csv")
@login_required
def export_csv():
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    sql = "SELECT student_id, name, date, time, status FROM attendance"
    params = []
    if start and end:
        sql += " WHERE date BETWEEN ? AND ?"
        params = [start, end]
    sql += " ORDER BY date DESC, time DESC"
    with connect_db() as con:
        rows = con.execute(sql, params).fetchall()
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Date", "Time", "Status"])
    writer.writerows([tuple(row) for row in rows])
    return Response("\ufeff" + output.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=attendance_export.csv"})


@app.route("/export/students.csv")
@login_required
def export_students():
    with connect_db() as con:
        rows = con.execute("SELECT student_id, name, email FROM students ORDER BY name COLLATE NOCASE").fetchall()
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Email"])
    writer.writerows([tuple(row) for row in rows])
    return Response("\ufeff" + output.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=students_export.csv"})


@app.route("/open-desktop-app", methods=["POST"])
@login_required
def open_desktop_app():
    # The Tkinter app continues to handle webcam face registration and LBPH recognition.
    main_file = BASE_DIR / "main.py"
    if not main_file.exists():
        flash("main.py was not found beside app.py. Copy your existing desktop application into this folder.", "danger")
    else:
        try:
            subprocess.Popen([sys.executable, str(main_file)], cwd=str(BASE_DIR))
            flash("Desktop face-recognition app launched. Use it for webcam registration and recognition.", "success")
        except Exception as exc:
            flash(f"Could not launch desktop app: {exc}", "danger")
    return redirect(url_for("dashboard"))


@app.route("/health")
def health():
    return {"status": "ok", "database": str(DB_PATH), "database_exists": DB_PATH.exists()}


initialize_db()

if __name__ == "__main__":
    print(f"Smart Attendance web dashboard using database: {DB_PATH}")
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)


