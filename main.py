import cv2
import os
import csv
import sqlite3
import smtplib
import tkinter as tk
import numpy as np
import calendar
from datetime import date

from datetime import datetime
from tkinter import messagebox, simpledialog, ttk
from email.message import EmailMessage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- CONFIGURATION ----------------

DB_NAME = "attendance.db"
FACE_DIR = "faces"
MODEL_FILE = "face_model.yml"

ADMIN_PASSWORD = "admin123"

# Email settings: configure these to enable email notifications.
SENDER_EMAIL = ""
SENDER_APP_PASSWORD = ""

os.makedirs(FACE_DIR, exist_ok=True)
# OpenCV face detector
cascade_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError(
        f"Face detection file not found: {cascade_path}"
    )

# ---------------- DATABASE ----------------

def connect_db():
    return sqlite3.connect(DB_NAME)


def initialize_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                name TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        """)


# ---------------- EMAIL NOTIFICATION ----------------

def send_email(email, name, date, time):
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
        return "Email not configured"

    try:
        msg = EmailMessage()
        msg["Subject"] = "Attendance Marked Successfully"
        msg["From"] = SENDER_EMAIL
        msg["To"] = email

        msg.set_content(
            f"Hello {name},\n\n"
            f"Your attendance has been marked successfully.\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"Status: Present\n\n"
            "Smart Attendance System"
        )

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)

        return "Email sent"

    except Exception as error:
        print("Email error:", error)
        return "Email failed; check configuration"


# ---------------- REGISTER STUDENT ----------------

def register_student():
    student_id = simpledialog.askstring(
        "Registration", "Enter student ID:"
    )
    if not student_id:
        return

    student_id = student_id.strip()

    name = simpledialog.askstring(
        "Registration", "Enter student name:"
    )
    if not name or not name.strip():
        return
    name = name.strip()

    email = simpledialog.askstring(
        "Registration", "Enter student email:"
    )
    email = (email or "").strip()

    with connect_db() as con:
        existing = con.execute(
            "SELECT student_id FROM students WHERE student_id=?",
            (student_id,)
        ).fetchone()

        if existing:
            messagebox.showerror(
                "Error",
                "This student ID is already registered."
            )
            return

        con.execute(
            "INSERT INTO students VALUES (?, ?, ?)",
            (student_id, name, email)
        )

    student_dir = os.path.join(FACE_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Cannot open webcam.")
        return

    count = 0
    messagebox.showinfo(
        "Face Registration",
        "Look at the camera and slowly change your face angle.\n"
        "30 face samples will be captured.\n"
        "Press Q to cancel."
    )

    try:
        while count < 30:
            ok, frame = camera.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, 1.2, 5, minSize=(100, 100)
            )

            if len(faces) > 0:
                x, y, w, h = max(
                    faces, key=lambda f: f[2] * f[3]
                )

                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Samples: {count}/30",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                # Capture approximately one sample every 4 frames.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if cv2.getWindowProperty(
                    "Register Face", cv2.WND_PROP_VISIBLE
                ) < 0:
                    break

                # Save a sample on selected frames.
                if count == 0 or frame_number % 4 == 0:
                    cv2.imwrite(
                        os.path.join(
                            student_dir, f"{count}.jpg"
                        ),
                        face
                    )
                    count += 1

            frame_number = locals().get("frame_number", 0) + 1
            cv2.imshow("Register Face", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if count < 10:
        messagebox.showwarning(
            "Registration Incomplete",
            "Not enough face samples were captured. "
            "Register the student again using a new ID "
            "after removing the incomplete registration."
        )
        return

    train_model()

    messagebox.showinfo(
        "Success",
        f"Registered {name} with {count} face samples."
    )


# ---------------- TRAIN FACE MODEL ----------------

def train_model():
    images = []
    labels = []
    student_ids = []

    with connect_db() as con:
        students = con.execute(
            "SELECT student_id FROM students"
        ).fetchall()

    for (student_id,) in students:
        folder = os.path.join(FACE_DIR, student_id)

        if not os.path.isdir(folder):
            continue

        valid_images = 0

        for filename in os.listdir(folder):
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            path = os.path.join(folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            images.append(cv2.resize(image, (200, 200)))
            labels.append(len(student_ids))
            valid_images += 1

        if valid_images > 0:
            student_ids.append(student_id)

    if not images or not student_ids:
        messagebox.showwarning(
            "Training", "Register students and capture faces first."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(MODEL_FILE)

    # Store the mapping from model labels to student IDs.
    with open("labels.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(student_ids))


# ---------------- MARK ATTENDANCE ----------------

def mark_attendance(student_id):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    with connect_db() as con:
        student = con.execute(
            "SELECT name, email FROM students WHERE student_id=?",
            (student_id,)
        ).fetchone()

        if not student:
            return "Student not found"

        name, email = student

        existing = con.execute(
            "SELECT id FROM attendance "
            "WHERE student_id=? AND date=?",
            (student_id, today)
        ).fetchone()

        if existing:
            return f"{name}: attendance already marked today"

        con.execute(
            "INSERT INTO attendance "
            "(student_id, name, date, time, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, name, today, current_time, "Present")
        )

    email_status = send_email(
        email, name, today, current_time
    )

    return (
        f"Attendance marked for {name}\n"
        f"Date: {today}\nTime: {current_time}\n"
        f"{email_status}"
    )


def recognize_face():
    if not os.path.exists(MODEL_FILE):
        messagebox.showwarning(
            "Missing Model",
            "Register at least one student first."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    with open("labels.txt", encoding="utf-8") as file:
        student_ids = [
            line.strip() for line in file if line.strip()
        ]

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Cannot open webcam.")
        return

    attempts = 0
    result = "Press SPACE to attempt recognition"

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, 1.2, 5, minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2
                )

            cv2.putText(
                frame, result[:65], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Failed attempts: {attempts}/3",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )

            cv2.imshow("Smart Attendance", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Count one attempt only when SPACE is pressed.
            if key == ord(" ") and attempts < 3:
                if len(faces) == 0:
                    attempts += 1
                    result = "No face detected"
                else:
                    x, y, w, h = max(
                        faces, key=lambda f: f[2] * f[3]
                    )

                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))

                    label, distance = recognizer.predict(face)

                    # LBPH returns a distance, not a percentage.
                    # This threshold must be tested and calibrated.
                    if label < len(student_ids) and distance < 65:
                        student_id = student_ids[label]
                        result = mark_attendance(student_id)

                        messagebox.showinfo("Attendance", result)
                        break
                    else:
                        attempts += 1
                        result = "Unknown face - try again"

                if attempts >= 3:
                    messagebox.showwarning(
                        "Security Warning",
                        "3 unsuccessful attempts detected.\n"
                        "Recognition has been stopped.\n"
                        "Close the camera and contact the administrator."
                    )
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()



# ---------------- ATTENDANCE ANALYTICS ----------------

def attendance_analytics():
    """Show daily/monthly attendance charts using actual SQLite records."""
    analytics = tk.Toplevel(root)
    analytics.title("Attendance Analytics")
    analytics.geometry("1100x760")
    analytics.minsize(900, 650)

    now = datetime.now()
    controls = ttk.Frame(analytics, padding=10)
    controls.pack(fill="x")

    ttk.Label(controls, text="Month:").pack(side="left", padx=(0, 5))
    month_var = tk.StringVar(value=f"{now.month:02d}")
    month_box = ttk.Combobox(
        controls, textvariable=month_var, state="readonly", width=5,
        values=[f"{m:02d}" for m in range(1, 13)]
    )
    month_box.pack(side="left", padx=(0, 15))

    ttk.Label(controls, text="Year:").pack(side="left", padx=(0, 5))
    year_var = tk.StringVar(value=str(now.year))
    year_box = ttk.Combobox(
        controls, textvariable=year_var, state="readonly", width=7,
        values=[str(y) for y in range(now.year - 5, now.year + 2)]
    )
    year_box.pack(side="left", padx=(0, 15))

    summary = ttk.Label(analytics, text="", font=("Arial", 11, "bold"), padding=10)
    summary.pack(fill="x")

    chart_frame = ttk.Frame(analytics, padding=(10, 0, 10, 5))
    chart_frame.pack(fill="both", expand=True)

    note = ttk.Label(
        analytics,
        text=("Note: Absent = registered students minus students marked Present, "
              "calculated only for dates with at least one attendance record. "
              "Days with no records are omitted; holidays/non-working days are not inferred."),
        wraplength=1050, justify="left", padding=10
    )
    note.pack(fill="x")

    def refresh_analytics():
        try:
            selected_month = int(month_var.get())
            selected_year = int(year_var.get())
        except ValueError:
            messagebox.showerror("Invalid period", "Select a valid month and year.", parent=analytics)
            return

        with connect_db() as con:
            total_students = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            today = datetime.now().strftime("%Y-%m-%d")
            today_present = con.execute(
                "SELECT COUNT(DISTINCT student_id) FROM attendance "
                "WHERE date=? AND status='Present'", (today,)
            ).fetchone()[0]

            month_prefix = f"{selected_year:04d}-{selected_month:02d}"
            rows = con.execute(
                "SELECT date, COUNT(DISTINCT CASE WHEN status='Present' THEN student_id END) "
                "AS present FROM attendance WHERE date LIKE ? GROUP BY date ORDER BY date",
                (month_prefix + "-%",)
            ).fetchall()

            student_rows = con.execute(
                "SELECT s.student_id, s.name, "
                "COUNT(DISTINCT CASE WHEN a.status='Present' AND a.date LIKE ? THEN a.date END) "
                "AS present_days "
                "FROM students s LEFT JOIN attendance a ON s.student_id=a.student_id "
                "GROUP BY s.student_id, s.name ORDER BY s.name",
                (month_prefix + "-%",)
            ).fetchall()

        recorded_dates = [row[0] for row in rows]
        present_counts = [int(row[1] or 0) for row in rows]
        absent_counts = [max(total_students - count, 0) for count in present_counts]
        rates = [(count / total_students * 100) if total_students else 0 for count in present_counts]
        avg_rate = (sum(rates) / len(rates)) if rates else 0
        today_absent = max(total_students - today_present, 0)
        summary.config(
            text=(f"Registered students: {total_students}     |     "
                  f"Present today: {today_present}     |     "
                  f"Not marked present today: {today_absent}     |     "
                  f"Average attendance on recorded days ({month_prefix}): {avg_rate:.1f}%")
        )

        for child in chart_frame.winfo_children():
            child.destroy()

        fig = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
        ax_daily = fig.add_subplot(211)
        ax_monthly = fig.add_subplot(212)

        if recorded_dates:
            labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%d %b") for d in recorded_dates]
            x = np.arange(len(labels))
            width = 0.38
            ax_daily.bar(x - width / 2, present_counts, width, label="Present")
            ax_daily.bar(x + width / 2, absent_counts, width, label="Absent / not marked present")
            ax_daily.set_xticks(x)
            ax_daily.set_xticklabels(labels, rotation=45, ha="right")
            ax_daily.set_ylabel("Number of students")
            ax_daily.set_title(f"Daily Attendance — {calendar.month_name[selected_month]} {selected_year}")
            ax_daily.legend(loc="upper right")
            ax_daily.grid(axis="y", alpha=0.25)

            ax_monthly.plot(labels, rates, marker="o", linewidth=2, label="Attendance rate")
            ax_monthly.set_ylim(0, 100)
            ax_monthly.set_ylabel("Attendance (%)")
            ax_monthly.set_title("Attendance Rate Trend")
            ax_monthly.tick_params(axis="x", rotation=45)
            ax_monthly.grid(True, alpha=0.25)
            ax_monthly.legend(loc="best")
        else:
            ax_daily.text(0.5, 0.5, "No attendance records for this month", ha="center", va="center")
            ax_daily.set_axis_off()
            ax_monthly.set_axis_off()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Refresh the student-wise table in the same analytics window.
        for item in student_table.get_children():
            student_table.delete(item)
        days_with_records = len(recorded_dates)
        for student_id, name, present_days in student_rows:
            present_days = int(present_days or 0)
            pct = (present_days / days_with_records * 100) if days_with_records else 0
            student_table.insert("", "end", values=(student_id, name, present_days, f"{pct:.1f}%"))

    ttk.Button(controls, text="Update Charts", command=refresh_analytics).pack(side="left")

    table_frame = ttk.LabelFrame(analytics, text="Student-wise Monthly Attendance", padding=8)
    table_frame.pack(fill="x", padx=10, pady=(0, 8))
    student_table = ttk.Treeview(
        table_frame, columns=("ID", "Name", "Present Days", "Rate"),
        show="headings", height=5
    )
    for col, width in (("ID", 130), ("Name", 300), ("Present Days", 150), ("Rate", 130)):
        student_table.heading(col, text=col)
        student_table.column(col, width=width, anchor="center")
    student_table.pack(fill="x")

    refresh_analytics()


# ---------------- ADMIN PANEL ----------------

def admin_panel():
    password = simpledialog.askstring(
        "Admin Login", "Enter admin password:", show="*"
    )

    if password != ADMIN_PASSWORD:
        messagebox.showerror("Access Denied", "Incorrect password.")
        return

    window = tk.Toplevel(root)
    window.title("Admin Attendance Dashboard")
    window.geometry("850x450")

    columns = ("ID", "Name", "Date", "Time", "Status")

    table = ttk.Treeview(
        window, columns=columns, show="headings"
    )

    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=150)

    table.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_records():
        for item in table.get_children():
            table.delete(item)

        with connect_db() as con:
            records = con.execute(
                "SELECT student_id, name, date, time, status "
                "FROM attendance ORDER BY date DESC, time DESC"
            ).fetchall()

        for record in records:
            table.insert("", "end", values=record)

    def export_csv():
        with connect_db() as con:
            records = con.execute(
                "SELECT student_id, name, date, time, status "
                "FROM attendance ORDER BY date DESC, time DESC"
            ).fetchall()

        filename = (
            "attendance_" +
            datetime.now().strftime("%Y%m%d_%H%M%S") +
            ".csv"
        )

        with open(
            filename, "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(records)

        messagebox.showinfo(
            "Export Complete", f"Saved file: {filename}"
        )

    buttons = tk.Frame(window)
    buttons.pack(pady=8)

    tk.Button(
        buttons, text="Refresh Records",
        command=refresh_records
    ).pack(side="left", padx=5)

    tk.Button(
        buttons, text="Export CSV",
        command=export_csv
    ).pack(side="left", padx=5)

    tk.Button(
        buttons, text="Attendance Analytics",
        command=attendance_analytics
    ).pack(side="left", padx=5)

    refresh_records()


# ---------------- MAIN WINDOW ----------------

initialize_db()

root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("450x380")
root.resizable(False, False)

tk.Label(
    root, text="SMART ATTENDANCE SYSTEM",
    font=("Arial", 17, "bold")
).pack(pady=25)

tk.Button(
    root, text="Register Student / Capture Face",
    width=32, height=2, command=register_student
).pack(pady=8)

tk.Button(
    root, text="Recognize Face / Mark Attendance",
    width=32, height=2, command=recognize_face
).pack(pady=8)

tk.Button(
    root, text="Admin Panel",
    width=32, height=2, command=admin_panel
).pack(pady=8)

tk.Label(
    root,
    text="Admin password: admin123",
    font=("Arial", 9)
).pack(pady=12)

root.mainloop()
