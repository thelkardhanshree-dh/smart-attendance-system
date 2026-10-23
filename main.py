
import cv2
import os
import csv
import sqlite3
import smtplib
import tkinter as tk
import numpy as np

from datetime import datetime
from tkinter import messagebox, simpledialog, ttk
from email.message import EmailMessage

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
