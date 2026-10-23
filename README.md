
# Smart Attendance System

A Python-based Smart Attendance System using face recognition, OpenCV, SQLite, and Tkinter.

## Features

- Face detection and recognition using a webcam
- Student registration and face sample capture
- Automatic attendance marking with date and time
- Security warning after three failed recognition attempts
- Admin panel to view attendance records
- Export attendance records to CSV
- Optional email notifications

## Technologies Used

- Python
- OpenCV
- NumPy
- SQLite
- Tkinter
- SMTP Email

## Installation

1. Clone this repository or download the source code.
2. Open the project folder in VS Code.
3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it on Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Run the application:

   ```bash
   python main.py
   ```

## Usage

1. Register a student with a unique ID, name, and email.
2. Capture face samples using the webcam.
3. Start face recognition to mark attendance.
4. Open the admin panel to view and export attendance records.

## Notes

- Register students before attempting recognition.
- Email notifications require sender email credentials and an app password.
- The default demo admin password is `admin123`. Change it before any real deployment.
- This project is an educational prototype. It does not include reliable liveness detection or production-grade access controls.

## Privacy

Do not commit personal face images, trained models containing personal data, attendance databases, email credentials, or other sensitive information to a public repository.
