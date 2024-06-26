# init_db.py

from app import app, db

# Use the application context to create tables
with app.app_context():
    db.create_all()

print("Database tables created.")
