from app import app, db, User  # Make sure app.py has app, db, and User

with app.app_context():  # Correct usage
    users = User.query.all()
    for user in users:
        print(f"ID: {user.id}, Username: {user.username}, Password: {user.password}")
