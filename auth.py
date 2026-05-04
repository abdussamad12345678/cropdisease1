import pickle
import os

USER_FILE = "users.pkl"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "rb") as f:
        return pickle.load(f)

def save_users(users):
    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

def register(username, password):
    users = load_users()

    if username in users:
        return False, "User already exists"

    users[username] = password
    save_users(users)
    return True, "Registration successful"

def login(username, password):
    users = load_users()

    if username not in users:
        return False, "User not found"

    if users[username] != password:
        return False, "Wrong password"

    return True, "Login successful"
