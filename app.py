import requests

BASE_URL = "http://127.0.0.1:8000"  # Assuming the FastAPI server is running locally

# Step 1: Start a new session
def start_session():
    response = requests.post(f"{BASE_URL}/start-session")
    if response.status_code == 200:
        session_id = response.json()["session_id"]
        print(f"Session started successfully. Session ID: {session_id}")
        return session_id
    else:
        print("Failed to start a session")
        return None

# Step 2: Send a chat message
def send_chat_message(session_id, prompt):
    payload = {"prompt": prompt}
    response = requests.post(f"{BASE_URL}/chat", json=payload, params={"session_id": session_id})
    if response.status_code == 200:
        chat_response = response.json()["response"]
        print(f"Chatbot response: {chat_response}")
        return chat_response
    else:
        print("Failed to send chat message")
        return None

# Step 3: Get chat history
def get_chat_history(session_id):
    response = requests.get(f"{BASE_URL}/chat-history", params={"session_id": session_id})
    if response.status_code == 200:
        history = response.json()
        print("Chat History:")
        for message in history:
            print(f"User: {message['human']}")
            print(f"AI: {message['ai']}")
        return history
    else:
        print("Failed to retrieve chat history")
        return None

# Testing the API
if __name__ == "__main__":
    # Start a new session
    session_id = start_session()
    
    if session_id:
        # Send a few chat messages
        send_chat_message(session_id, "Hello, how are you?")
        send_chat_message(session_id, "What's the weather like today?")
        
        # Retrieve the chat history
        get_chat_history(session_id)
