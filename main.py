import os
import requests
import numpy as np
import uvicorn
import subprocess
import sys
import re
import json
import sqlite3
from dateutil.parser import parse
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# OpenAI Proxy Configuration
OPENAI_API_URL_COMPLETIONS = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
OPENAI_API_URL_EMBEDDINGS = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
COMPLETION_MODEL = "gpt-4o-mini"
EMBEDDINGS_MODEL = "text-embedding-3-small"
API_KEY = os.environ["AIPROXY_TOKEN"]

# Operation task summaries
summaries = {
    1: "Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with $\{user.email\} as the only argument. (NOTE: This will generate data files required for the next tasks.)",
    2: "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place",
    3: "Counting number of days.",
    4: "Sort contacts by last and first name",
    5: "Extract first lines from the 10 most recent logs.",
    6: "Extract H1 headings from Markdown files.",
    7: "Extract sender’s email from an email file.",
    8: "Extract credit card number from an image.",
    9: "Find the most similar comments using embeddings.",
    10: "Calculate total sales for Gold concert tickets."
}

class TaskRequest(BaseModel):
    task: str

# Task Functions
def task_1(task_description: str):
    # Install uv if not already installed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])

    # Run datagen.py with the user's email as an argument
    user_email = os.environ.get("USER_EMAIL")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not provided")

    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    subprocess.run([sys.executable, "-m", "uv", datagen_url, user_email], check=True)

def task_2(task_description: str):
    # Extract file name and version using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the file name and version from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "version": {"type": "string"}
                        },
                        "required": ["file_name", "version"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    file_name = file_info.get("file_name")
    version = file_info.get("version")

    if not file_name or not version:
        raise HTTPException(status_code=400, detail="File name or version not provided")

    # Format the file using prettier
    file_path = os.path.join("/data", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    subprocess.run(["prettier", "--write", file_path], check=True)

def task_3(task_description: str):
    print("Task 3")
    # ##Using "function call" feature of open ai, extract the "day", "original file name" and the 
    # "new file name" which may or may not be given from the "query" string. The "original file name" 
    # has dates written within the file,  and count the number dates which have the same day as returned 
    # from the open ai function call. The format of the date may be different and would need to be identified from the file.
    # Extract day, original file name, and new file name using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the day, original file name, and new file name from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string"},
                            "original_file_name": {"type": "string"},
                            "new_file_name": {"type": "string"}
                        },
                        "required": ["day", "original_file_name"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    day = file_info.get("day")
    original_file_name = file_info.get("original_file_name")
    new_file_name = file_info.get("new_file_name")
    print(day, original_file_name, new_file_name)

    if not day or not original_file_name:
        raise HTTPException(status_code=400, detail="Day or original file name not provided")

    # Read the original file and count the number of dates with the same day
    file_path = os.path.join("/data", original_file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    date_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
    dates = date_pattern.findall(content)
    day_count = sum(1 for date in dates if parse(date).strftime("%A") == day)

    # Write the day count to the new file if provided
    if new_file_name:
        new_file_path = os.path.join("/data", new_file_name)
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(f"{day_count}")

    return {"day": day, "original_file_name": original_file_name, "new_file_name": new_file_name, "day_count": day_count}

def task_4(task_description: str):
    # Extract original file name and new file name using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the original file name and new file name from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "original_file_name": {"type": "string"},
                            "new_file_name": {"type": "string"}
                        },
                        "required": ["original_file_name"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    original_file_name = file_info.get("original_file_name")
    new_file_name = file_info.get("new_file_name")

    if not original_file_name:
        raise HTTPException(status_code=400, detail="Original file name not provided")

    # Read the original file and sort contacts
    file_path = os.path.join("/data", original_file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found")

    with open(file_path, "r", encoding="utf-8") as f:
        contacts = [line.strip().split(",") for line in f.readlines()]

    # Sort contacts by last name and then first name
    contacts.sort(key=lambda x: (x[1], x[0]))

    # Write sorted contacts to the new file or overwrite the original file
    output_file_path = os.path.join("/data", new_file_name) if new_file_name else file_path
    with open(output_file_path, "w", encoding="utf-8") as f:
        for contact in contacts:
            f.write(",".join(contact) + "\n")

    return {"original_file_name": original_file_name, "new_file_name": new_file_name, "status": "sorted"}

def task_5(task_description: str):
    # Extract extension, folder, and new file name using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the extension, folder, and new file name from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extension": {"type": "string"},
                            "folder": {"type": "string"},
                            "new_file_name": {"type": "string"}
                        },
                        "required": ["folder"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    extension = file_info.get("extension", ".log")
    folder = file_info.get("folder")
    new_file_name = file_info.get("new_file_name", "logs_recent.txt")

    if not folder:
        raise HTTPException(status_code=400, detail="Folder not provided")

    # Get the 10 most recent log files
    log_files = [f for f in os.listdir(folder) if f.endswith(extension)]
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    recent_logs = log_files[:10]

    # Extract the first line from each log file
    first_lines = []
    for log_file in recent_logs:
        log_file_path = os.path.join(folder, log_file)
        with open(log_file_path, "r", encoding="utf-8") as f:
            first_lines.append(f.readline().strip())

    # Write the first lines to the new file
    new_file_path = os.path.join(folder, new_file_name)
    with open(new_file_path, "w", encoding="utf-8") as f:
        for line in first_lines:
            f.write(line + "\n")

    return {"folder": folder, "new_file_name": new_file_name, "status": "first lines extracted"}

def task_6(task_description: str):
    # Extract directory using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_directory_info",
                    "description": "Extracts the directory containing markdown files from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"}
                        },
                        "required": ["directory"]
                    }
                }
            ],
            "function_call": {"name": "extract_directory_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract directory info")

    directory_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    directory = directory_info.get("directory")

    if not directory:
        raise HTTPException(status_code=400, detail="Directory not provided")

    # Extract H1 headings from markdown files
    index = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            relative_path = os.path.relpath(file_path, "/data/docs")
                            index[relative_path] = title
                            break

    # Write the index to a JSON file
    index_file_path = "/data/docs/index.json"
    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    return {"directory": directory, "status": "index created"}

def task_7(task_description: str):
    # Extract email file path and new directory path using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the email file path and new directory path from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_file_path": {"type": "string"},
                            "new_directory_path": {"type": "string"}
                        },
                        "required": ["email_file_path", "new_directory_path"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    email_file_path = file_info.get("email_file_path")
    new_directory_path = file_info.get("new_directory_path")

    if not email_file_path or not new_directory_path:
        raise HTTPException(status_code=400, detail="Email file path or new directory path not provided")

    # Read the email file content
    if not os.path.exists(email_file_path):
        raise HTTPException(status_code=404, detail="Email file not found")

    with open(email_file_path, "r", encoding="utf-8") as f:
        email_content = f.read()

    # Extract the sender's email address using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": email_content}],
            "functions": [
                {
                    "name": "extract_email_address",
                    "description": "Extracts the sender's email address from the email content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_address": {"type": "string"}
                        },
                        "required": ["email_address"]
                    }
                }
            ],
            "function_call": {"name": "extract_email_address"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract email address")

    email_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    email_address = email_info.get("email_address")

    if not email_address:
        raise HTTPException(status_code=400, detail="Email address not extracted")

    # Write the email address to the new directory path
    new_file_path = os.path.join(new_directory_path, "sender_email.txt")
    os.makedirs(new_directory_path, exist_ok=True)
    with open(new_file_path, "w", encoding="utf-8") as f:
        f.write(email_address)

    return {"email_file_path": email_file_path, "new_directory_path": new_directory_path, "status": "email extracted"}

def task_8(task_description: str):
    # Extract credit card path and output path using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the credit card path and output path from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "credit_card_path": {"type": "string"},
                            "output_path": {"type": "string"}
                        },
                        "required": ["credit_card_path"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    credit_card_path = file_info.get("credit_card_path")
    output_path = file_info.get("output_path", "/data/credit-card.txt")

    if not credit_card_path:
        raise HTTPException(status_code=400, detail="Credit card path not provided")

    # Read the image content
    if not os.path.exists(credit_card_path):
        raise HTTPException(status_code=404, detail="Credit card image not found")

    with open(credit_card_path, "rb") as f:
        image_content = f.read()

    # Extract the credit card number using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": "Extract the credit card number from the image"}],
            "functions": [
                {
                    "name": "extract_credit_card_number",
                    "description": "Extracts the credit card number from the image content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "credit_card_number": {"type": "string"}
                        },
                        "required": ["credit_card_number"]
                    }
                }
            ],
            "function_call": {"name": "extract_credit_card_number"},
            "input": image_content
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract credit card number")

    card_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    credit_card_number = card_info.get("credit_card_number")

    if not credit_card_number:
        raise HTTPException(status_code=400, detail="Credit card number not extracted")

    # Write the credit card number to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(credit_card_number)

    return {"credit_card_path": credit_card_path, "output_path": output_path, "status": "credit card number extracted"}

def task_9(task_description: str):
    # Extract comments file path and output file path using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the comments file path and output file path from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "comments_file_path": {"type": "string"},
                            "output_file_path": {"type": "string"}
                        },
                        "required": ["comments_file_path", "output_file_path"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    comments_file_path = file_info.get("comments_file_path")
    output_file_path = file_info.get("output_file_path")

    if not comments_file_path or not output_file_path:
        raise HTTPException(status_code=400, detail="Comments file path or output file path not provided")

    # Read the comments from the file
    if not os.path.exists(comments_file_path):
        raise HTTPException(status_code=404, detail="Comments file not found")

    with open(comments_file_path, "r", encoding="utf-8") as f:
        comments = f.readlines()

    # Calculate embeddings for each comment
    comment_embeddings = [get_embeddings(comment.strip()) for comment in comments]

    # Find the most similar pair of comments
    max_similarity = -1
    most_similar_pair = (None, None)
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(comment_embeddings[i], comment_embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (comments[i].strip(), comments[j].strip())

    # Write the most similar pair of comments to the output file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(most_similar_pair[0] + "\n")

def task_10(task_description: str):
    # Extract input file name and output file name using OpenAI function calling
    response = requests.post(
        OPENAI_API_URL_COMPLETIONS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": task_description}],
            "functions": [
                {
                    "name": "extract_file_info",
                    "description": "Extracts the input file name and output file name from the task description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input_file_name": {"type": "string"},
                            "output_file_name": {"type": "string"}
                        },
                        "required": ["input_file_name", "output_file_name"]
                    }
                }
            ],
            "function_call": {"name": "extract_file_info"}
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to extract file info")

    file_info = response.json().get("choices")[0].get("message").get("function_call").get("arguments")
    input_file_name = file_info.get("input_file_name")
    output_file_name = file_info.get("output_file_name")

    if not input_file_name or not output_file_name:
        raise HTTPException(status_code=400, detail="Input file name or output file name not provided")

    # Connect to the SQLite database and calculate total sales for Gold tickets

    input_file_path = os.path.join("/data", input_file_name)
    if not os.path.exists(input_file_path):
        raise HTTPException(status_code=404, detail="Input file not found")

    conn = sqlite3.connect(input_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]
    conn.close()

    # Write the total sales to the output file
    output_file_path = os.path.join("/data", output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(str(total_sales))

    return {"input_file_name": input_file_name, "output_file_name": output_file_name, "total_sales": total_sales}

task_functions = {
    1: task_1,
    2: task_2,
    3: task_3,
    4: task_4,
    5: task_5,
    6: task_6,
    7: task_7,
    8: task_8,
    9: task_9,
    10: task_10
}

# Function to get embeddings
def get_embeddings(text: str):
    response = requests.post(
        OPENAI_API_URL_EMBEDDINGS,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": EMBEDDINGS_MODEL, "input": text}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Embedding generation failed")
    
    return response.json()["data"][0]["embedding"]

# Pre-calculate embeddings for summaries
summary_embeddings = {task_id: get_embeddings(summary) for task_id, summary in summaries.items()}

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Task Execution Function
def execute_task(task_description: str):
    task_embedding = get_embeddings(task_description)
    similarities = {}
    
    for task_id, summary in summaries.items():
        summary_embedding = get_embeddings(summary)
        similarity = cosine_similarity(task_embedding, summary_embedding)
        similarities[task_id] = similarity
    
    best_match = max(similarities, key=similarities.get)
    print(best_match)
    return
    task_function = task_functions[best_match]
    task_function(task_description)
    
    return {"status": "success", "task_id": best_match, "details": summaries[best_match]}

@app.post("/run")
def run(request: TaskRequest):
    try:
        result = execute_task(request.task)
        return result
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
def read(path: str = Query(..., description="Path of the file to read")):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=403, detail="Access to this path is restricted")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(path, "r", encoding="utf-8") as f:
        return {"content": f.read()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
