Write-Host "Starting FastAPI Backend on port 8000..."
Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\uvicorn.exe" -ArgumentList "api:app", "--host", "0.0.0.0", "--port", "8000"

Start-Sleep -Seconds 3

Write-Host "Starting Streamlit Frontend on port 8501..."
Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\streamlit.exe" -ArgumentList "run", "app.py"

Write-Host "Both services are running in the background. Press Ctrl+C to exit this script (note: you may need to kill the processes manually if they persist)."
