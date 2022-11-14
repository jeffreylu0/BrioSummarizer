# Setup

1. Clone repo and start model API

    Without Docker:
    ```
    python -m venv venv/
    source venv/bin/activate
    pip install -r requirements.txt
    python -m app.main
    ```

    With Docker:
    ```
    docker compose up
    ```
2. Once API is running, go to the Chrome Extensions tab and upload the extension folder into the 'Load unpacked' section on the upper left
