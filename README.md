# MediScan

## Description

In critical situations such as sudden illness or natural disasters, accessing a patient's medical history swiftly can be a matter of life and death. The MediScan project aims to revolutionize medical emergency response by providing instant access to a patient's medical history through facial recognition technology. By simply scanning a patient's face, doctors can access vital medical information, allergies, and previous treatments, enabling them to make informed decisions rapidly and administer appropriate treatment promptly.

## Workflow

![MediScan Workflow](workflow.png)

## Running the MediScan Project

1. Navigate to the MediScan directory:
    ```bash
    cd MediScan
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - For Windows:
        ```bash
        venv\Scripts\Activate
        ```
    - For Linux:
        ```bash
        source venv/bin/activate
        ```

4. Install project dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. Change to the `frontend` directory:
    ```bash
    cd frontend
    ```

6. Run the backend server:
    ```bash
    python manage.py runserver
    ```

7. Open another terminal and change to the `MediScan` directory:

    ```bash
    cd MediScan
    ```

8. Run the frontend:

    ```bash
    npm run build
    ```

9. Open any web browser and go to:

    ```bash
    localhost:8000
    ```

Now, your MediScan project should be up and running!
