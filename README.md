# 🚀 My Streamlit Web App

A [Streamlit](https://streamlit.io/) web app for *AI Copilot for data teams*.

---

## 📦 Requirements

- Python *3.9+*
- [pip](https://pip.pypa.io/en/stable/)

---

## ⚙ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo.git
```
This will download the project files to your local machine.
```bash
cd Confluentia_TeamInnov8
```
Navigate into the project folder so that you can run commands in the correct directory.

### 2. Creating Virtual environment(optional)
```bash
python -m venv venv
```
This creates an isolated Python environment to avoid conflicts with other projects.
```bash
source venv/bin/activate     # On Linux/Mac
```
Activate the virtual environment on Linux or Mac. Your terminal prompt should now show (venv) at the start.
```bash
venv\Scripts\activate        # On Windows
```
Activate the virtual environment on Windows. All Python packages installed now will stay local to this project.
### 3. Installing dependencies
```bash
pip install -r requirements.txt
```
This installs all the necessary Python libraries required to run the app, including Streamlit and any AI-related packages.

### 4. Configuring .env file
```bash
GROQ_API_KEY=your_api_key_here
```
Replace <your_api_key_here> with your actual API key. This allows the app to connect to the AI backend securely.
### 5. Running the App
```bash
streamlit run app.py
```
This will launch the web app in your default browser. You can now interact with the AI Copilot interface directly.
