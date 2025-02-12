import requests

TABLEAU_SERVER_URL = "https://your-tableau-server.com"
TABLEAU_API_TOKEN = "your-api-token"

def export_to_tableau(predicted_class):
    """
    Sends defect classification data to Tableau.
    """
    payload = {"defect": predicted_class}
    try:
        response = requests.post(f"{TABLEAU_SERVER_URL}/update_dashboard", json=payload, headers={"Authorization": f"Bearer {TABLEAU_API_TOKEN}"})
        return "Exported to Tableau ✅" if response.status_code == 200 else "Tableau Export Failed ❌"
    except Exception as e:
        return f"Tableau Error: {str(e)}"

