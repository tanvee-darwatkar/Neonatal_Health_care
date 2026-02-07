"""
Demo client to show the Neonatal Cry Detection System in action
"""

import urllib.request
import json
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_dashboard_data():
    try:
        response = urllib.request.urlopen('http://127.0.0.1:5000/api/dashboard')
        return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return None

def display_dashboard(data):
    clear_screen()
    
    print("=" * 70)
    print("🏥 NEONATAL CRY DETECTION SYSTEM - LIVE DASHBOARD")
    print("=" * 70)
    
    # Patient Info
    patient = data['patient']
    print(f"\n👶 PATIENT: {patient['id']} | Age: {patient['age']} | Status: {patient['status']}")
    
    # Cry Detection
    cry = data['cryDetection']
    status_icon = "🔴" if cry['status'] == 'distress' else "🟢"
    print(f"\n{status_icon} CRY DETECTION:")
    print(f"   Status: {cry['status'].upper()}")
    print(f"   Type: {cry['cryType']}")
    print(f"   Confidence: {cry['confidence']}%")
    print(f"   Intensity: {cry['intensity']}/100")
    print(f"   Last Detected: {cry['lastDetected']}")
    
    # Vitals
    print(f"\n💓 VITAL SIGNS:")
    for vital in data['vitals']:
        status_icon = "✅" if vital['status'] == 'normal' else "⚠️"
        print(f"   {status_icon} {vital['title']}: {vital['value']} {vital['unit']} (Normal: {vital['normalRange']})")
    
    # Alerts
    if data['alerts']:
        print(f"\n🚨 RECENT ALERTS:")
        for alert in data['alerts'][:3]:
            alert_icon = "🔴" if alert['type'] == 'critical' else "🟡"
            print(f"   {alert_icon} [{alert['timestamp']}] {alert['message']}")
    
    # Risk Assessment
    risk = data['riskAssessment']
    risk_color = "🟢" if risk['overall'] == 'low' else "🟡" if risk['overall'] == 'medium' else "🔴"
    print(f"\n{risk_color} RISK ASSESSMENT: {risk['overall'].upper()} (Confidence: {risk['confidence']}%)")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit | Refreshing every 3 seconds...")
    print("=" * 70)

def main():
    print("🚀 Starting Neonatal Cry Detection Demo Client...")
    print("📡 Connecting to server at http://127.0.0.1:5000")
    time.sleep(1)
    
    try:
        while True:
            data = get_dashboard_data()
            if data:
                display_dashboard(data)
            else:
                print("❌ Failed to connect to server. Make sure run_simple_server.py is running.")
            
            time.sleep(3)
    
    except KeyboardInterrupt:
        clear_screen()
        print("\n👋 Demo client stopped. Thank you!")

if __name__ == "__main__":
    main()
