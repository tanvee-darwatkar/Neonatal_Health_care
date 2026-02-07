"""
Unified Dashboard - Shows both Motion Detection and Cry Detection
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

def get_status_icon(status):
    """Get icon based on status"""
    status_lower = status.lower()
    if status_lower in ['distress', 'unsafe', 'critical']:
        return "🔴"
    elif status_lower in ['abnormal', 'monitor', 'warning']:
        return "🟡"
    else:
        return "🟢"

def display_unified_dashboard(data):
    clear_screen()
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🏥 NEONATAL MONITORING SYSTEM" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Patient Info
    patient = data['patient']
    print(f"\n👶 PATIENT: {patient['id']} | Age: {patient['age']} | Status: {patient['status']}")
    
    print("\n" + "─" * 80)
    print("                          🔊 CRY DETECTION")
    print("─" * 80)
    
    # Cry Detection
    cry = data['cryDetection']
    cry_icon = get_status_icon(cry['status'])
    print(f"{cry_icon} Status: {cry['status'].upper()}")
    print(f"   Type: {cry['cryType']}")
    print(f"   Confidence: {cry['confidence']}%")
    print(f"   Intensity: {cry['intensity']}/100")
    print(f"   Last Detected: {cry['lastDetected']}")
    
    print("\n" + "─" * 80)
    print("                        📹 MOTION MONITORING")
    print("─" * 80)
    
    # Motion Monitoring
    motion = data['motionMonitoring']
    motion_icon = get_status_icon(motion['status'])
    print(f"{motion_icon} Status: {motion['status']}")
    print(f"   Still Time: {motion['stillTime']}s")
    print(f"   Motion Level: {motion['motion']}")
    print(f"   Confidence: {motion['confidence']}%")
    print(f"   Alert Active: {'YES' if motion['alertActive'] else 'NO'}")
    
    print("\n" + "─" * 80)
    print("                          💓 VITAL SIGNS")
    print("─" * 80)
    
    # Vitals
    for vital in data['vitals']:
        status_icon = "✅" if vital['status'] == 'normal' else "⚠️"
        print(f"{status_icon} {vital['title']}: {vital['value']} {vital['unit']} (Normal: {vital['normalRange']})")
    
    # Risk Assessment
    risk = data['riskAssessment']
    risk_icon = "🟢" if risk['overall'] == 'low' else "🟡" if risk['overall'] == 'medium' else "🔴"
    print(f"\n{risk_icon} OVERALL RISK: {risk['overall'].upper()} (Confidence: {risk['confidence']}%)")
    
    # Recent Alerts
    if data['alerts']:
        print("\n" + "─" * 80)
        print("                        🚨 RECENT ALERTS")
        print("─" * 80)
        for i, alert in enumerate(data['alerts'][:5], 1):
            alert_icon = "🔴" if alert['type'] == 'critical' else "🟡"
            print(f"{i}. {alert_icon} [{alert['timestamp']}] {alert['message']}")
    
    print("\n" + "═" * 80)
    print("Press Ctrl+C to exit | Refreshing every 3 seconds...")
    print("═" * 80)

def main():
    print("🚀 Starting Unified Neonatal Monitoring Dashboard...")
    print("📡 Connecting to server at http://127.0.0.1:5000")
    time.sleep(1)
    
    try:
        while True:
            data = get_dashboard_data()
            if data:
                display_unified_dashboard(data)
            else:
                print("❌ Failed to connect to server. Make sure run_simple_server.py is running.")
            
            time.sleep(3)
    
    except KeyboardInterrupt:
        clear_screen()
        print("\n👋 Dashboard stopped. Thank you!")

if __name__ == "__main__":
    main()
