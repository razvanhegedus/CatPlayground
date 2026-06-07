import smtplib
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_session_summary(duration_sec, date_str, diagnostic_text):
    """Compiles and transmits cat play session analytics via secure SMTP."""
    smtp_server = st.secrets["SMTP_SERVER"]
    port = int(st.secrets["SMTP_PORT"])
    sender = st.secrets["SENDER_EMAIL"]
    password = st.secrets["SENDER_PASSWORD"]
    receiver = st.secrets["RECEIVER_EMAIL"]

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🎯 CajuNet Project: Play Session Report - {date_str}"
    msg['From'] = sender
    msg['To'] = receiver

    html_template = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; color: #333; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; padding: 25px; border: 1px solid #e1e4e8;">
          <h2 style="color: #00FFCC; background-color: #0E1117; padding: 10px; border-radius: 4px; text-align: center;">🎯 CajuNet Automated Play Analytics</h2>
          <p>An autonomous tracking interaction cycle has finished processing. Review the log details gathered from the edge device below:</p>
          
          <p><b>Session Timestamp:</b> {date_str}</p>
          <p><b>Total Playtime Duration:</b> <span style="font-size: 1.2em; color: #00FFCC; font-weight: bold;">{duration_sec} seconds</span></p>

          <h3 style="color: #333; margin-top: 25px; border-bottom: 2px solid #eaeaea; padding-bottom: 5px;">📋 Operational Evaluation Summary</h3>
          <div style="background-color: #f1f3f5; border-left: 4px solid #00FFCC; padding: 15px; font-style: italic; white-space: pre-wrap;">
{diagnostic_text}
          </div>
          
          <p style="font-size: 0.85em; color: #777; margin-top: 30px; border-top: 1px solid #eaeaea; padding-top: 15px;">
            This diagnostic record was routed on-demand from your hosted cloud-bridge stream application.
          </p>
        </div>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(html_template, 'html'))

    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())