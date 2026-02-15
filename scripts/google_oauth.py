#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-auth>=2.0.0",
#   "google-auth-oauthlib>=1.0.0",
#   "google-api-python-client>=2.0.0",
# ]
# ///
"""
Syne Google OAuth - Headless authorization for Google APIs

Setup (one-time):
1. Go to https://console.cloud.google.com/apis/credentials
2. Create project if needed, name it "Syne"
3. Configure OAuth consent screen (External, just your email as test user)
4. Create OAuth 2.0 Client ID (Desktop app type)
5. Download JSON, save as persona/.google_credentials.json
6. Run: python sfa_google_oauth.py --setup
7. Visit the URL, authorize, paste the code
8. Refresh token saved to persona/.env

Usage after setup:
  python sfa_google_oauth.py --test          # Test all APIs
  python sfa_google_oauth.py --calendar      # List next 10 calendar events
  python sfa_google_oauth.py --gmail         # List recent emails
  python sfa_google_oauth.py --drive         # List recent drive files
"""

import argparse
import json
import os
from pathlib import Path

# Self-locate: persona/bin/*.py -> persona/
_SCRIPT_DIR = Path(__file__).parent.resolve()
PERSONA_HOME = Path(os.environ.get("PERSONA_HOME", _SCRIPT_DIR.parent.parent))

# Paths
CREDS_FILE = PERSONA_HOME / ".google_credentials.json"
TOKEN_FILE = PERSONA_HOME / ".google_token.json"
ENV_FILE = PERSONA_HOME / ".env"

# Scopes for Calendar, Gmail, Drive
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def setup_oauth():
    """One-time setup: Get refresh token via browser auth."""
    if not CREDS_FILE.exists():
        print(f"ERROR: {CREDS_FILE} not found")
        print("\nSetup instructions:")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create/select project named 'Syne'")
        print(
            "3. Configure OAuth consent screen (External, add your email as test user)"
        )
        print("4. Create OAuth 2.0 Client ID (Desktop app type)")
        print(f"5. Download JSON and save as {CREDS_FILE}")
        print("6. Run this script again with --setup")
        return False

    from google_auth_oauthlib.flow import InstalledAppFlow

    # Use out-of-band flow for headless
    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDS_FILE), scopes=SCOPES, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
    )

    auth_url, _ = flow.authorization_url(prompt="consent")

    print("\n=== Google OAuth Setup ===")
    print("\n1. Visit this URL in your browser:")
    print(f"\n{auth_url}\n")
    print("2. Sign in and authorize access")
    print("3. Copy the authorization code and paste it here:")

    code = input("\nAuthorization code: ").strip()

    try:
        flow.fetch_token(code=code)
        creds = flow.credentials

        # Save token
        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes),
        }

        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)

        # Also add refresh token to .env for easy access
        env_content = ENV_FILE.read_text() if ENV_FILE.exists() else ""
        if "GOOGLE_REFRESH_TOKEN=" not in env_content:
            with open(ENV_FILE, "a") as f:
                f.write(f"\nGOOGLE_REFRESH_TOKEN={creds.refresh_token}\n")

        print("\n✓ Authorization successful!")
        print(f"✓ Token saved to {TOKEN_FILE}")
        print(f"✓ Refresh token added to {ENV_FILE}")
        print("\nYou can now use --test, --calendar, --gmail, --drive")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False


def get_credentials():
    """Load credentials from token file, refresh if needed."""
    if not TOKEN_FILE.exists():
        print(f"ERROR: {TOKEN_FILE} not found. Run --setup first.")
        return None

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    with open(TOKEN_FILE) as f:
        token_data = json.load(f)

    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes"),
    )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Update saved token
        token_data["token"] = creds.token
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)

    return creds


def test_calendar():
    """Test Calendar API access."""
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        return False

    service = build("calendar", "v3", credentials=creds)

    import datetime

    now = datetime.datetime.utcnow().isoformat() + "Z"

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = events_result.get("items", [])

    if not events:
        print("No upcoming events found.")
    else:
        print("\n=== Next 10 Calendar Events ===")
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(f"  {start}: {event['summary']}")

    return True


def test_gmail():
    """Test Gmail API access."""
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        return False

    service = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(userId="me", maxResults=10).execute()

    messages = results.get("messages", [])

    if not messages:
        print("No messages found.")
    else:
        print("\n=== Recent 10 Emails ===")
        for msg in messages:
            msg_detail = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From"],
                )
                .execute()
            )

            headers = {h["name"]: h["value"] for h in msg_detail["payload"]["headers"]}
            subject = headers.get("Subject", "(no subject)")[:60]
            sender = headers.get("From", "unknown")[:40]
            print(f"  From: {sender}")
            print(f"  Subject: {subject}\n")

    return True


def test_drive():
    """Test Drive API access."""
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        return False

    service = build("drive", "v3", credentials=creds)

    results = (
        service.files()
        .list(pageSize=10, fields="files(id, name, mimeType, modifiedTime)")
        .execute()
    )

    files = results.get("files", [])

    if not files:
        print("No files found.")
    else:
        print("\n=== Recent 10 Drive Files ===")
        for f in files:
            print(f"  {f['name']}")
            print(f"    Type: {f['mimeType']}")
            print(f"    Modified: {f.get('modifiedTime', 'unknown')}\n")

    return True


def test_all():
    """Test all API access."""
    print("Testing Google API access...")

    print("\n--- Calendar ---")
    try:
        test_calendar()
        print("✓ Calendar: OK")
    except Exception as e:
        print(f"✗ Calendar: {e}")

    print("\n--- Gmail ---")
    try:
        test_gmail()
        print("✓ Gmail: OK")
    except Exception as e:
        print(f"✗ Gmail: {e}")

    print("\n--- Drive ---")
    try:
        test_drive()
        print("✓ Drive: OK")
    except Exception as e:
        print(f"✗ Drive: {e}")


def main():
    parser = argparse.ArgumentParser(description="Syne Google OAuth")
    parser.add_argument("--setup", action="store_true", help="One-time OAuth setup")
    parser.add_argument("--test", action="store_true", help="Test all APIs")
    parser.add_argument("--calendar", action="store_true", help="List calendar events")
    parser.add_argument("--gmail", action="store_true", help="List recent emails")
    parser.add_argument("--drive", action="store_true", help="List drive files")

    args = parser.parse_args()

    if args.setup:
        setup_oauth()
    elif args.test:
        test_all()
    elif args.calendar:
        test_calendar()
    elif args.gmail:
        test_gmail()
    elif args.drive:
        test_drive()
    else:
        parser.print_help()
        print("\n\nFirst time? Run with --setup")


if __name__ == "__main__":
    main()
