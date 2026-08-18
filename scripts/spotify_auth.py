"""One-time Spotify authorization (Authorization Code flow).

You create the app and click "Agree"; your secret stays in your environment and
never touches this repo. Reads SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET (and
optional SPOTIFY_REDIRECT_URI) from the environment, opens the consent page,
catches the redirect on a local port, and saves the tokens to
data/.spotify_token.json (gitignored).

Setup:
  1. https://developer.spotify.com/dashboard -> Create app.
  2. Add Redirect URI EXACTLY: http://127.0.0.1:8888/callback
  3. export SPOTIFY_CLIENT_ID=... ; export SPOTIFY_CLIENT_SECRET=...
  4. python scripts/spotify_auth.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

REDIRECT = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SCOPE = "user-top-read"
TOKEN_FILE = Path("data/.spotify_token.json")
_code_holder: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        q = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(q)
        _code_holder["code"] = (params.get("code") or [None])[0]
        _code_holder["error"] = (params.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Authorized. You can close this tab and return to the terminal.</h2>")

    def log_message(self, *a):  # silence
        pass


def main() -> None:
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your environment first.")

    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": cid, "scope": SCOPE,
        "redirect_uri": REDIRECT, "show_dialog": "false",
    })
    host, port = urllib.parse.urlparse(REDIRECT).hostname, urllib.parse.urlparse(REDIRECT).port or 80
    server = HTTPServer((host, port), _Handler)
    print("Opening browser for Spotify consent… approve, then return here.")
    webbrowser.open(auth_url)
    server.handle_request()  # blocks until the redirect hits /callback

    if _code_holder.get("error") or not _code_holder.get("code"):
        sys.exit(f"Authorization failed: {_code_holder.get('error')}")

    basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    r = requests.post("https://accounts.spotify.com/api/token",
                      data={"grant_type": "authorization_code", "code": _code_holder["code"],
                            "redirect_uri": REDIRECT},
                      headers={"Authorization": f"Basic {basic}"}, timeout=15)
    r.raise_for_status()
    tok = r.json()
    TOKEN_FILE.parent.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"refresh_token": tok["refresh_token"],
                                      "access_token": tok["access_token"]}))
    print(f"Saved tokens -> {TOKEN_FILE}. You can now run scripts/build_music.py.")


if __name__ == "__main__":
    main()
