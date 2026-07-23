"""
execution/dummy_target_agent.py

A minimal fake "agent under test" so you can run the full real pipeline
(generator -> dedup -> real harness -> real judge -> causal report)
end-to-end locally, before pointing it at an actual agent like SupportSense.

Run this in its own terminal window (separate from the one running your
audit), and leave it running:

    python execution\\dummy_target_agent.py

Then in a SECOND terminal (with venv activated), run:

    python -c "from agents.graph import run_audit; state = run_audit(n_per_category=10, mode='real', target_endpoint='http://localhost:8899'); print(state['report'])"

This dummy agent deliberately behaves a bit badly on purpose (occasionally
ignores the request, occasionally errors) so you'll see the judge actually
catch some failures instead of everything trivially passing.
"""
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler


class DummyAgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        user_input = body.get("input", "")

        roll = random.random()
        if roll < 0.10:
            # Simulate an agent that occasionally errors out entirely.
            self.send_response(500)
            self.end_headers()
            return
        elif roll < 0.30:
            # Simulate an agent that ignores the actual request.
            output = "Thanks for reaching out! Have a great day."
            tool_used = "generic_reply"
        else:
            # Simulate a mostly-reasonable response.
            output = f"Got it - here's help with: {user_input}"
            tool_used = "kb_lookup"

        response = {"output": output, "tool_used": tool_used}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        print(f"[dummy agent] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8899), DummyAgentHandler)
    print("Dummy target agent running on http://localhost:8899 - Ctrl+C to stop.")
    server.serve_forever()
