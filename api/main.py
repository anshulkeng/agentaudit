"""
Exposes the audit pipeline as an HTTP API. Run with:
    uvicorn api.main:app --reload --port 8000
"""
import uuid
from fastapi import FastAPI
from agents.graph import run_audit

app = FastAPI(title="AgentAudit")
AUDITS: dict = {}


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/audit")
def start_audit(n_per_category: int = 40):
    audit_id = str(uuid.uuid4())
    state = run_audit(n_per_category=n_per_category)
    AUDITS[audit_id] = {
        "overall_failure_rate": state["overall_failure_rate"],
        "causal_results": state["causal_results"],
        "report": state["report"],
        "num_cases": len(state["cases"]),
    }
    return {"audit_id": audit_id, "status": "complete"}


@app.get("/audit/{audit_id}")
def get_audit(audit_id: str):
    return AUDITS.get(audit_id, {"error": "not found"})
