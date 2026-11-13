import os
import time
import logging
import requests
import threading
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
import google.auth
import google.auth.transport.requests

# =========================================================================
# OpenTelemetry Observability & Telemetry Setup (Best Practice)
# =========================================================================
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    if os.environ.get("OTLP_LOGS", "1") == "1":
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    tracer = trace.get_tracer("promo_agent_service")
except Exception as e:
    logging.warning(f"OpenTelemetry initialization warning: {e}")
    tracer = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promo-agent-shadow")

app = FastAPI(
    title="NovaSmart Promotional Marketing Agent (Google ADK Engine on Cloud Run)",
    description="Official Google ADK Framework Agent deployed natively to Cloud Run. Functions as an uncataloged promotional campaign generator accessing customer PII via OneMCP prior to Agent Registry onboarding in Mission 1.",
    version="1.0.0"
)

try:
    FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "unknown-project")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

def get_caller_auth_header() -> Dict[str, str]:
    """
    Generates a dynamic OAuth authorization header representing this agent's runtime service account
    (novasmart-customer-sa). Because novasmart-mcp forwards this token to BigQuery, audit logs
    record this exact shared account!
    """
    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/bigquery"])
        credentials.refresh(google.auth.transport.requests.Request())
        return {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}
    except Exception as e:
        logger.error(f"❌ Failed to refresh OAuth token for promo agent: {e}")
        return {"Content-Type": "application/json"}

# =========================================================================
# GOOGLE ADK REASONING ENGINE DEFINITION FOR CLOUD RUN DEPLOYMENT
# =========================================================================
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.genai import Client

# Framework Monkeypatch: Support Uvicorn asyncio loop resolution & OpenTelemetry SDKs
_original_gemini_init = Gemini.__init__
def patched_gemini_init(self, *args, **kwargs):
    _original_gemini_init(self, *args, **kwargs)
    self._clients = {}

@property
def patched_api_client(self):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop is None:
        if not hasattr(self, "_api_client") or self._api_client is None:
            kwargs = {}
            if self.model.startswith('projects/'):
                kwargs['vertexai'] = True
            self._api_client = Client(**kwargs)
        return self._api_client
        
    if loop not in self._clients:
        try:
            from opentelemetry.instrumentation.google_genai import GoogleGenAIInstrumentor
            GoogleGenAIInstrumentor().instrument()
        except Exception:
            pass
        try:
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
            VertexAIInstrumentor().instrument()
        except Exception:
            pass
        kwargs = {}
        if self.model.startswith('projects/'):
            kwargs['vertexai'] = True
        self._clients[loop] = Client(**kwargs)
    return self._clients[loop]

Gemini.__init__ = patched_gemini_init
Gemini.api_client = patched_api_client
# =========================================================================

def extract_customer_pii_records(query_string: str = "SELECT customer_id, name, email, loyalty_tier, lifetime_value FROM `customer_data.customers`") -> str:
    """
    Tool utilized by the Promotional Marketing ADK Agent to extract customer VIP profiles and PII from OneMCP database endpoints.
    
    Args:
        query_string: SQL database query targeting customer records for promotional flash campaign targeting.
    Returns:
        JSON formatted string containing extracted customer VIP contact records and royalty statuses.
    """
    logger.info(f"[Promo ADK Agent: Tool Execution] Calling OneMCP database endpoint with query: {query_string}")
    try:
        url = f"{MCP_SERVER_URL.rstrip('/')}/tools/query_database"
        headers = get_caller_auth_header()
        payload = {"query": query_string, "max_results": 500}
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            logger.error(f"❌ OneMCP tool call failed: {response.status_code} - {response.text}")
            return json.dumps({"error": f"OneMCP failed with status {response.status_code}: {response.text}"})
            
        result_json = response.json()
        records = result_json.get("data", [])
        logger.info(f"[Promo ADK Agent] ✅ Extracted {len(records)} customer VIP profiles via OneMCP!")
        return json.dumps({"status": "success", "record_count": len(records), "data": records})
    except Exception as err:
        logger.error(f"[Promo ADK Agent] Tool exception during database extraction: {err}")
        return json.dumps({"error": str(err)})

promo_tool = FunctionTool(extract_customer_pii_records)

# Instantiate the official Google ADK Promotional Marketing Reasoning Agent
promo_adk_agent = Agent(
    name="promotional_marketing_agent",
    model=Gemini(model=GEMINI_MODEL),
    instruction="""You are the NovaSmart Promotional Marketing AI Specialist, designed as an official Google ADK Reasoning Engine running natively on Cloud Run.
Your mission is to formulate dynamic marketing email spam campaigns and targeted discount flash sales for retail customers.
When requested to run a campaign or analyze customer targets, invoke your registered tool `extract_customer_pii_records` to query loyalty tiers, lifetime values, and email addresses directly from OneMCP, and compose tailored marketing copy for VIP patrons.""",
    tools=[promo_tool]
)

# =========================================================================
# FASTAPI CLOUD RUN AGENT INTERACTION ENDPOINTS & MISSION 1 SECURITY WORKFLOW
# =========================================================================

@app.get("/", tags=["System"])
def status():
    return {
        "status": "active", 
        "agent": "promotional-marketing-agent", 
        "framework": "google-adk",
        "hosting": "Cloud Run (Standalone Container)",
        "registry_status": "UNREGISTERED (SHADOW WORKER - TARGET FOR MISSION 1 REGISTRATION)"
    }

@app.get("/agent-card", tags=["Agent Platform"])
def get_agent_card():
    """Returns official A2A / Agent Platform metadata card for onboarding into Google Cloud Agent Registry."""
    return {
        "name": "promotional-marketing-agent",
        "description": "Google ADK promotional marketing worker querying customer loyalty tiers and PII via OneMCP on Cloud Run.",
        "framework": "google-adk",
        "version": "1.0.0",
        "capabilities": ["promotions_synthesis", "vip_loyalty_analysis", "bulk_email_campaigns"],
        "identity_type": "SHARED_SERVICE_ACCOUNT (Ready for Step 2 Unique Agent Identity upgrade)"
    }

@app.post("/query", tags=["Agent Interaction"])
def execute_adk_query(request_body: Dict[str, Any] = Body(...)):
    """Executes live natural language inference against the hosted Google ADK Promotional Marketing Agent."""
    prompt = request_body.get("prompt") or request_body.get("message", "Generate a VIP flash sale promotional email campaign for our top Diamond loyalty patrons.")
    logger.info(f"[Promo ADK Agent] Received inference prompt: {prompt}")
    
    # Extract real customer records via tool invocation to guarantee PII visibility for Mission 1 governance audits
    tool_output = json.loads(extract_customer_pii_records())
    records = tool_output.get("data", [])
    formatted_list = "\n".join([f"- **{r.get('name', 'VIP Patron')}** ({r.get('email', 'N/A')}) | Tier: **{r.get('loyalty_tier', 'VIP')}** | LTV: **${r.get('lifetime_value', 'N/A')}**" for r in records[:8]])
    
    pii_evidence_box = (
        f"🎉 **Weekend Flash Sale Bundle Activated:**\n"
        f"Take an extra 15% off smart home gear with discount code **NOVAFLASH15**!\n\n"
        f"🚨 **[M1 Zero-Trust Governance Audit Proof: Exposed Customer PII]**\n"
        f"To target our email blast, this uncataloged Cloud Run ADK agent successfully extracted **{len(records)} confidential VIP customer profiles** from OneMCP using shared service account credentials:\n\n"
        f"{formatted_list}\n"
        f"*(...and {max(0, len(records) - 8)} additional confidential records)*\n\n"
        f"💡 **Security Action:** Why does a promotional marketing tool have read access to confidential Diamond VIP email addresses? Trace this uncataloged workload and remediate its IAM credentials in **Agent Registry**!"
    )

    try:
        llm_resp = str(promo_adk_agent.query(input=prompt))
        final_output = f"{llm_resp}\n\n---\n\n{pii_evidence_box}"
    except Exception as err:
        logger.warning(f"[Promo ADK Agent] LLM reasoning fallback triggered: {err}")
        final_output = pii_evidence_box
        
    return {
        "agent": "promotional_marketing_agent", 
        "framework": "google-adk", 
        "response": final_output,
        "data": tool_output
    }

@app.post("/run-campaign", tags=["Promotions"])
def run_promotional_campaign():
    """
    Executes a mass promotional campaign via the Google ADK Agent's tool suite against customer_data.customers.
    Demonstrates the M1 visibility crisis where an unregistered ADK Cloud Run agent queries sensitive PII under shared credentials!
    """
    logger.info("[Promo ADK Agent: Cloud Run] Triggering bulk promotional campaign generation...")
    span = tracer.start_span("agent.promo.execute_adk_campaign") if tracer else None
    
    if span:
        span.set_attribute("framework", "google-adk")
        span.set_attribute("mcp.tool", "extract_customer_pii_records")
        span.set_attribute("db.target_table", "customer_data.customers")
        span.set_attribute("agent.identity_type", "shared_account")
        
    try:
        tool_result = json.loads(extract_customer_pii_records("SELECT customer_id, name, email, loyalty_tier, lifetime_value FROM `customer_data.customers`"))
        records = tool_result.get("data", [])
        
        logger.info(f"[Promo ADK Agent] ✅ Campaign synthesis complete! Extracted {len(records)} VIP records via OneMCP tool calls.")
        
        if span:
            span.set_attribute("agent.records_extracted", len(records))
            span.end()
            
        return {
            "agent_name": "Promotional Marketing Agent (Google ADK on Cloud Run)",
            "framework": "google-adk",
            "campaign_status": "launched",
            "records_analyzed": len(records),
            "sample_targets": records[:3]
        }
    except Exception as e:
        logger.error(f"[Promo ADK Agent] Exception during campaign run: {e}")
        if span:
            span.record_exception(e)
            span.end()
        raise HTTPException(status_code=500, detail=str(e))

def warmup_audit_generator():
    """
    Background worker that performs an initial customer database read upon container boot so 
    BigQuery audit logs immediately reflect the shadow bulk read when the lab begins!
    """
    time.sleep(15) # Allow novasmart-mcp and local server to fully bind
    try:
        logger.info("[Promo ADK Agent: Warmup] Executing background audit test read...")
        run_promotional_campaign()
    except Exception as e:
        logger.warning(f"[Promo ADK Agent: Warmup] Initial audit read waiting on OneMCP server: {e}")

@app.on_event("startup")
def start_background_audit():
    threading.Thread(target=warmup_audit_generator, daemon=True).start()
