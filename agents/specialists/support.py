import json
import uuid
from agents.utils import transfer_to_triage
from mock_data.incidents import INCIDENTS, INCIDENT_CATEGORIES, runtime_incidents
from agents.constants import STAYFORLONG_CONTACT
import config
from agents.plugin import AgentPlugin


def lookup_incident(ticket_id: str) -> str:
    """Look up an existing support ticket by ticket ID (format: INC-XXX)."""
    ticket = INCIDENTS.get(ticket_id.upper()) or runtime_incidents.get(ticket_id.upper())
    if not ticket:
        return json.dumps({
            "found": False,
            "message": f"Ticket '{ticket_id}' not found.",
        })
    return json.dumps({
        "found": True,
        "ticket_id": ticket["ticket_id"],
        "category": ticket["category"],
        "description": ticket["description"],
        "status": ticket["status"],
        "priority": ticket["priority"],
        "created_at": ticket["created_at"],
        "resolved_at": ticket.get("resolved_at"),
        "assigned_to": ticket.get("assigned_to"),
        "notes": ticket.get("notes"),
    })


def create_incident(category: str, description: str, booking_id: str) -> str:
    """Create a new support ticket for an issue. Category must be one of: maintenance, noise, cleanliness, appliance, wifi, access, safety, billing, other."""
    if category not in INCIDENT_CATEGORIES:
        category = "other"

    ticket_id = f"INC-{str(uuid.uuid4())[:6].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "booking_id": booking_id,
        "category": category,
        "description": description,
        "status": "open",
        "priority": "high" if category in ["maintenance", "safety", "access"] else "medium",
        "created_at": "2024-02-15T10:00:00Z",
        "resolved_at": None,
        "assigned_to": "Support Team",
        "notes": None,
    }
    runtime_incidents[ticket_id] = ticket

    return json.dumps({
        "created": True,
        "ticket_id": ticket_id,
        "category": category,
        "status": "open",
        "message": (
            f"Ticket {ticket_id} created successfully. "
            "Our support team will contact you within 2 hours. "
            "You will receive an email confirmation."
        ),
    })


def escalate_to_human(reason: str) -> str:
    """Escalate this conversation to a human Stayforlong agent when the issue cannot be resolved automatically."""
    return json.dumps({
        "escalated": True,
        "reason": reason,
        "message": (
            "Your case has been escalated to a Stayforlong human agent. "
            "You will receive a WhatsApp message within the next 5-10 minutes. "
            "We apologize for the inconvenience."
        ),
        "whatsapp_notification_sent": True,
        "estimated_response_time": "5-10 minutes",
    })


_contact = STAYFORLONG_CONTACT

PLUGIN = AgentPlugin(
    name="Support",
    routing_hint="Incidents, complaints, maintenance problems, issues during stay",
    instruction=(
        "You are the support agent for Stayforlong. Always respond in the language the user writes in; default to {lang_name} if unclear. "
        "You have been transferred from the main assistant — the user's question is already in the conversation. "
        "NEVER greet the user or say 'Hola' / 'Hello' / 'How can I help' — go straight to answering.\n\n"

        "SCOPE — what you handle:\n"
        "✅ Incidents, problems during stay: maintenance, noise, cleanliness, appliances, WiFi issues\n"
        "✅ Creating new support tickets\n"
        "✅ Checking status of an existing ticket (INC-XXX)\n"
        "✅ Escalating to a human agent\n\n"

        "OUT OF SCOPE — call transfer_to_triage IMMEDIATELY, never attempt to answer:\n"
        "🔄 Reservation details, booking status, prices, cancellation policies\n"
        "🔄 Property amenities, hotel facilities, check-in/out times\n\n"

        "Process for in-scope issues:\n"
        "1. Listen with empathy and understand the problem.\n"
        "2. If the guest mentions an existing ticket (INC-XXX), use lookup_incident to check status.\n"
        "3. For a new issue, use create_incident to register it (ask for booking_id if not provided).\n"
        "4. If the problem persists or the guest is very frustrated, use escalate_to_human.\n\n"

        f"If you cannot resolve the issue or the guest requests human assistance, provide:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n\n"
        "IMPORTANT: For ANYTHING outside your scope, call transfer_to_triage IMMEDIATELY."
    ),
    model=config.GEMINI_MODEL,
    is_fallback=False,
    get_tools=lambda: [lookup_incident, create_incident, escalate_to_human, transfer_to_triage],
)
