import json
from google.adk.agents import LlmAgent
from agents.utils import transfer_to_triage
from mock_data.reservations import RESERVATIONS, EMAIL_INDEX, CANCELLATION_POLICIES
from agents.constants import STAYFORLONG_CONTACT
import config


def lookup_reservation(booking_id: str, guest_name: str = "") -> str:
    """Look up a reservation by booking ID (format: SFL-YYYY-NNN).
    Without guest_name: returns only non-sensitive info (property, dates, status).
    With guest_name provided: verifies identity and returns full details if name matches."""
    res = RESERVATIONS.get(booking_id.upper())
    if not res:
        return json.dumps({
            "found": False,
            "message": f"No reservation found with ID '{booking_id}'. Please verify the booking number.",
        })

    public_info = {
        "found": True,
        "booking_id": res["booking_id"],
        "property": res["property_name"],
        "check_in": res["check_in"],
        "check_out": res["check_out"],
        "nights": res["nights"],
        "room_type": res["room_type"],
        "status": res["status"],
        "cancellation_policy": res["cancellation_policy"],
    }

    if not guest_name:
        public_info["privacy_note"] = (
            "Only basic info is shown without identity verification. "
            "To access price, payment status, and special requests, "
            "please ask the guest for their full name."
        )
        return json.dumps(public_info)

    stored_name = res["guest_name"].lower()
    provided_name = guest_name.strip().lower()
    name_match = provided_name in stored_name or stored_name in provided_name

    if not name_match:
        return json.dumps({
            "found": True,
            "identity_verified": False,
            "message": (
                "The name provided does not match the name on this reservation. "
                "Please verify your full name and try again."
            ),
        })

    public_info.update({
        "identity_verified": True,
        "guest_name": res["guest_name"],
        "total_price": f"{res['total_price']} {res['currency']}",
        "payment_status": res["payment_status"],
        "cancellation_deadline": res["cancellation_deadline"],
        "special_requests": res["special_requests"],
    })
    return json.dumps(public_info)


def get_reservations_by_email(email: str) -> str:
    """Find the booking ID associated with a guest email address. Returns only basic non-sensitive info."""
    res = EMAIL_INDEX.get(email.lower().strip())
    if not res:
        return json.dumps({
            "found": False,
            "message": f"No reservations found for email '{email}'.",
        })
    return json.dumps({
        "found": True,
        "booking_id": res["booking_id"],
        "property": res["property_name"],
        "check_in": res["check_in"],
        "check_out": res["check_out"],
        "status": res["status"],
        "note": "Use lookup_reservation with the booking_id and guest_name to access full details.",
    })


def check_cancellation_policy(booking_id: str) -> str:
    """Get the cancellation policy description for a reservation. No identity verification required."""
    res = RESERVATIONS.get(booking_id.upper())
    if not res:
        return json.dumps({"found": False, "message": f"Reservation '{booking_id}' not found."})
    policy_name = res["cancellation_policy"]
    return json.dumps({
        "booking_id": booking_id,
        "policy_type": policy_name,
        "policy_description": CANCELLATION_POLICIES.get(policy_name, "Policy not available."),
        "current_status": res["status"],
        "note": "Cancellation deadline is only shown after identity verification.",
    })


_contact = STAYFORLONG_CONTACT

booking_agent = LlmAgent(
    name="Booking",
    model=config.GEMINI_MODEL,
    instruction=(
        "You are the reservations specialist for Stayforlong. Always respond in {lang_name}. "
        "You have been transferred from the main assistant — the user's question is already in the conversation. "
        "NEVER greet the user or say 'Hola' / 'Hello' / 'How can I help' — go straight to answering.\n\n"

        "SCOPE — what you handle:\n"
        "✅ Booking status, confirmation, check-in/out dates, number of nights\n"
        "✅ Room type, price, payment status, cancellation policies and deadlines\n"
        "✅ Finding a booking by email\n"
        "✅ Modification or cancellation requests → you cannot do them directly, "
        "but inform the guest they must contact our team and provide the contact details below.\n\n"

        "OUT OF SCOPE — call transfer_to_triage IMMEDIATELY, never attempt to answer:\n"
        "🔄 Hotel/property amenities, facilities, WiFi, parking, gym, pool\n"
        "🔄 Check-in procedures, key pickup, self check-in instructions\n"
        "🔄 Incidents, complaints, maintenance problems, noise\n\n"

        "MODIFICATION & CANCELLATION REQUESTS:\n"
        "• You CANNOT modify or cancel reservations directly.\n"
        "• When a guest asks to modify dates, room type, or cancel: look up their booking "
        "to confirm the details and cancellation policy, then direct them to our team:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n"
        "• NEVER call transfer_to_triage for modification or cancellation requests — handle them yourself.\n\n"

        "PRIVACY & SECURITY POLICY — MANDATORY:\n"
        "• With booking ID only → call lookup_reservation(booking_id) → you may share: "
        "property name, check-in/out dates, room type, booking status, and cancellation policy type.\n"
        "• Sensitive data (price, payment status, cancellation deadline, special requests) → "
        "ALWAYS ask for the guest's full name first, then call lookup_reservation(booking_id, guest_name).\n"
        "• NEVER reveal email addresses or other guests' personal data.\n"
        "• If name verification fails, inform the guest and ask them to double-check their name.\n\n"

        "If the user only has an email address, use get_reservations_by_email to find their booking ID.\n\n"
        f"If you cannot resolve the issue, tell the guest to contact Stayforlong directly:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n\n"
        "IMPORTANT: For ANYTHING outside your scope, call transfer_to_triage IMMEDIATELY."
    ),
    tools=[lookup_reservation, get_reservations_by_email, check_cancellation_policy, transfer_to_triage],
)
