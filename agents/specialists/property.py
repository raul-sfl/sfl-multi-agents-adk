import json
from agents.utils import transfer_to_triage
from mock_data.properties import PROPERTIES, PROPERTY_ALIASES
from agents.constants import STAYFORLONG_CONTACT, STAYFORLONG_BASE_URL
import config
from agents.plugin import AgentPlugin

PROPERTY_URLS = {
    "PROP-BCN-001": f"{STAYFORLONG_BASE_URL}/apartamentos-barcelona/gran-via",
    "PROP-MAD-003": f"{STAYFORLONG_BASE_URL}/apartamentos-madrid/salamanca",
    "PROP-LIS-002": f"{STAYFORLONG_BASE_URL}/apartamentos-lisboa/lx-factory",
}


def _find_property(name_or_id: str) -> dict | None:
    """Internal helper: resolve a property by ID or name alias."""
    key = name_or_id.upper()
    if key in PROPERTIES:
        return PROPERTIES[key]
    lower = name_or_id.lower().strip()
    prop_id = PROPERTY_ALIASES.get(lower)
    if prop_id:
        return PROPERTIES.get(prop_id)
    for alias, pid in PROPERTY_ALIASES.items():
        if alias in lower or lower in alias:
            return PROPERTIES.get(pid)
    return None


def lookup_property(name_or_id: str) -> str:
    """Get general information about a Stayforlong property. Accepts property ID (PROP-XXX-NNN) or city/name like 'Barcelona', 'Madrid', 'Lisboa', 'Gran Via', 'Salamanca', 'LX Factory'."""
    prop = _find_property(name_or_id)
    if not prop:
        return json.dumps({
            "found": False,
            "message": (
                f"Property '{name_or_id}' not found. "
                "Available properties are: Barcelona (Gran Via), Madrid (Salamanca), Lisbon (LX Factory)."
            ),
        })
    pid = prop["property_id"]
    return json.dumps({
        "found": True,
        "property_id": pid,
        "name": prop["name"],
        "city": prop["city"],
        "country": prop["country"],
        "address": prop["address"],
        "stars": prop["stars"],
        "type": prop["type"],
        "check_in_time": prop["check_in_time"],
        "check_out_time": prop["check_out_time"],
        "reception_hours": prop["reception_hours"],
        "self_checkin": prop["self_checkin"],
        "early_checkin_available": prop["early_checkin_available"],
        "late_checkout_available": prop["late_checkout_available"],
        "rating": prop["stayforlong_rating"],
        "total_reviews": prop["total_reviews"],
        "stayforlong_url": PROPERTY_URLS.get(pid, STAYFORLONG_BASE_URL),
    })


def get_property_amenities(property_id: str) -> str:
    """Get the full list of amenities for a Stayforlong property. Accepts property ID or city name."""
    prop = _find_property(property_id)
    if not prop:
        return json.dumps({"found": False, "message": f"Property '{property_id}' not found."})

    amenities = prop["amenities"]
    summary = []

    wifi = amenities.get("wifi", {})
    if wifi.get("available"):
        summary.append(f"WiFi: available, {wifi.get('speed_mbps')} Mbps, {wifi.get('cost', 'included')}")

    parking = amenities.get("parking", {})
    if parking.get("available"):
        summary.append(f"Parking: {parking.get('cost', 'included')}" +
                       (" (advance reservation required)" if parking.get("reservation_required") else ""))
    elif parking.get("nearby_garage"):
        summary.append(f"Parking: not on-site, {parking.get('nearby_garage')}")

    gym = amenities.get("gym", {})
    if gym.get("available"):
        summary.append(f"Gym: available ({gym.get('hours', '')})")

    pool = amenities.get("pool", {})
    if pool.get("available"):
        summary.append("Pool: available")

    laundry = amenities.get("laundry", {})
    if laundry.get("available"):
        summary.append(f"Laundry: {laundry.get('type')}")

    kitchen = amenities.get("kitchen", {})
    if kitchen.get("available"):
        equip = kitchen.get("equipment", [])
        summary.append(f"Kitchen: {kitchen.get('type')}" +
                       (f" with {', '.join(equip)}" if equip else ""))

    cleaning = amenities.get("cleaning", {})
    if cleaning:
        summary.append(f"Cleaning: {cleaning.get('frequency')}, {'included' if cleaning.get('included') else 'not included'}")

    extras = []
    if amenities.get("air_conditioning"):
        extras.append("air conditioning")
    if amenities.get("elevator"):
        extras.append("elevator")
    if amenities.get("pets_allowed"):
        extras.append("pets allowed" + (f" ({amenities.get('pets_policy')})" if amenities.get("pets_policy") else ""))
    else:
        extras.append("no pets")
    if amenities.get("smoking_allowed"):
        extras.append("smoking allowed")
    else:
        extras.append("no smoking")

    if extras:
        summary.append("Other: " + ", ".join(extras))

    pid = prop["property_id"]
    return json.dumps({
        "found": True,
        "property_name": prop["name"],
        "amenities_summary": summary,
        "stayforlong_url": PROPERTY_URLS.get(pid, STAYFORLONG_BASE_URL),
    })


def get_checkin_info(property_id: str) -> str:
    """Get check-in and check-out times and procedures for a property."""
    prop = _find_property(property_id)
    if not prop:
        return json.dumps({"found": False, "message": f"Property '{property_id}' not found."})

    pid = prop["property_id"]
    info = {
        "found": True,
        "property_name": prop["name"],
        "check_in_time": prop["check_in_time"],
        "check_out_time": prop["check_out_time"],
        "reception_hours": prop["reception_hours"],
        "early_checkin_available": prop["early_checkin_available"],
        "late_checkout_available": prop["late_checkout_available"],
        "self_checkin": prop["self_checkin"],
        "stayforlong_url": PROPERTY_URLS.get(pid, STAYFORLONG_BASE_URL),
    }
    if prop["self_checkin"] and prop.get("self_checkin_method"):
        info["self_checkin_method"] = prop["self_checkin_method"]

    return json.dumps(info)


_contact = STAYFORLONG_CONTACT

PLUGIN = AgentPlugin(
    name="Accommodations",
    routing_hint="Accommodation info, amenities, check-in/out times, facilities",
    instruction=(
        "You are the accommodation specialist for Stayforlong, a long-stay apartment platform. "
        "Always respond in {lang_name}. "
        "You have been transferred from the main assistant — the user's question is already in the conversation. "
        "NEVER greet the user or say 'Hola' / 'Hello' / 'How can I help' — go straight to answering.\n\n"

        "SCOPE — what you handle:\n"
        "✅ Property/accommodation general info: address, stars, type, ratings\n"
        "✅ Amenities: WiFi, parking, gym, pool, kitchen, cleaning, pets policy\n"
        "✅ Check-in/check-out times and procedures, self check-in instructions\n"
        "✅ Reception hours, early check-in, late check-out availability\n\n"

        "OUT OF SCOPE — call transfer_to_triage IMMEDIATELY, never attempt to answer:\n"
        "🔄 Reservation details, booking status, prices, cancellation policies\n"
        "🔄 Incidents, complaints, maintenance problems during stay\n\n"

        "POLICY — MANDATORY:\n"
        "• NEVER suggest the guest contact the property directly (no direct property phones or emails).\n"
        "• Always refer guests to the Stayforlong listing page (stayforlong_url from tool results) "
        "for more details or to manage their booking.\n"
        "• If you cannot resolve the query, refer to Stayforlong support:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n\n"

        "Available properties: Gran Via (Barcelona), Residencia Salamanca (Madrid), "
        "LX Factory Residences (Lisbon).\n\n"
        "Use lookup_property for general info, get_property_amenities for amenities detail, "
        "and get_checkin_info for arrival/departure procedures. "
        "Always include the stayforlong_url in your response so the guest can see full details.\n\n"
        "IMPORTANT: For ANYTHING outside your scope, call transfer_to_triage IMMEDIATELY."
    ),
    model=config.GEMINI_MODEL,
    is_fallback=False,
    get_tools=lambda: [lookup_property, get_property_amenities, get_checkin_info, transfer_to_triage],
)
