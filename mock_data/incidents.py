INCIDENTS = {
    "INC-001": {
        "ticket_id": "INC-001",
        "booking_id": "SFL-2024-002",
        "category": "maintenance",
        "description": "Hot water not working in bathroom",
        "status": "in_progress",
        "priority": "high",
        "created_at": "2024-02-05T14:30:00Z",
        "resolved_at": None,
        "assigned_to": "Maintenance Team",
        "notes": "Plumber scheduled for tomorrow morning 9-11am",
    },
    "INC-002": {
        "ticket_id": "INC-002",
        "booking_id": "SFL-2024-001",
        "category": "noise",
        "description": "Loud neighbors on the floor above",
        "status": "resolved",
        "priority": "medium",
        "created_at": "2024-03-18T22:15:00Z",
        "resolved_at": "2024-03-18T23:00:00Z",
        "assigned_to": "Property Manager",
        "notes": "Neighbor was spoken to, issue resolved",
    },
}

INCIDENT_CATEGORIES = [
    "maintenance",
    "noise",
    "cleanliness",
    "appliance",
    "wifi",
    "access",
    "safety",
    "billing",
    "other",
]

# In-memory store for newly created incidents during the session
runtime_incidents: dict = {}
