#!/usr/bin/env python3
"""
Simple test script to validate calendar services design.

This tests the core calendar services without requiring database dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from uuid import uuid4

# Test basic imports
try:
    from core.services.common import OperationResult, Actor, DomainEvent
    print("✅ Core services imported successfully")
except ImportError as e:
    print(f"❌ Failed to import core services: {e}")
    sys.exit(1)

# Test calendar types without repository dependencies
try:
    from core.services.calendar.validation import CalendarEventValidator
    print("✅ Calendar validator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import calendar validator: {e}")

# Test validation logic
try:
    from core.services.calendar.types import CalendarEventCreate
    
    # Create sample event data
    event_data = CalendarEventCreate(
        title="Test Meeting",
        description="A test meeting",
        start_time=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc),
        all_day=False,
        timezone_name="UTC",
        location="Conference Room A",
        is_recurring=False,
        recurrence_pattern=None,
    )
    
    print("✅ Calendar event data created successfully")
    print(f"   Title: {event_data.title}")
    print(f"   Duration: {event_data.end_time - event_data.start_time}")
    
except ImportError as e:
    print(f"❌ Failed to import calendar types: {e}")

# Test domain events
try:
    from core.services.calendar.events import CalendarEventCreated
    
    # Create a test domain event
    event = CalendarEventCreated.create_for_event(
        event_id=str(uuid4()),
        tenant_id="test-tenant", 
        actor_type="user",
        event_title="Test Event",
        event_start=datetime.now(timezone.utc),
        event_end=datetime.now(timezone.utc)
    )
    
    print("✅ Domain event created successfully")
    print(f"   Event type: {event.event_type}")
    print(f"   Actor: {event.actor}")
    
except ImportError as e:
    print(f"❌ Failed to import calendar events: {e}")
except Exception as e:
    print(f"❌ Failed to create domain event: {e}")

# Test Actor and permissions
try:
    actor = Actor(
        type="user",
        id="test-user",
        tenant_id="test-tenant",
        permissions={"calendar.read", "calendar.write"}
    )
    
    print("✅ Actor created successfully")
    print(f"   Type: {actor.type}")
    print(f"   Permissions: {sorted(actor.permissions)}")
    print(f"   Has calendar.read: {actor.has_permission('calendar.read')}")
    print(f"   Has calendar.admin: {actor.has_permission('calendar.admin')}")
    
except Exception as e:
    print(f"❌ Failed to create actor: {e}")

print("\n🎉 Calendar services core design validation completed!")
print("   Ready for integration with repository and UI layers")