"""
Database Schemas for GymControl AI

Each Pydantic model represents a MongoDB collection. The collection name is the lowercase of the class name.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date, datetime

# Types
UserRole = Literal['admin', 'staff']
PaymentMethod = Literal['efectivo', 'tarjeta', 'transferencia', 'otro']
MembershipStatus = Literal['active', 'expired', 'cancelled']

class Profiles(BaseModel):
    id: Optional[str] = Field(None, description="Supabase auth id or generated UUID")
    full_name: str
    phone: Optional[str] = None
    role: UserRole = 'staff'
    email: Optional[str] = None
    created_at: Optional[datetime] = None

class MembershipPlans(BaseModel):
    id: Optional[str] = None
    name: str
    duration_days: int
    price: float
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

class Members(BaseModel):
    id: Optional[str] = None
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True

class Memberships(BaseModel):
    id: Optional[str] = None
    member_id: str
    plan_id: str
    start_date: date
    end_date: date
    status: MembershipStatus = 'active'
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None

class Payments(BaseModel):
    id: Optional[str] = None
    membership_id: str
    amount: float
    payment_date: date
    method: PaymentMethod
    recorded_by: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None

class Checkins(BaseModel):
    id: Optional[str] = None
    member_id: str
    checkin_time: Optional[datetime] = None
    recorded_by: Optional[str] = None

class GymSettings(BaseModel):
    id: Optional[str] = None
    gym_name: str = 'Mi Gimnasio'
    currency: str = 'MXN'
    opening_hours: Optional[str] = None
    grace_period_days: int = 3
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
