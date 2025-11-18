import os
from datetime import datetime, timedelta, date
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db

# ---------- Types ----------
UserRole = Literal['admin', 'staff']
PaymentMethod = Literal['efectivo', 'tarjeta', 'transferencia', 'otro']
MembershipStatus = Literal['active', 'expired', 'cancelled']

# ---------- Helpers ----------

def today_utc_date() -> date:
    return datetime.utcnow().date()


def compute_membership_status(start_date: date, end_date: date, status: str) -> str:
    if status == 'cancelled':
        return 'cancelled'
    d = today_utc_date()
    if start_date <= d <= end_date:
        return 'active'
    if d > end_date:
        return 'expired'
    return 'active'


def get_settings() -> Dict[str, Any]:
    settings = db['gymsettings'].find_one({})
    if not settings:
        default_doc = {
            'gym_name': 'Mi Gimnasio',
            'currency': 'MXN',
            'opening_hours': 'Lunes a Viernes 6:00–22:00',
            'grace_period_days': 3,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }
        res = db['gymsettings'].insert_one(default_doc)
        settings = db['gymsettings'].find_one({'_id': res.inserted_id})
    settings['id'] = str(settings['_id'])
    return settings


def membership_badge(membership: Dict[str, Any]) -> str:
    """Return status badge: 'Activo' | 'Por vencer' | 'Vencido' based on dates and settings."""
    if membership.get('status') == 'cancelled':
        return 'Vencido'
    end: date = membership['end_date']
    start: date = membership['start_date']
    d = today_utc_date()
    if start <= d <= end:
        # expiring soon if within 7 days
        if (end - d).days <= 7:
            return 'Por vencer'
        return 'Activo'
    if d > end:
        return 'Vencido'
    return 'Activo'


def serialize_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = dict(doc)
    if '_id' in d:
        d['id'] = str(d.pop('_id'))
    # convert date/datetime to iso
    for k, v in list(d.items()):
        if isinstance(v, (datetime,)):
            d[k] = v.isoformat()
        if isinstance(v, (date,)):
            d[k] = v.isoformat()
    return d

# ---------- Schemas ----------

class PlanCreate(BaseModel):
    name: str
    duration_days: int
    price: float
    description: Optional[str] = None
    is_active: bool = True

class Plan(PlanCreate):
    id: str

class MemberCreate(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    notes: Optional[str] = None
    is_active: bool = True

class Member(MemberCreate):
    id: str

class MembershipCreate(BaseModel):
    plan_id: str
    start_date: date
    end_date: date

class Membership(BaseModel):
    id: str
    member_id: str
    plan_id: str
    start_date: date
    end_date: date
    status: MembershipStatus
    created_by: Optional[str] = None

class PaymentCreate(BaseModel):
    membership_id: str
    amount: float
    payment_date: date
    method: PaymentMethod
    notes: Optional[str] = None

class Payment(BaseModel):
    id: str
    membership_id: str
    amount: float
    payment_date: date
    method: PaymentMethod
    recorded_by: Optional[str] = None
    notes: Optional[str] = None

class CheckinCreate(BaseModel):
    member_id: str

class Checkin(BaseModel):
    id: str
    member_id: str
    checkin_time: datetime
    recorded_by: Optional[str] = None

class SettingsUpdate(BaseModel):
    gym_name: Optional[str] = None
    currency: Optional[str] = None
    opening_hours: Optional[str] = None
    grace_period_days: Optional[int] = None

# ---------- App ----------

app = FastAPI(title="GymControl AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "GymControl AI Backend running"}

# ---------- Plans ----------

@app.get("/plans", response_model=List[Plan])
def list_plans(only_active: bool = False):
    q = {}
    if only_active:
        q["is_active"] = True
    plans = [serialize_id(p) for p in db['membershipplans'].find(q).sort("created_at", -1)]
    return plans

@app.post("/plans", response_model=Plan)
def create_plan(payload: PlanCreate):
    doc = payload.dict()
    doc['created_at'] = datetime.utcnow()
    res = db['membershipplans'].insert_one(doc)
    created = db['membershipplans'].find_one({"_id": res.inserted_id})
    return serialize_id(created)

@app.patch("/plans/{plan_id}", response_model=Plan)
def update_plan(plan_id: str, payload: PlanCreate):
    db['membershipplans'].update_one({'_id': db.ObjectId(plan_id) if hasattr(db, 'ObjectId') else plan_id}, {"$set": payload.dict()})
    doc = db['membershipplans'].find_one({'_id': db.ObjectId(plan_id) if hasattr(db, 'ObjectId') else plan_id})
    if not doc:
        raise HTTPException(404, "Plan no encontrado")
    return serialize_id(doc)

# ---------- Members ----------

@app.get("/members", response_model=List[Member])
def list_members(q: Optional[str] = Query(None, description="name/phone search")):
    query: Dict[str, Any] = {}
    if q:
        # simple regex OR on first_name, last_name, phone
        query = {"$or": [
            {"first_name": {"$regex": q, "$options": "i"}},
            {"last_name": {"$regex": q, "$options": "i"}},
            {"phone": {"$regex": q, "$options": "i"}},
        ]}
    items = [serialize_id(m) for m in db['members'].find(query).sort("created_at", -1)]
    return items

@app.post("/members", response_model=Member)
def create_member(payload: MemberCreate):
    doc = payload.dict()
    doc['created_at'] = datetime.utcnow()
    res = db['members'].insert_one(doc)
    created = db['members'].find_one({"_id": res.inserted_id})
    return serialize_id(created)

@app.get("/members/{member_id}", response_model=Member)
def get_member(member_id: str):
    doc = db['members'].find_one({"_id": member_id})
    if not doc:
        doc = db['members'].find_one({"_id": db.ObjectId(member_id)}) if hasattr(db, 'ObjectId') else None
    if not doc:
        raise HTTPException(404, "Socio no encontrado")
    return serialize_id(doc)

# ---------- Memberships ----------

@app.get("/members/{member_id}/memberships")
def list_member_memberships(member_id: str):
    items = []
    for m in db['memberships'].find({"member_id": member_id}).sort("start_date", -1):
        m = serialize_id(m)
        m['badge'] = membership_badge({
            'start_date': datetime.fromisoformat(m['start_date']).date(),
            'end_date': datetime.fromisoformat(m['end_date']).date(),
            'status': m.get('status', 'active')
        })
        items.append(m)
    return items

@app.post("/members/{member_id}/memberships", response_model=Membership)
def create_membership(member_id: str, payload: MembershipCreate):
    # ensure plan exists
    plan = db['membershipplans'].find_one({"_id": payload.plan_id}) or (db['membershipplans'].find_one({"_id": db.ObjectId(payload.plan_id)}) if hasattr(db, 'ObjectId') else None)
    if not plan:
        raise HTTPException(400, "Plan inválido")
    doc = {
        'member_id': member_id,
        'plan_id': serialize_id(plan)['id'],
        'start_date': payload.start_date,
        'end_date': payload.end_date,
        'status': 'active',
        'created_at': datetime.utcnow(),
    }
    res = db['memberships'].insert_one(doc)
    created = db['memberships'].find_one({"_id": res.inserted_id})
    return serialize_id(created)

@app.post("/memberships/{membership_id}/cancel")
def cancel_membership(membership_id: str):
    upd = db['memberships'].update_one({"_id": membership_id}, {"$set": {"status": 'cancelled'}})
    if upd.matched_count == 0 and hasattr(db, 'ObjectId'):
        upd = db['memberships'].update_one({"_id": db.ObjectId(membership_id)}, {"$set": {"status": 'cancelled'}})
    if upd.matched_count == 0:
        raise HTTPException(404, "Membresía no encontrada")
    return {"ok": True}

@app.get("/memberships")
def list_memberships(state: Optional[str] = None, plan_id: Optional[str] = None, start_from: Optional[date] = None, start_to: Optional[date] = None):
    q: Dict[str, Any] = {}
    if plan_id:
        q['plan_id'] = plan_id
    if start_from or start_to:
        q['start_date'] = {}
        if start_from:
            q['start_date']['$gte'] = start_from
        if start_to:
            q['start_date']['$lte'] = start_to
    items = []
    for m in db['memberships'].find(q).sort('start_date', -1):
        m = serialize_id(m)
        status = compute_membership_status(
            datetime.fromisoformat(m['start_date']).date(),
            datetime.fromisoformat(m['end_date']).date(),
            m.get('status', 'active')
        )
        if state and state != 'all':
            if state == 'active' and status != 'active':
                continue
            if state == 'expired' and status != 'expired':
                continue
            if state == 'cancelled' and status != 'cancelled':
                continue
        m['computed_status'] = status
        items.append(m)
    return items

# ---------- Payments ----------

@app.get("/payments")
def list_payments(date_from: Optional[date] = None, date_to: Optional[date] = None, method: Optional[PaymentMethod] = None):
    q: Dict[str, Any] = {}
    if date_from or date_to:
        q['payment_date'] = {}
        if date_from:
            q['payment_date']['$gte'] = date_from
        if date_to:
            q['payment_date']['$lte'] = date_to
    if method:
        q['method'] = method
    items = [serialize_id(p) for p in db['payments'].find(q).sort('payment_date', -1)]
    return items

@app.post("/payments", response_model=Payment)
def create_payment(payload: PaymentCreate):
    # ensure membership exists
    mem = db['memberships'].find_one({"_id": payload.membership_id}) or (db['memberships'].find_one({"_id": db.ObjectId(payload.membership_id)}) if hasattr(db, 'ObjectId') else None)
    if not mem:
        raise HTTPException(400, "Membresía inválida")
    doc = payload.dict()
    doc['created_at'] = datetime.utcnow()
    res = db['payments'].insert_one(doc)
    created = db['payments'].find_one({"_id": res.inserted_id})
    return serialize_id(created)

# ---------- Check-ins ----------

@app.get("/checkins")
def list_checkins(day: Optional[date] = None):
    q: Dict[str, Any] = {}
    if day:
        # filter by date portion
        start_dt = datetime.combine(day, datetime.min.time())
        end_dt = datetime.combine(day, datetime.max.time())
        q['checkin_time'] = {'$gte': start_dt, '$lte': end_dt}
    items = [serialize_id(c) for c in db['checkins'].find(q).sort('checkin_time', -1)]
    return items

@app.post("/members/{member_id}/checkin")
def register_checkin(member_id: str):
    # Determine membership validity with grace period
    settings = get_settings()
    grace = int(settings.get('grace_period_days', 3))
    d = today_utc_date()
    active = None
    in_grace = None
    for m in db['memberships'].find({'member_id': member_id, 'status': 'active'}).sort('end_date', -1):
        start = m['start_date']
        end = m['end_date']
        if start <= d <= end:
            active = m
            break
        if end < d <= (end + timedelta(days=grace)):
            in_grace = m
            break
    if active is None and in_grace is None:
        raise HTTPException(400, "Membresía vencida. Pide al socio que renueve en recepción.")
    doc = {
        'member_id': member_id,
        'checkin_time': datetime.utcnow(),
    }
    res = db['checkins'].insert_one(doc)
    status = 'ok'
    message = 'Entrada registrada'
    if in_grace is not None and active is None:
        status = 'warning'
        message = 'Entrada registrada (en período de gracia)'
    return {'ok': True, 'status': status, 'message': message, 'id': str(res.inserted_id)}

# ---------- Settings ----------

@app.get("/settings")
def read_settings():
    return get_settings()

@app.patch("/settings")
def update_settings(payload: SettingsUpdate):
    current = get_settings()
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    updates['updated_at'] = datetime.utcnow()
    db['gymsettings'].update_one({'_id': db.ObjectId(current['id'])} if hasattr(db, 'ObjectId') else {'_id': current['id']}, {'$set': updates})
    return get_settings()

# ---------- Dashboard stats ----------

@app.get("/stats/overview")
def stats_overview():
    d = today_utc_date()
    month_start = d.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    # Active members = count of distinct members with active membership today
    active_member_ids = set()
    for m in db['memberships'].find({'status': 'active', 'start_date': {'$lte': d}, 'end_date': {'$gte': d}}):
        active_member_ids.add(m['member_id'])
    active_members = len(active_member_ids)
    # Expiring soon: end in next 7 days
    expiring_soon = db['memberships'].count_documents({'status': 'active', 'end_date': {'$gt': d, '$lte': d + timedelta(days=7)}})
    # Expired this month: end_date in current month and before today
    expired_this_month = db['memberships'].count_documents({'status': 'active', 'end_date': {'$gte': month_start, '$lt': next_month, '$lt': d}})
    # Monthly revenue
    revenue = 0.0
    for p in db['payments'].find({'payment_date': {'$gte': month_start, '$lt': next_month}}):
        try:
            revenue += float(p.get('amount', 0))
        except Exception:
            pass
    return {
        'active_members': active_members,
        'expiring_soon': int(expiring_soon),
        'expired_this_month': int(expired_this_month),
        'monthly_revenue': revenue,
    }

@app.get("/stats/payments_daily")
def stats_payments_daily(days: int = 30):
    d = today_utc_date()
    start = d - timedelta(days=days - 1)
    # group by day
    buckets: Dict[str, float] = {}
    for i in range(days):
        key = (start + timedelta(days=i)).isoformat()
        buckets[key] = 0.0
    for p in db['payments'].find({'payment_date': {'$gte': start, '$lte': d}}):
        key = p['payment_date'].isoformat() if isinstance(p['payment_date'], date) else str(p['payment_date'])[:10]
        try:
            buckets[key] += float(p.get('amount', 0))
        except Exception:
            pass
    series = [{'date': k, 'amount': v} for k, v in sorted(buckets.items())]
    return series

# ---------- Seed demo data ----------

@app.post("/seed")
def seed():
    # plans
    if db['membershipplans'].count_documents({}) == 0:
        db['membershipplans'].insert_many([
            {"name": "Mensual", "duration_days": 30, "price": 500.0, "description": "Plan mensual", "is_active": True, "created_at": datetime.utcnow()},
            {"name": "Trimestral", "duration_days": 90, "price": 1300.0, "description": "Plan trimestral", "is_active": True, "created_at": datetime.utcnow()},
            {"name": "Anual", "duration_days": 365, "price": 4500.0, "description": "Plan anual", "is_active": True, "created_at": datetime.utcnow()},
        ])
    plans = list(db['membershipplans'].find({}))
    # settings
    get_settings()
    # members
    if db['members'].count_documents({}) == 0:
        demo_members = []
        for i in range(10):
            demo_members.append({
                'first_name': f"Juan{i}",
                'last_name': f"Pérez{i}",
                'email': f"juan{i}@demo.com",
                'phone': f"555-000{i}",
                'created_at': datetime.utcnow(),
                'is_active': True,
            })
        db['members'].insert_many(demo_members)
    members = list(db['members'].find({}))
    # memberships + payments + checkins
    if db['memberships'].count_documents({}) == 0:
        d = today_utc_date()
        for idx, m in enumerate(members):
            plan = plans[idx % len(plans)]
            # Some active, some expiring soon, some expired
            if idx < 4:  # active 25 days ago to 5 days ahead
                start = d - timedelta(days=25)
                end = start + timedelta(days=plan['duration_days'])
            elif idx < 7:  # expiring within next 7
                start = d - timedelta(days=plan['duration_days'] - 5)
                end = d + timedelta(days=5)
            else:  # expired
                start = d - timedelta(days=plan['duration_days'] + 10)
                end = start + timedelta(days=plan['duration_days'])
            mem_doc = {
                'member_id': str(m['_id']),
                'plan_id': str(plan['_id']),
                'start_date': start,
                'end_date': end,
                'status': 'active',
                'created_at': datetime.utcnow(),
            }
            res = db['memberships'].insert_one(mem_doc)
            # payments random for last 30 days
            for j in range(3):
                pd = d - timedelta(days=(idx * j) % 20)
                db['payments'].insert_one({
                    'membership_id': str(res.inserted_id),
                    'amount': float(plan['price']) / 3.0,
                    'payment_date': pd,
                    'method': 'efectivo',
                    'created_at': datetime.utcnow(),
                })
            # checkins last 10 days randomly
            for k in range(5):
                cd = datetime.utcnow() - timedelta(days=(idx + k) % 10, hours=k)
                db['checkins'].insert_one({
                    'member_id': str(m['_id']),
                    'checkin_time': cd,
                })
    return {"ok": True}


# Health/test
@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            collections = db.list_collection_names()
            response["collections"] = collections[:10]
            response["connection_status"] = "Connected"
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
