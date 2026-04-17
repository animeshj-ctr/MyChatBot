import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_URL
from models.models import Base, User, Expense  # your SQLAlchemy models

import re
from calendar import monthrange

load_dotenv('config/dev/.env')

# ---------------- App ----------------
app = FastAPI(title="FastAPI MyAPI")

# Serve static directory for index.html, css, js
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- DB setup ----------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def parse_period(text: str):
    """
    Returns (start_date, end_date) or (None, None)
    """
    t = text.lower()
    today = date.today()

    # Date range: 2026-03-01 to 2026-03-15
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(?:to|-)\s+(\d{4}-\d{2}-\d{2})", t)
    if m:
        try:
            s = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            e = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            if s > e:
                s, e = e, s
            return s, e
        except:
            pass

    if "today" in t:
        return today, today
    if "yesterday" in t:
        y = today - timedelta(days=1)
        return y, y
    if "this month" in t:
        s = today.replace(day=1)
        e = today.replace(day=monthrange(today.year, today.month)[1])
        return s, e
    if "last month" in t:
        y = today.year
        m = today.month - 1
        if m == 0:
            y -= 1
            m = 12
        s = date(y, m, 1)
        e = date(y, m, monthrange(y, m)[1])
        return s, e

    return None, None

def find_user_from_text(text: str, db: Session):
    """
    Tries to detect user by `user 3` style id or by name within text.
    Returns (user_obj, user_id or None)
    """
    t = text.lower()

    # Pattern: "user 3" or "userid 3" or "employee 3"
    m = re.search(r"\b(user|userid|employee)\s+(\d+)\b", t)
    if m:
        uid = int(m.group(2))
        u = db.query(User).filter(User.userid == uid).first()
        if u:
            return u, u.userid

    # Try to detect name after 'for ' or 'of ' (very simple heuristic)
    m = re.search(r"\b(?:for|of)\s+([a-zA-Z][a-zA-Z\s]{1,40})$", text.strip(), flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        u = db.query(User).filter(User.name.ilike(f"%{name}%")).first()
        if u:
            return u, u.userid

    return None, None

def query_mcp_server(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    Sends a tool invocation request to the MCP server.
    Assumes MCP server endpoint is set in env var MCP_SERVER_URL.
    """
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")  # Default to local MCP server
    payload = {
        "tool": tool_name,
        "parameters": parameters
    }
    try:
        response = requests.post(mcp_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result.get("result", "MCP server error: No result")
    except requests.RequestException as e:
        return f"MCP server error: {str(e)}"
    
def answer_business_query(text: str, db: Session) -> str | None:
    """
    Try to answer using the Expenses/Users DB.
    Returns a human-friendly string or None if not understood.
    """
    t = text.lower().strip()
    start, end = parse_period(t)
    user_obj, uid = find_user_from_text(t, db)

    # Build filters
    filters = []
    if uid:
        filters.append(Expense.userid == uid)
    if start and end:
        filters.append(Expense.expense_date.between(start, end))

    # Identify intent
    is_total = any(w in t for w in ["total", "sum", "aggregate"])
    is_debit = any(w in t for w in ["debit", "spent", "spend", "expense"])
    is_credit = any(w in t for w in ["credit", "credited", "income"])
    is_net = any(w in t for w in ["net", "balance"])
    is_list = any(w in t for w in ["list", "show"]) and "expense" in t
    is_userinfo = any(w in t for w in ["user info", "user detail", "contact", "email", "phone"])
    is_topmode = ("top" in t and "mode" in t) or ("top" in t and "payment" in t)

    # Extract N for "last N" or "top N"
    N = 5
    nm = re.search(r"\b(last|top)\s+(\d+)\b", t)
    if nm:
        try:
            N = max(1, min(50, int(nm.group(2))))
        except:
            pass

    # 1) Totals (debit / credit / net)
    if is_total or is_net or is_debit or is_credit:
        q = db.query(Expense).filter(*filters)

        # Separate sums
        debit_sum = db.query(func.coalesce(func.sum(Expense.amount), 0.0)) \
                      .filter(*(filters + [Expense.credit_or_debit == "debit"])) \
                      .scalar() or 0.0
        credit_sum = db.query(func.coalesce(func.sum(Expense.amount), 0.0)) \
                       .filter(*(filters + [Expense.credit_or_debit == "credit"])) \
                       .scalar() or 0.0

        # Decide what to return
        if is_net:
            net = debit_sum - credit_sum  # define "net spend" = debit - credit
            title = "Net spend"
            val = net
        elif is_credit and not is_debit:
            title = "Total credit"
            val = credit_sum
        else:
            # default to debit (spend)
            title = "Total debit (spend)"
            val = debit_sum

        who = f" for user {uid}" if uid else " (all users)"
        per = ""
        if start and end:
            per = f" between {start.isoformat()} and {end.isoformat()}"
        return f"{title}{who}{per}: ₹{val:,.2f} (credit: ₹{credit_sum:,.2f}, debit: ₹{debit_sum:,.2f})"

    # 2) List last N expenses
    if is_list:
        items = db.query(Expense).filter(*filters) \
                 .order_by(Expense.expense_date.desc()) \
                 .limit(N).all()
        if not items:
            who = f"user {uid}" if uid else "all users"
            if start and end:
                return f"No expenses found for {who} between {start} and {end}."
            return f"No expenses found for {who}."
        lines = []
        for e in items:
            lines.append(
                f"{e.expense_date} | {e.credit_or_debit.upper():6} | ₹{e.amount:,.2f} | {e.mode_of_payment} | {e.description}"
            )
        header = f"Last {len(items)} expenses" + (f" for user {uid}" if uid else " (all users)")
        if start and end:
            header += f" between {start} and {end}"
        return header + ":\n" + "\n".join(lines)

    # 3) User info
    if is_userinfo or ("user" in t and ("detail" in t or "info" in t)):
        # If user not identified, try to extract a numeric id
        if not uid:
            m = re.search(r"\buser\s+(\d+)\b", t)
            if m:
                uid = int(m.group(1))
        if not uid:
            return "Please specify a user id or name, e.g., 'user 3 details'."

        u = db.query(User).filter(User.userid == uid).first()
        if not u:
            return f"User {uid} not found."
        return (f"User {u.userid}: {u.name}, EmployeeID={u.employeeid}, "
                f"DOB={u.dob}, Phone={u.contact_number}, Email={u.email or '-'}")

    # 4) Top N payment modes by amount
    if is_topmode:
        rows = db.query(Expense.mode_of_payment,
                        func.coalesce(func.sum(Expense.amount), 0.0).label("amt")) \
                 .filter(*filters) \
                 .group_by(Expense.mode_of_payment) \
                 .order_by(func.coalesce(func.sum(Expense.amount), 0.0).desc()) \
                 .limit(N).all()
        if not rows:
            return "No data to compute top payment modes."
        header = f"Top {len(rows)} payment modes" + (f" for user {uid}" if uid else " (all users)")
        if start and end:
            header += f" between {start} and {end}"
        lines = [f"{i+1}. {m}: ₹{amt:,.2f}" for i, (m, amt) in enumerate(rows)]
        return header + ":\n" + "\n".join(lines)

    # Not understood → let LLM/echo handle
    return None

# ---------------- Pydantic Schemas ----------------
class UserCreate(BaseModel):
    employeeid: int
    name: str
    dob: date
    contact_number: str
    email: Optional[str] = None

class ExpenseCreate(BaseModel):
    userid: int  # FK to User
    mode_of_payment: str
    amount: float
    credit_or_debit: str  # "credit" or "debit"
    expense_date: date
    description: str

# (Optional) Response schemas – if you know exact fields, prefer these:
# class UserOut(BaseModel):
#     userid: int
#     employeeid: int
#     name: str
#     dob: date
#     contact_number: str
#     email: Optional[str] = None
#     class Config:
#         orm_mode = True  # Pydantic v1
#         # Pydantic v2: model_config = ConfigDict(from_attributes=True)

# class ExpenseOut(BaseModel):
#     expenseid: int
#     userid: int
#     mode_of_payment: str
#     amount: float
#     credit_or_debit: str
#     expense_date: date
#     description: str
#     class Config:
#         orm_mode = True

# ---------------- Home Route ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ---------------- Users & Expenses (sync = simpler) ----------------
@app.post("/users/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    # return db_user  # risky: ORM object not JSON serializable
    return jsonable_encoder(db_user)  # quick + safe

@app.post("/expenses/")
def create_expense(expense: ExpenseCreate, db: Session = Depends(get_db)):
    if expense.credit_or_debit not in ("credit", "debit"):
        raise HTTPException(status_code=400, detail="Invalid credit_or_debit value")

    user = db.query(User).filter(User.userid == expense.userid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User with this userid not found")

    db_expense = Expense(
        userid=expense.userid,
        mode_of_payment=expense.mode_of_payment,
        amount=expense.amount,
        credit_or_debit=expense.credit_or_debit,
        expense_date=expense.expense_date,
        description=expense.description,
    )
    db.add(db_expense)
    db.commit()
    db.refresh(db_expense)
    return jsonable_encoder(db_expense)

@app.get("/users/")
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return jsonable_encoder(users)

@app.get("/expenses/{userid}")
def get_expenses_by_user(userid: int, db: Session = Depends(get_db)):
    expenses = db.query(Expense).filter(Expense.userid == userid).all()
    return jsonable_encoder(expenses)

# ---------------- Chatbot (REST + WebSocket) ----------------
USE_ECHO = os.getenv("USE_ECHO", "true").lower() == "true"
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}

class ChatMessage(BaseModel):
    session_id: Optional[str] = "default"
    message: str

def get_response_from_llm(history: List[Dict[str, str]], user_msg: str) -> str:
    """
    Swap this with real LLM call when ready.
    """
    # if USE_ECHO:
    #     if "hello" in user_msg.lower():
    #         return "Hi! I'm your FastAPI chatbot. How can I help you today?"
    #     return f"You said: {user_msg}"

    from openai import OpenAI
    HF_SECRET_KEY= os.getenv("HUGGINGFACEHUB_API_TOKEN")


    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_SECRET_KEY,
    )

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for t in history:
        messages.append({"role": "user", "content": t["user"]})
        messages.append({"role": "assistant", "content": t["bot"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(model="MiniMaxAI/MiniMax-M2.7:novita", messages=messages, temperature=0.4)
        return resp.choices[0].message.content

    except Exception as e:
        print("Retrying...", e)
        time.sleep(2)

    return "LLM not configured. Set USE_ECHO=true or add an API key."

@app.post("/chat")
def chat(payload: ChatMessage, db: Session = Depends(get_db)):
    session_id = (payload.session_id or "default").strip()
    user_msg = (payload.message or "").strip()
    print(f"{user_msg} (session: {session_id})")
    if not user_msg:
        return {"bot": "Please say something.", "session_id": session_id}

    # 1) Try DB-based answer
    db_answer = answer_business_query(user_msg, db)
    if db_answer:
        return {"bot": db_answer, "session_id": session_id}

    # 2) Otherwise, fallback to LLM/echo
    history = CHAT_HISTORY.setdefault(session_id, [])
    bot_reply = get_response_from_llm(history, user_msg)
    history.append({"user": user_msg, "bot": bot_reply})
    CHAT_HISTORY[session_id] = history[-20:]
    return {"bot": bot_reply, "session_id": session_id}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = "default"
    CHAT_HISTORY.setdefault(session_id, [])

    try:
        while True:
            data = await ws.receive_json()
            user_msg = (data.get("message") or "").strip()
            if not user_msg:
                await ws.send_json({"bot": "Please say something."})
                continue

            history = CHAT_HISTORY[session_id]
            full = get_response_from_llm(history, user_msg)

            # Simulated streaming by words (real LLMs stream tokens)
            partial = ""
            for w in full.split():
                partial += w + " "
                await ws.send_json({"bot_partial": partial.strip()})

            await ws.send_json({"bot": full})
            history.append({"user": user_msg, "bot": full})
            CHAT_HISTORY[session_id] = history[-20:]

    except WebSocketDisconnect:
        pass