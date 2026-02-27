"""
Admin dashboard router — /admin

Exposes:
  GET /admin                            → HTML dashboard SPA
  GET /admin/api/stats                  → aggregate counts
  GET /admin/api/conversations          → list (filters: status, lang, limit)
  GET /admin/api/conversations/{id}     → conversation metadata
  GET /admin/api/conversations/{id}/messages → message thread
  GET /admin/api/users/{user_id}/conversations → all convs for a user

Auth: set ADMIN_API_KEY in .env.
  - Dashboard: pass ?key=YOUR_KEY in URL
  - API:       Authorization: Bearer YOUR_KEY  OR  ?key=YOUR_KEY
If ADMIN_API_KEY is empty the dashboard is open (dev only).
"""

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import HTMLResponse
from typing import Optional

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Auth dependency ───────────────────────────────────────────────────────────

async def require_admin(
    authorization: Optional[str] = Header(None),
    key: Optional[str] = Query(None),
) -> None:
    import config
    if not config.ADMIN_API_KEY:
        return  # open in dev mode
    provided: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[7:].strip()
    if not provided and key:
        provided = key
    if provided != config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass ?key=YOUR_KEY or Authorization: Bearer header.",
        )


# ── Dashboard HTML ────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stayforlong · Conversations Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f7;color:#1d1d1f;height:100vh;display:flex;flex-direction:column;overflow:hidden}
a{color:inherit;text-decoration:none}

/* ── Header ── */
header{background:linear-gradient(135deg,#c40a4c,#f60e5f);color:#fff;padding:14px 28px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
header h1{font-size:18px;font-weight:700;letter-spacing:-0.3px}
header p{font-size:12px;opacity:.75;margin-top:2px}
#last-refresh{font-size:11px;opacity:.7}

/* ── Stats bar ── */
.stats{display:flex;gap:12px;padding:14px 28px;background:#fff;border-bottom:1px solid #e8e8ed;flex-shrink:0}
.stat{flex:1;background:#f5f5f7;border-radius:10px;padding:12px 16px;text-align:center}
.stat-val{font-size:26px;font-weight:700;color:#f60e5f;line-height:1}
.stat-lbl{font-size:11px;color:#6e6e73;margin-top:4px}

/* ── Layout ── */
.main{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
.sidebar{width:380px;background:#fff;border-right:1px solid #e8e8ed;display:flex;flex-direction:column;flex-shrink:0}
.sidebar-head{padding:14px 16px;border-bottom:1px solid #e8e8ed;flex-shrink:0}
.search-row{display:flex;gap:8px;margin-bottom:10px}
.search-row input{flex:1;padding:7px 11px;border:1px solid #d2d2d7;border-radius:8px;font-size:13px;outline:none}
.search-row input:focus{border-color:#f60e5f}
.search-row button{padding:7px 14px;background:#f60e5f;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;white-space:nowrap}
.search-row button:hover{background:#d40050}
.filter-row{display:flex;gap:8px}
.filter-row select{flex:1;padding:6px 8px;border:1px solid #d2d2d7;border-radius:7px;font-size:12px;outline:none;background:#fff}

.conv-list{flex:1;overflow-y:auto}
.conv-item{padding:12px 16px;border-bottom:1px solid #f0f0f5;cursor:pointer;transition:background .12s;border-left:3px solid transparent}
.conv-item:hover{background:#fafafa}
.conv-item.active{background:#fff0f5;border-left-color:#f60e5f}
.conv-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px}
.conv-uid{font-size:12px;font-weight:600;font-family:monospace;color:#1d1d1f;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px}
.conv-time{font-size:11px;color:#9e9e9e;flex-shrink:0;margin-left:8px}
.conv-meta{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.badge{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:600;white-space:nowrap}
.b-lang{background:#e8e8ed;color:#494949}
.b-active{background:#d1fae5;color:#065f46}
.b-closed{background:#f1f1f4;color:#6b7280}
.b-agent{background:#fce7f3;color:#9d174d}
.conv-count{font-size:11px;color:#9e9e9e}

/* ── Detail panel ── */
.detail{flex:1;display:flex;flex-direction:column;background:#f5f5f7;overflow:hidden}
.detail-head{background:#fff;padding:16px 24px;border-bottom:1px solid #e8e8ed;flex-shrink:0}
.detail-head h3{font-size:14px;margin-bottom:8px;display:flex;align-items:center;gap:10px}
.detail-head h3 code{font-size:12px;background:#f5f5f7;padding:2px 8px;border-radius:5px;cursor:pointer;color:#f60e5f}
.detail-head h3 code:hover{background:#fce7f3}
.detail-meta{display:flex;flex-wrap:wrap;gap:14px}
.detail-meta span{font-size:12px;color:#6e6e73}
.detail-meta strong{color:#1d1d1f}
.detail-meta a{color:#f60e5f;font-family:monospace;font-size:11px}

.msgs-wrap{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:10px}
.msg{display:flex;flex-direction:column;max-width:72%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.assistant{align-self:flex-start;align-items:flex-start}
.msg-lbl{font-size:10px;color:#9d174d;font-weight:700;margin-bottom:2px;text-transform:uppercase;letter-spacing:.3px}
.msg-bubble{padding:9px 13px;border-radius:12px;font-size:13px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.msg.user .msg-bubble{background:#f60e5f;color:#fff;border-bottom-right-radius:3px}
.msg.assistant .msg-bubble{background:#fff;color:#1d1d1f;border:1px solid #e8e8ed;border-bottom-left-radius:3px}
.msg-ts{font-size:10px;color:#b0b0b8;margin-top:3px}

.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;color:#b0b0b8}
.empty-state p{font-size:14px}
.empty-state svg{width:56px;height:56px;opacity:.35}

.placeholder{text-align:center;padding:32px;color:#9e9e9e;font-size:13px}
.err{color:#e63946}

/* scrollbar */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#d2d2d7;border-radius:3px}
</style>
</head>
<body>

<header>
  <div>
    <h1>Stayforlong · Conversations Dashboard</h1>
    <p>Customer Experience Monitor</p>
  </div>
  <span id="last-refresh"></span>
</header>

<div class="stats">
  <div class="stat"><div class="stat-val" id="s-total">—</div><div class="stat-lbl">Total conversations</div></div>
  <div class="stat"><div class="stat-val" id="s-today">—</div><div class="stat-lbl">Started today</div></div>
  <div class="stat"><div class="stat-val" id="s-users">—</div><div class="stat-lbl">Unique users</div></div>
  <div class="stat"><div class="stat-val" id="s-active">—</div><div class="stat-lbl">Active now</div></div>
</div>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-head">
      <div class="search-row">
        <input id="q-user" type="text" placeholder="Search by user ID…" />
        <button onclick="doSearch()">Search</button>
      </div>
      <div class="filter-row">
        <select id="f-status" onchange="doRefresh()">
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="closed">Closed</option>
        </select>
        <select id="f-lang" onchange="doRefresh()">
          <option value="">All languages</option>
          <option value="es">Spanish</option>
          <option value="en">English</option>
          <option value="pt">Portuguese</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="ca">Catalan</option>
        </select>
      </div>
    </div>
    <div class="conv-list" id="conv-list"><div class="placeholder">Loading…</div></div>
  </div>

  <div class="detail" id="detail">
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <p>Select a conversation to view the full thread</p>
    </div>
  </div>
</div>

<script>
var API_KEY = new URLSearchParams(location.search).get('key') || '';
var activeConvId = null;
var activeUserId = null;

function hdrs() {
  var h = {'Content-Type':'application/json'};
  if (API_KEY) h['Authorization'] = 'Bearer ' + API_KEY;
  return h;
}

function rel(iso) {
  if (!iso) return '—';
  var s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

function fmt(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

var LANG = {es:'Spanish',en:'English',pt:'Portuguese',fr:'French',de:'German',it:'Italian',ca:'Catalan'};
function langName(c) { return LANG[c] || c || '—'; }

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function shortId(id) {
  if (!id) return '—';
  return id.length > 24 ? id.slice(0,10) + '…' + id.slice(-6) : id;
}

/* ── Stats ── */
async function loadStats() {
  try {
    var r = await fetch('/admin/api/stats', {headers: hdrs()});
    if (!r.ok) return;
    var d = await r.json();
    document.getElementById('s-total').textContent = d.total_conversations ?? '—';
    document.getElementById('s-today').textContent = d.conversations_today ?? '—';
    document.getElementById('s-users').textContent = d.total_users ?? '—';
    document.getElementById('s-active').textContent = d.active_conversations ?? '—';
  } catch(e){}
}

/* ── Conversations list ── */
async function loadList(userId) {
  var status = document.getElementById('f-status').value;
  var lang   = document.getElementById('f-lang').value;

  var url = userId
    ? '/admin/api/users/' + encodeURIComponent(userId) + '/conversations'
    : '/admin/api/conversations';

  var params = [];
  if (status) params.push('status=' + encodeURIComponent(status));
  if (lang)   params.push('lang='   + encodeURIComponent(lang));
  if (params.length) url += '?' + params.join('&');

  var list = document.getElementById('conv-list');
  try {
    var r = await fetch(url, {headers: hdrs()});
    if (r.status === 401) {
      list.innerHTML = '<div class="placeholder err">Unauthorized — add ?key=YOUR_API_KEY to the URL.</div>';
      return;
    }
    if (!r.ok) {
      list.innerHTML = '<div class="placeholder err">Error ' + r.status + '</div>';
      return;
    }
    var convs = await r.json();
    renderList(convs);
  } catch(e) {
    list.innerHTML = '<div class="placeholder err">Network error</div>';
  }
}

function renderList(convs) {
  var list = document.getElementById('conv-list');
  if (!convs.length) {
    list.innerHTML = '<div class="placeholder">No conversations found.</div>';
    return;
  }
  list.innerHTML = convs.map(function(c) {
    var agents = (c.agents_used || []).filter(function(a){
      return a && a !== 'Stayforlong' && a !== 'Stayforlong Assistant' && a !== 'triage_agent';
    });
    var agentBadges = agents.slice(0,3).map(function(a){
      return '<span class="badge b-agent">' + esc(a) + '</span>';
    }).join('');
    return (
      '<div class="conv-item' + (c.id===activeConvId?' active':'') + '" data-id="' + esc(c.id) + '">' +
      '<div class="conv-header">' +
        '<span class="conv-uid" title="' + esc(c.user_id) + '">' + esc(shortId(c.user_id)) + '</span>' +
        '<span class="conv-time">' + rel(c.last_activity_at) + '</span>' +
      '</div>' +
      '<div class="conv-meta">' +
        '<span class="badge b-lang">' + langName(c.language) + '</span>' +
        '<span class="badge ' + (c.status==='active'?'b-active':'b-closed') + '">' + esc(c.status||'?') + '</span>' +
        '<span class="conv-count">' + (c.message_count||0) + ' msgs</span>' +
        agentBadges +
      '</div>' +
      '</div>'
    );
  }).join('');

  list.querySelectorAll('.conv-item').forEach(function(el) {
    el.addEventListener('click', function() { selectConv(el.dataset.id); });
  });
}

/* ── Conversation detail ── */
async function selectConv(id) {
  activeConvId = id;

  // highlight selected item
  document.querySelectorAll('#conv-list .conv-item').forEach(function(el){
    el.classList.toggle('active', el.dataset.id === id);
  });

  var panel = document.getElementById('detail');
  panel.innerHTML = '<div class="empty-state"><p>Loading conversation…</p></div>';

  try {
    var [cr, mr] = await Promise.all([
      fetch('/admin/api/conversations/' + id, {headers: hdrs()}),
      fetch('/admin/api/conversations/' + id + '/messages?limit=200', {headers: hdrs()})
    ]);
    if (!cr.ok || !mr.ok) {
      panel.innerHTML = '<div class="empty-state"><p class="err">Error loading conversation.</p></div>';
      return;
    }
    var conv = await cr.json();
    var msgs = await mr.json();
    renderDetail(conv, msgs);
  } catch(e) {
    panel.innerHTML = '<div class="empty-state"><p class="err">Network error.</p></div>';
  }
}

function renderDetail(conv, msgs) {
  var agents = (conv.agents_used||[]).filter(Boolean);
  var panel = document.getElementById('detail');

  var head = (
    '<div class="detail-head">' +
    '<h3>Conversation <code id="copy-id" title="Click to copy">' + esc(conv.id) + '</code></h3>' +
    '<div class="detail-meta">' +
      '<span><strong>User:</strong> <a href="#" onclick="jumpUser(' + JSON.stringify(conv.user_id) + ');return false">' + esc(conv.user_id) + '</a></span>' +
      '<span><strong>Language:</strong> ' + langName(conv.language) + '</span>' +
      '<span><strong>Status:</strong> ' + esc(conv.status||'—') + '</span>' +
      '<span><strong>Messages:</strong> ' + (conv.message_count||0) + '</span>' +
      '<span><strong>Started:</strong> ' + fmt(conv.started_at) + '</span>' +
      '<span><strong>Last activity:</strong> ' + fmt(conv.last_activity_at) + '</span>' +
      (agents.length ? '<span><strong>Agents:</strong> ' + esc(agents.join(', ')) + '</span>' : '') +
    '</div>' +
    '</div>'
  );

  var thread = '<div class="msgs-wrap" id="msgs-wrap">';
  if (!msgs.length) {
    thread += '<div class="placeholder">No messages recorded.</div>';
  } else {
    msgs.forEach(function(m) {
      var isUser = m.role === 'user';
      var showLabel = !isUser && m.agent && m.agent !== 'Stayforlong Assistant';
      thread += (
        '<div class="msg ' + (isUser?'user':'assistant') + '">' +
        (showLabel ? '<div class="msg-lbl">' + esc(m.agent) + '</div>' : '') +
        '<div class="msg-bubble">' + esc(m.content||'') + '</div>' +
        '<div class="msg-ts">' + fmt(m.timestamp) + '</div>' +
        '</div>'
      );
    });
  }
  thread += '</div>';

  panel.innerHTML = head + thread;

  // Copy ID
  var copyEl = document.getElementById('copy-id');
  if (copyEl) {
    copyEl.addEventListener('click', function(){
      navigator.clipboard.writeText(conv.id).then(function(){
        copyEl.textContent = 'Copied!';
        setTimeout(function(){ copyEl.textContent = conv.id; }, 1500);
      });
    });
  }

  // Scroll to bottom
  setTimeout(function(){
    var w = document.getElementById('msgs-wrap');
    if (w) w.scrollTop = w.scrollHeight;
  }, 30);
}

/* ── User jump ── */
function jumpUser(uid) {
  document.getElementById('q-user').value = uid;
  activeUserId = uid;
  loadList(uid);
}

function doSearch() {
  activeUserId = document.getElementById('q-user').value.trim() || null;
  loadList(activeUserId);
}

function doRefresh() {
  loadList(activeUserId);
}

/* ── Init & auto-refresh ── */
function tick() {
  loadStats();
  loadList(activeUserId);
  document.getElementById('last-refresh').textContent = 'Refreshed ' + new Date().toLocaleTimeString();
}

document.getElementById('q-user').addEventListener('keydown', function(e){
  if (e.key === 'Enter') doSearch();
});

tick();
setInterval(tick, 30000);
</script>
</body>
</html>"""


# ── API routes ────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(_auth=Depends(require_admin)):
    return HTMLResponse(_DASHBOARD_HTML)


@router.get("/api/stats")
async def api_stats(_auth=Depends(require_admin)):
    from services.conversation_logger import conversation_logger
    return await conversation_logger.get_stats()


@router.get("/api/conversations")
async def api_list_conversations(
    status: Optional[str] = None,
    lang: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    result = await conversation_logger.list_conversations(limit=limit)
    convs = result.get("items", []) if isinstance(result, dict) else result
    if status:
        convs = [c for c in convs if c.get("status") == status]
    if lang:
        convs = [c for c in convs if c.get("language") == lang]
    return convs


@router.get("/api/conversations/{conv_id}")
async def api_get_conversation(conv_id: str, _auth=Depends(require_admin)):
    from services.conversation_logger import conversation_logger
    conv = await conversation_logger.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.get("/api/conversations/{conv_id}/messages")
async def api_get_messages(
    conv_id: str,
    limit: int = Query(100, ge=1, le=500),
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    return await conversation_logger.get_conversation_messages(conv_id, limit=limit)


@router.get("/api/users/{user_id}/conversations")
async def api_user_conversations(
    user_id: str,
    status: Optional[str] = None,
    lang: Optional[str] = None,
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    result = await conversation_logger.list_conversations(user_id=user_id)
    convs = result.get("items", []) if isinstance(result, dict) else result
    if status:
        convs = [c for c in convs if c.get("status") == status]
    if lang:
        convs = [c for c in convs if c.get("language") == lang]
    return convs
