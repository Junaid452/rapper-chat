import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from django.shortcuts import render, redirect
from django.http import (
    HttpResponse, JsonResponse, StreamingHttpResponse, QueryDict,
    HttpResponseBadRequest, HttpResponseForbidden, HttpResponseNotFound, HttpResponseNotAllowed,
)
from django.conf import settings
from django.urls import reverse
from django.utils.html import escape
from django.utils.http import url_has_allowed_host_and_scheme, urlencode
from django.views.decorators.csrf import csrf_exempt
from django_cotton import render_component
from django_htmx.http import trigger_client_event
from langchain_core.messages import AIMessageChunk
from langgraph.errors import GraphRecursionError
from polar_sdk.webhooks import WebhookVerificationError, WebhookUnknownTypeError

from . import billing
from . import evaluator
from . import tracing
from .models import Conversation, Message
from .llm import build_agent, build_history, generate_title, RECURSION_LIMIT
from .utils import render_markdown, sse_frame

# How often the streaming generator checks whether the user asked it to stop.
# The browser freezes the reply on click, so this no longer gates how responsive
# stopping feels -- it gates how much extra text is generated (and billed) after
# the click, and how far the saved reply runs past what the user saw.
STOP_POLL_INTERVAL = 0.1  # seconds

# Shown when a reply produced no text at all -- the model returned an empty
# stream. The message row is still created (with empty content) so the
# conversation isn't left looking unanswered and won't re-stream on the next
# pane load. A stop always captures at least one token, since the stop check
# only runs after a token has been yielded, so it does not normally land here.
STOPPED_EMPTY_HTML = '<p class="italic text-base-content/50">Response stopped.</p>'

# Placeholder shown while a tool runs. The model emits no prose while it decides
# to call one, and the search itself takes seconds, so without this the bubble
# sits empty and the reply looks hung.
SEARCHING_HTML = (
    '<span class="loading loading-dots loading-sm align-middle text-accent"></span>'
    '<span class="ml-2 text-sm opacity-70">{label}</span>'
)
# Model-written, so it is escaped before it lands in the status line, and capped
# so a runaway query can't push the reply off screen.
MAX_STATUS_QUERY_CHARS = 80

# How much of the conversation the prompt evaluator is shown. It has to judge
# whether a short follow-up is clear *in context*, which needs the thread, but
# not all of it -- the turns immediately before it are what make "what about the
# second one?" answerable or not.
def _recent_transcript(conversation, exclude_pk):
    """A plain-text tail of the conversation for the evaluator.

    Critiques are left out: the evaluator judging its own past output drifts,
    each round agreeing harder with the last.
    """
    turns = list(conversation.messages.dialogue().exclude(pk=exclude_pk))

    lines, budget = [], settings.CHAT_EVALUATOR_CONTEXT_TOTAL_CHARS
    # Newest first so the total budget is spent on the turns nearest the message
    # being judged, then flipped back into reading order.
    for m in reversed(turns[-settings.CHAT_EVALUATOR_CONTEXT_TURNS:]):
        body = ' '.join(m.content.split())[:settings.CHAT_EVALUATOR_CONTEXT_CHARS]
        line = f"{'User' if m.role == 'user' else 'Assistant'}: {body}"
        if len(line) > budget:
            break
        budget -= len(line)
        lines.append(line)
    return '\n'.join(reversed(lines))


# Ceiling on waiting for a critique that has not come back by the time the answer
# has. Past this the reply ships without it: the critique is commentary, and
# holding a finished answer hostage to it would be the wrong trade.
EVAL_WAIT_SECONDS = 20


def _persist_critique(request, conversation, kind, score, text):
    """Store one critique and return the bubble HTML, or '' if there was none.

    Returns the rendered component rather than the row because the caller's only
    use for it is an SSE frame, and rendering here keeps the two representations
    of a critique -- stored and streamed -- coming from the same template.
    """
    # A score with no prose is still a verdict worth showing -- the bubble
    # renders the badge on its own. Only a critique with neither is nothing.
    if not (text or '').strip() and score is None:
        return ''
    message = Message.objects.create(
        conversation=conversation, role='evaluator', critique_of=kind,
        score=score, content=text, content_html=render_markdown(text),
    )
    return render_component(request, 'evaluator-bubble', message=message, entering=True)


def _same_origin_path(url, request):
    """The path+query of url if it points back at this site, else None. Returning
    only the path (never the host) means the result is relative by construction,
    so it can't send anyone off-site even if the check above it ever slipped."""
    if not url:
        return None
    if not url_has_allowed_host_and_scheme(url, allowed_hosts={request.get_host()},
                                           require_https=request.is_secure()):
        return None
    parts = urlsplit(url)
    return parts.path + (f'?{parts.query}' if parts.query else '')


def _with_reason(url, reason):
    parts = urlsplit(url)
    query = QueryDict(parts.query, mutable=True)
    query['reason'] = reason
    return f'{parts.path}?{query.urlencode()}'


def _reauth_redirect(request):
    """Bounce a browser whose session cookie expired through the chat shell,
    which loads clerk-js, renews the cookie, and forwards on to ?next=.

    A safe request resumes exactly where it was headed, so an expired cookie
    costs nothing but a flicker. An unsafe one can't be replayed that way --
    quietly re-submitting a billing action the user can no longer see is far
    worse than asking them to click again -- so it returns to the page holding
    the form, flagged so that page can explain why nothing happened.
    """
    if request.method in ('GET', 'HEAD'):
        next_url = request.get_full_path()
    else:
        referer = _same_origin_path(request.META.get('HTTP_REFERER', ''), request)
        next_url = _with_reason(referer or reverse('index'), 'session_expired')
    return redirect(f"{reverse('index')}?{urlencode({'next': next_url})}")


def require_clerk_auth(view_func):
    """Reject the request unless the Clerk middleware authenticated the user.

    A browser navigating here is sent through the re-auth hop above rather than
    shown the JSON body, so an expired session self-heals instead of dead-ending
    on a bare error page. Everything else -- htmx, EventSource -- keeps the 401,
    because the client's retry handler is wired to that status (see app.js).
    """
    def _wrapped_view(request, *args, **kwargs):
        if not request.clerk_user_id:
            if not request.htmx and 'text/html' in request.headers.get('Accept', ''):
                return _reauth_redirect(request)
            return JsonResponse({'error': 'Unauthorized. Please sign in.'}, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def _user_conversations(request):
    return Conversation.objects.filter(owner=request.clerk_user_id)


def _reject_if_over_quota(request):
    """None if the user may send another message; otherwise a response that sends
    them to the pricing page. Both send views are hx-post forms, so a plain 302
    would just get swapped into the target div by htmx instead of navigating the
    browser -- HX-Redirect is the header htmx honors to force a full redirect.
    The ?reason=limit query param lets the pricing page distinguish this from
    someone browsing plans proactively, so it doesn't claim a cap was hit when
    it wasn't."""
    if billing.has_quota(request.clerk_user_id):
        return None
    response = HttpResponse(status=204)
    response['HX-Redirect'] = '/pricing/?reason=limit'
    return response


def _rendered_messages(conversation):
    """Stored messages ready for templating — every markdown-bearing turn
    pre-rendered to HTML.

    That means critiques as well as replies. Only the user's own turns are shown
    as plain text, so the test here is the role that is *not* rendered rather
    than a list of the ones that are: an evaluator bubble replayed with no
    content_html renders as an empty shell, which is exactly what a role
    allowlist here caused once already.
    """
    items = []
    for m in conversation.messages.all():
        html = ''
        if m.role != 'user':
            if not m.content_html and m.content:
                # Backfills rows saved before content_html existed; new rows are
                # already populated when written and skip straight past this.
                m.content_html = render_markdown(m.content)
                m.save(update_fields=['content_html'])
            html = m.content_html
        items.append({
            'role': m.role,
            'content': m.content,
            'content_html': html,
            # Only meaningful on critiques, but the bubble reads both: without
            # them a replayed critique loses its score badge and is labelled as
            # a critique of the reply no matter which one it is.
            'critique_of': m.critique_of,
            'score': m.score,
        })
    return items


def _list_partial(request, active_id=None):
    """Render the sidebar conversation list (an htmx-swappable partial)."""
    return HttpResponse(render_component(request, 'conversation-list',
        conversations=_user_conversations(request),
        active_id=active_id,
    ))


def index(request):
    """Serve the single-page chat shell."""
    return render(request, 'chat/index.html', {
        'CLERK_PUBLISHABLE_KEY': settings.CLERK_PUBLISHABLE_KEY,
    })


@require_clerk_auth
def conversations_partial(request):
    """GET: the sidebar list, loaded by htmx once Clerk is ready."""
    if not request.htmx:
        return redirect('index')
    return _list_partial(request)


@csrf_exempt
@require_clerk_auth
def conversation_detail(request, conversation_id):
    """Rename (PUT) or delete (DELETE) a conversation; returns the refreshed list."""
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.clerk_user_id)
    except Conversation.DoesNotExist:
        return HttpResponseNotFound('Conversation not found')

    if request.method == 'DELETE':
        conversation.delete()
        return _list_partial(request)

    if request.method == 'PUT':
        # htmx sends the body form-encoded even for PUT, so parse it as a QueryDict.
        new_title = (QueryDict(request.body).get('title') or '').strip()
        if new_title:
            conversation.title = new_title[:100]
            conversation.save()
        return _list_partial(request, active_id=conversation.id)

    return HttpResponseNotAllowed(['PUT', 'DELETE'])


def _pane_response(request, conversation, refresh_list=False):
    """Render the chat pane (topbar + messages + input) for a conversation, telling
    the client which conversation is now active via a conversation-active client
    event. Optionally piggybacks an out-of-band refresh of the sidebar list (used
    when a new chat is created)."""
    # Excludes critiques: they are written around a turn, not as one, so a
    # conversation whose last row is a critique is still answered (or still
    # waiting) according to the turn before it.
    last = conversation.messages.dialogue().last()
    conversations = _user_conversations(request) if refresh_list else None
    active_id = conversation.id if refresh_list else None
    response = HttpResponse(render_component(request, 'chat-pane',
        conversation=conversation,
        messages=_rendered_messages(conversation),
        # If the latest turn is the user's, render an SSE-wired assistant bubble that
        # streams the pending reply as soon as the pane is inserted.
        pending=bool(last and last.role == 'user'),
        refresh_list=refresh_list,
        conversations=conversations,
        active_id=active_id,
    ))
    trigger_client_event(response, 'conversation-active', {'id': conversation.id})
    return response


@require_clerk_auth
def conversation_pane(request, conversation_id):
    """GET: the chat pane for an existing conversation (selected from the sidebar)."""
    if not request.htmx:
        return redirect('index')
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.clerk_user_id)
    except Conversation.DoesNotExist:
        return HttpResponseNotFound('Conversation not found')
    return _pane_response(request, conversation)


@csrf_exempt
@require_clerk_auth
def start_conversation(request):
    """POST: create a conversation from the first message, then return its pane
    (with the pending reply streaming) plus an OOB refresh of the sidebar."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    content = (request.POST.get('content') or '').strip()
    if not content:
        return HttpResponseBadRequest('Content cannot be empty')

    quota_response = _reject_if_over_quota(request)
    if quota_response is not None:
        return quota_response

    conversation = Conversation.objects.create(owner=request.clerk_user_id, title=content[:60])
    Message.objects.create(conversation=conversation, role='user', content=content)
    return _pane_response(request, conversation, refresh_list=True)


# ── Streaming (SSE) message flow ────────────────────────────────────────────
# A reply is produced in two steps because EventSource can only issue GET requests:
# (1) create_message saves the user turn and returns the message pair (user bubble +
# an SSE-wired assistant bubble), then (2) stream_reply streams the answer.

@csrf_exempt
@require_clerk_auth
def create_message(request, conversation_id):
    """POST: save the user's message and return the message-pair partial. The
    assistant bubble it contains connects to stream_reply to receive the answer."""
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.clerk_user_id)
    except Conversation.DoesNotExist:
        return HttpResponseNotFound('Conversation not found')
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    content = (request.POST.get('content') or '').strip()
    if not content:
        return HttpResponseBadRequest('Content cannot be empty')

    quota_response = _reject_if_over_quota(request)
    if quota_response is not None:
        return quota_response

    Message.objects.create(conversation=conversation, role='user', content=content)
    return HttpResponse(render_component(request, 'message-pair',
        conversation=conversation,
        content=content,
    ))


def _chunk_text(chunk):
    """The plain text carried by an AI message chunk.

    `.content` is a string for most providers but a list of typed blocks for
    some, so take only the text blocks -- anything else (reasoning traces, tool
    call fragments) is machine detail that must not reach the bubble.
    """
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # A list element is either a typed block or, per the message schema, a
        # bare string -- which is already the text and would otherwise be dropped.
        return ''.join(
            block if isinstance(block, str)
            else block.get('text', '') if block.get('type') == 'text'
            else ''
            for block in content
        )
    return ''


def _status_frames(payload):
    """SSE `status` frames for one LangGraph node update.

    Names the search when the model asks for a tool, then switches to reading
    once the results are back. Neither frame clears the line: the model stays
    silent for several seconds after a search while it reads, and a line cleared
    at the tool boundary would leave that gap blank, which reads as a hang. The
    caller clears it on the first token of the answer instead.
    """
    for node, update in (payload or {}).items():
        for msg in (update or {}).get('messages') or []:
            calls = getattr(msg, 'tool_calls', None)
            if calls:
                raw = (calls[0].get('args') or {}).get('query') or ''
                query = ' '.join(raw.split())[:MAX_STATUS_QUERY_CHARS]
                label = (f'Searching the web for &ldquo;{escape(query)}&rdquo;&hellip;'
                         if query else 'Searching the web&hellip;')
                yield sse_frame('status', SEARCHING_HTML.format(label=label))
            elif node == 'tools':
                # A second label rather than a second spinner: the wait spans two
                # phases, and a caption that changes is what distinguishes work
                # in progress from a stalled request.
                yield sse_frame(
                    'status', SEARCHING_HTML.format(label='Reading the results&hellip;'))


@require_clerk_auth
def stream_reply(request, conversation_id):
    """Stream the assistant's reply to the latest user message token-by-token via SSE.

    Emits `token` events as text arrives, persists the assembled reply once the
    stream finishes, then emits a final `done` event carrying the rendered HTML.
    """
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.clerk_user_id)
    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)

    # Guard against EventSource auto-reconnect regenerating a reply: only stream when
    # the most recent message is a user turn still awaiting an answer. Critiques
    # are excluded -- the prompt critique is written mid-stream, so counting it
    # would make a reconnect think the turn had already been answered.
    last = conversation.messages.dialogue().last()
    if last is None or last.role != 'user':
        return StreamingHttpResponse(sse_frame('done', ''), content_type='text/event-stream')

    # Build history eagerly (before streaming starts) so the DB query isn't deferred
    # into the generator.
    history = build_history(conversation)

    # Clear any flag left behind by an earlier stream so a stale request can't
    # cut this one short. Runs in the view body, which executes before the
    # generator is first iterated, so it can't race a genuine stop.
    Conversation.objects.filter(pk=conversation.pk).update(stop_requested=False)

    question = last.content
    transcript = _recent_transcript(conversation, exclude_pk=last.pk)

    def event_stream():
        """The reply, wrapped in one Langfuse trace.

        One trace per turn and one session per conversation: a conversation
        never signals that it has ended, so a trace spanning one would stay open
        indefinitely, while a turn always closes. The session id ties them back
        together in Langfuse's session view.

        The root span's input and output become the trace's, which is what the
        traces table shows -- so they are the question and the finished reply,
        not the internals of this generator.
        """
        with tracing.turn(
            'chat-turn',
            session_id=str(conversation.id),
            user_id=request.clerk_user_id,
            input=question,
            # Tags are fixed at creation, which suits naming the feature a trace
            # came from -- it separates answering a message from naming a
            # conversation in dashboards, and both are known before either runs.
            tags=['chat'],
        ) as turn_span:
            yield from _turn_events(turn_span)

    def _turn_events(turn_span):
        collected = []
        # Whether a status line is currently on screen, so it is cleared exactly
        # once -- when the answer starts, when the run ends without one, or on
        # the error paths below.
        status_active = False
        prompt_eval_sent = False
        # Whether the reader cut the reply short. Only used to label the turn for
        # Langfuse: a half-written answer is not the model's work to be judged on.
        stopped = False

        # Started before the agent and read while it streams, so the critique of
        # the question is written *alongside* the answer rather than in front of
        # it. Sequencing them would add its whole latency to the wait for the
        # first token, which is the one number in this endpoint worth guarding.
        pool = ThreadPoolExecutor(max_workers=1) if evaluator.is_enabled() else None
        prompt_future = (pool.submit(evaluator.evaluate_prompt, question, transcript)
                         if pool else None)

        def prompt_eval_frame(block=False):
            """The prompt critique's SSE frame once available, else None. Emits at
            most once; `block` waits for a critique still in flight."""
            nonlocal prompt_eval_sent
            if prompt_eval_sent or prompt_future is None:
                return None
            if not block and not prompt_future.done():
                return None
            prompt_eval_sent = True
            try:
                score, text = prompt_future.result(timeout=EVAL_WAIT_SECONDS)
            except Exception as e:
                print(f'Prompt evaluation failed: {e}')
                return None
            html = _persist_critique(request, conversation, 'prompt', score, text)
            return sse_frame('prompt_eval', html) if html else None

        try:
            next_stop_check = time.monotonic() + STOP_POLL_INTERVAL
            # "messages" carries the tokens; "updates" carries the node
            # transitions, which are the only signal that a tool is running --
            # the model produces no text while it decides to call one.
            stream = build_agent().stream(
                {'messages': history},
                stream_mode=['messages', 'updates'],
                config={
                    'recursion_limit': RECURSION_LIMIT,
                    # Nests the whole graph run -- every model call and every
                    # web_search hop -- under the turn span opened above.
                    'callbacks': tracing.callbacks(),
                    # Otherwise the graph arrives named "LangGraph", which says
                    # nothing about which of the app's three model calls it is.
                    'run_name': 'reply-agent',
                },
            )
            for mode, payload in stream:
                if mode == 'updates':
                    for frame in _status_frames(payload):
                        status_active = True
                        yield frame
                else:
                    chunk, _meta = payload
                    # Tool results arrive on this stream too. They are input for
                    # the model's next turn, not part of the reply, so only the
                    # model's own chunks are forwarded.
                    if isinstance(chunk, AIMessageChunk):
                        text = _chunk_text(chunk)
                        if text:
                            if status_active:
                                # The answer itself is the first thing worth
                                # showing in place of the status, so the line
                                # survives the whole tool call and the model's
                                # silence after it, and goes exactly here.
                                status_active = False
                                yield sse_frame('status', '')
                            collected.append(text)
                            yield sse_frame('token', text)

                # Time-based rather than per-event: tokens arrive far faster than
                # a person can click, so polling every token would spend a query
                # per token for no extra responsiveness. Checked on every event,
                # not just tokens, so a stop still lands during a search -- though
                # an HTTP call already in flight runs to completion first.
                now = time.monotonic()
                if now >= next_stop_check:
                    next_stop_check = now + STOP_POLL_INTERVAL
                    # Checked on the same tick as the stop flag rather than per
                    # token: the critique lands whenever it lands, and a tenth of
                    # a second either way is not perceptible.
                    frame = prompt_eval_frame()
                    if frame:
                        yield frame
                    if Conversation.objects.filter(pk=conversation.pk, stop_requested=True).exists():
                        stopped = True
                        break

            # Stopped mid-search, or the run ended having only ever called tools:
            # either way no token arrived to take the line down, and leaving a
            # spinner above a finished reply would read as still running.
            if status_active:
                yield sse_frame('status', '')
        except GraphRecursionError:
            # The model kept calling tools instead of answering and hit the cap.
            # Whatever it had already said is kept, so this falls through to the
            # normal persist path rather than being treated as a failure.
            print(f'Agent hit the {RECURSION_LIMIT}-step limit without answering')
            # Not an error -- the turn still returns whatever the model said --
            # but the one case worth being able to filter for, since it means
            # the recursion limit, not the model, ended the reply.
            turn_span.update(level='WARNING',
                             status_message='hit the agent recursion limit',
                             metadata={'recursion_limit_hit': True})
            yield sse_frame('status', '')
            collected.append(
                '\n\n*I kept searching without reaching an answer — '
                'try narrowing the question.*'
            )
        except Exception as e:
            print(f"LangChain/OpenRouter streaming error: {e}")
            # Recorded explicitly because the exception is handled here rather
            # than allowed to propagate, so the span would otherwise close as a
            # success and this turn would not show up when filtering for errors.
            turn_span.update(output={'error': str(e)}, level='ERROR',
                             status_message=str(e),
                             metadata={'turn_outcome': 'error'})
            yield sse_frame('status', '')
            # The question was still worth critiquing even though answering it
            # failed, and the slot for it is already on the page.
            frame = prompt_eval_frame(block=True)
            if frame:
                yield frame
            yield sse_frame('error', str(e))
            # Always close with `done` too: the bubble's sse-close="done" is what stops
            # EventSource from auto-reconnecting and re-hitting the API in a loop.
            # The payload replaces the bubble, so surface the failure there.
            yield sse_frame('done', '<p class="text-error">Sorry — the reply failed to generate. Please try again.</p>')
            return
        finally:
            if pool:
                pool.shutdown(wait=False)

        # Ordered before the answer is persisted so the critique of the question
        # keeps its place in the transcript ahead of the reply it prompted.
        frame = prompt_eval_frame(block=True)
        if frame:
            yield frame

        # A stopped reply takes this same path deliberately: it is persisted and
        # rendered exactly like a complete one, so it survives a reload and reads
        # back as an ordinary (if short) assistant turn.
        full = ''.join(collected)
        full_html = render_markdown(full) if full.strip() else STOPPED_EMPTY_HTML
        # The reply as the trace's output. Set here rather than at the end of the
        # generator so a turn the reader stopped early still records what was
        # actually produced.
        #
        # turn_outcome says whether this turn is a fair thing to grade. The
        # LLM-as-a-judge evaluator in Langfuse filters on it, because the three
        # other endings all put something in `output` that is not an answer: an
        # error object, a half-written reply the reader cancelled, or nothing at
        # all. Judged unfiltered they would read as the model answering badly,
        # and the score would track how often people hit stop rather than how
        # good the answers are. 'stopped' wins over an empty string -- a stop
        # that landed before the first token is still a stop, not a failure to
        # speak.
        turn_span.update(
            output=full,
            metadata={'turn_outcome': (
                'stopped' if stopped else 'answered' if full.strip() else 'empty'
            )},
        )
        Message.objects.create(conversation=conversation, role='assistant', content=full, content_html=full_html)
        conversation.stop_requested = False
        conversation.save(update_fields=['stop_requested', 'updated_at'])

        # Before `done`, not after: `done` is the event named in sse-close, so it
        # tears the connection down and anything sent behind it is never
        # delivered. The reader loses nothing by the ordering -- the answer is
        # already fully drawn by the live render, and `done` only swaps in the
        # server's copy of the same thing.
        #
        # Skipped for an empty or stopped reply, where a critique would be
        # scoring the user's interruption rather than the model's work.
        if full.strip() and evaluator.is_enabled():
            try:
                score, text = evaluator.evaluate_response(question, full, transcript)
            except Exception as e:
                print(f'Response evaluation failed: {e}')
            else:
                html = _persist_critique(request, conversation, 'response', score, text)
                if html:
                    yield sse_frame('response_eval', html)

        yield sse_frame('done', full_html)

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # disable proxy/server buffering so tokens flush immediately
    return response


@csrf_exempt
@require_clerk_auth
def conversation_title(request, conversation_id):
    """POST: replace the placeholder title with one the model writes.

    Called by the client once the first reply has finished, rather than during
    it -- naming the chat is not something the user is waiting on, so it must
    not delay the reply. Only ever acts on the opening exchange; later calls are
    a no-op, which also means a rename can't be overwritten afterwards.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.clerk_user_id)
    except Conversation.DoesNotExist:
        return HttpResponseNotFound('Conversation not found')

    # One more turn than the opening exchange, so "the conversation has moved
    # on" stays distinguishable from "this is still the first question and its
    # answer" -- the latter being the only state this endpoint acts on.
    turns = list(conversation.messages.dialogue()[:3])
    if len(turns) != 2 or turns[0].role != 'user':
        return HttpResponse(status=204)

    # Its own trace rather than part of the turn's: naming a chat is a separate
    # request that lands after the reply has finished, so there is no turn left
    # open to attach it to. The shared session id is what still files it under
    # the conversation it named.
    with tracing.turn(
        'generate-title',
        session_id=str(conversation.id),
        user_id=request.clerk_user_id,
        input=turns[0].content,
        tags=['title'],
    ) as span:
        title = generate_title(turns[0].content, turns[1].content)
        span.update(output=title)
    if not title:
        return HttpResponse(status=204)

    conversation.title = title
    conversation.save(update_fields=['title'])
    response = _list_partial(request, active_id=conversation.id)
    trigger_client_event(response, 'conversation-titled',
                         {'id': conversation.id, 'title': title})
    return response


@csrf_exempt
@require_clerk_auth
def stop_stream(request, conversation_id):
    """POST: ask the in-flight reply for this conversation to stop early.

    This only raises the flag; the streaming generator notices it, breaks out of
    the token loop, and finishes through its normal completion path. Leaving the
    generator as the sole writer of the assistant message is what keeps a stop
    from racing a reply that was about to finish on its own and producing two
    rows. Safe to call repeatedly, and a no-op if nothing is streaming.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    updated = Conversation.objects.filter(
        id=conversation_id, owner=request.clerk_user_id
    ).update(stop_requested=True)
    if not updated:
        return HttpResponseNotFound('Conversation not found')
    return HttpResponse(status=204)


# ── Settings & billing (Polar.sh) ────────────────────────────────────────────

@require_clerk_auth
def settings_page(request):
    """GET: landing page for the sidebar's Settings link. Just a "Manage
    subscription" entry point today, kept separate from /pricing/ so that page
    stays dedicated to plan comparison/checkout rather than doubling as the
    general settings screen."""
    return render(request, 'chat/settings.html', {
        'CLERK_PUBLISHABLE_KEY': settings.CLERK_PUBLISHABLE_KEY,
    })


@require_clerk_auth
def pricing(request):
    """GET: plan comparison + checkout buttons. Reached either by choice (from
    /settings/) or automatically (via HX-Redirect, see _reject_if_over_quota)
    once a free user exhausts their quota -- ?reason=limit tells these apart so
    the headline doesn't claim a cap was hit when the user got here on their own.

    Already-active subscribers see their current plan marked and get "Switch to"
    buttons (change_plan) on the rest instead of "Subscribe" (start_checkout), so
    upgrading/downgrading changes their existing subscription rather than
    starting a second, competing one."""
    subscriber = billing.get_or_create_subscriber(request.clerk_user_id)
    current = subscriber if subscriber.status == 'active' else None

    plans = billing.plan_display()
    for plan in plans:
        plan['monthly_is_current'] = bool(current and current.plan == plan['key'] and current.billing_interval == 'month')
        plan['annual_is_current'] = bool(current and current.plan == plan['key'] and current.billing_interval == 'year')
        plan['monthly_is_pending'] = bool(current and current.pending_plan == plan['key'] and current.pending_billing_interval == 'month')
        plan['annual_is_pending'] = bool(current and current.pending_plan == plan['key'] and current.pending_billing_interval == 'year')

    return render(request, 'chat/pricing.html', {
        'plans': plans,
        'limit_reached': request.GET.get('reason') == 'limit',
        'session_expired': request.GET.get('reason') == 'session_expired',
        'current_subscriber': current,
        'CLERK_PUBLISHABLE_KEY': settings.CLERK_PUBLISHABLE_KEY,
    })


@require_clerk_auth
def start_checkout(request, plan_key):
    """POST: create a Polar hosted checkout for plan_key and send the browser to
    it. A plain (non-htmx) form post carrying a normal CSRF token, so Django's
    redirect is a normal top-level navigation to Polar's off-site checkout page."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        checkout_url = billing.create_checkout_session(
            request.clerk_user_id, plan_key,
            success_url=request.build_absolute_uri('/'),
        )
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    return redirect(checkout_url)


@require_clerk_auth
def change_plan(request, plan_key):
    """POST: move the caller's existing active subscription onto plan_key in
    place (see billing.change_plan), then send them back to the pricing page to
    see the result."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        billing.change_plan(request.clerk_user_id, plan_key)
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    return redirect('pricing')


@require_clerk_auth
def resume_with_plan(request, plan_key):
    """POST: undo a pending cancellation while switching to plan_key, effective
    at the next renewal rather than charged now (see billing.resume_with_plan)."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        billing.resume_with_plan(request.clerk_user_id, plan_key)
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    return redirect('pricing')


@require_clerk_auth
def cancel_subscription(request):
    """POST: schedule the caller's subscription to cancel at period end (see
    billing.cancel_subscription), then back to the pricing page."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        billing.cancel_subscription(request.clerk_user_id)
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    return redirect('pricing')


@require_clerk_auth
def resume_subscription(request):
    """POST: undo a pending cancellation (see billing.resume_subscription),
    then back to the pricing page."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        billing.resume_subscription(request.clerk_user_id)
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    return redirect('pricing')


@csrf_exempt
def polar_webhook(request):
    """POST: Polar calls this on subscription lifecycle events. Not behind
    @require_clerk_auth -- Polar, not a signed-in browser, is the caller here;
    trust comes from the webhook signature instead."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        event = billing.verify_webhook(request.body, dict(request.headers))
    except WebhookVerificationError:
        return HttpResponseForbidden('Invalid signature')
    except WebhookUnknownTypeError:
        # A newer Polar event type this SDK version doesn't know about yet --
        # acknowledge rather than error, so Polar doesn't retry it forever.
        return HttpResponse(status=200)
    billing.handle_webhook_event(event)
    return HttpResponse(status=200)
