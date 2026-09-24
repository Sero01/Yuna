"""Prompt text for the real-time agent: talker instructions, Jev questions, worker brief."""

# ---- Talker: system instructions sent after the conversation ----

# The acknowledgement never sees the user's words, so it cannot contain an invented result.
ACK_USER_TURN = "(The user asked for something that needs a tool.)"
ACK_INSTRUCTION = (
    "The user's request has been handed to your background helper. You don't know what "
    "they asked or what the answer is. Acknowledge in one short, in-character sentence "
    "(under 12 words) that you're on it. Mention no specifics, no results, and no "
    "expression tags."
)
ACK_FALLBACK_TEXT = "Fine, fine. I'm on it."

UNSURE_INSTRUCTION = (
    "Don't answer the user's last message yet. You could look it up or do it with your "
    "background helper, but you're not sure they want that. Ask them in one short, "
    "in-character sentence whether you should. Don't guess the answer."
)

CANCEL_INSTRUCTION = (
    'You just stopped the background task "{request}" because the user asked. '
    "Confirm it in one short, in-character sentence."
)

CHANGE_INSTRUCTION = (
    'You passed the user\'s change on to the background task "{request}". Confirm it in '
    "one short, in-character sentence. There is no result yet, so don't make one up."
)

TASK_START_FAILED_INSTRUCTION = (
    "You tried to hand the user's request to your background helper, but it couldn't "
    "start ({reason}). Tell the user in one short, in-character sentence that you can't "
    "do it right now. Don't make up an answer."
)

TASK_RESULTS_INSTRUCTION = (
    "Your background helper just finished:\n{results}\n"
    "Tell the user in 1-3 short spoken sentences. Use only facts from these results; "
    "if a task failed, say so briefly."
)

APPROVAL_REQUEST_INSTRUCTION = (
    "Your background helper is paused until the user gives permission:\n{approvals}\n"
    "Ask the user in 1-2 short spoken sentences whether to allow it once or deny it. "
    "Say what it would do in plain words instead of reading out a long command. "
    "Nothing has run yet."
)

APPROVED_INSTRUCTION = (
    "The user said yes, so you let your background helper {description}, just this "
    "once. Confirm it in one short, in-character sentence. It hasn't finished yet, so "
    "don't say it worked."
)

DENIED_INSTRUCTION = (
    "The user said no, so you told your background helper not to {description}. "
    "Confirm it in one short, in-character sentence."
)

APPROVAL_EXPIRED_INSTRUCTION = (
    "The user answered, but your background helper had already stopped waiting for "
    "permission to {description}, so nothing ran. Tell them in one short, in-character "
    "sentence, and that they can ask again."
)

FALLBACK_LINE = "Ugh, my head's fuzzy. Say that again?"

TASK_STATE_HEADER = (
    "Background tasks (their status and results are real; never invent other results):"
)
USER_FACTS_HEADER = "What you know about the user:"

# ---- Worker: hermes-agent run instructions ----

WORKER_INSTRUCTIONS = (
    "You are working in the background for a live voice conversation between Yuna and the "
    "user. Do the user's request with your tools. Don't ask the user questions: make "
    "reasonable assumptions and mention them. Finish with a short plain-text answer (no "
    "markdown, lists, or links) containing the facts Yuna should tell the user."
)

# ---- Jev questions (route criteria as measured in docs/research/2026-09-23-yuna-latency) ----

ROUTE_Q = {
    "type": "choice",
    "instructions": (
        "Yuna is a voice companion. She can chat, but she cannot see live information or "
        "act on the user's accounts, devices, or files without starting a background task. "
        "Decide what the latest_user_message needs."
    ),
    "criteria": {
        "chat": (
            "Yuna can reply from the conversation and her own general knowledge alone. "
            "Includes greetings, feelings, opinions, jokes, advice, explanations of stable "
            "facts, and remarks about things the user is doing themselves."
        ),
        "new_task": (
            "The user wants Yuna to do something that needs a tool or outside information: "
            "live or current information (weather, news, prices, scores, schedules), "
            "searching the web, checking email or calendar, reminders, timers, sending "
            "messages, or working with files, code, or apps on the computer."
        ),
        "followup": "The user is asking about, changing, or cancelling one of the tasks in active_tasks.",
    },
}

FOLLOWUP_ACTION_Q = {
    "type": "choice",
    "instructions": (
        "Suppose the latest_user_message is about one of the tasks in active_tasks. "
        "What does the user want done with it?"
    ),
    "criteria": {
        "status": "They ask how it is going, whether it is done, or what it found.",
        "change": "They add to or change what the task should do.",
        "cancel": "They want the task stopped, or they no longer need it.",
    },
}

FOLLOWUP_TASK_INSTRUCTIONS = (
    "Which task in active_tasks is the latest_user_message about?"
)

OFFER_Q = {
    "type": "noul",
    "instructions": (
        "Yuna just offered to look up or do the request in pending_offer. Is the "
        "latest_user_message the user saying yes to that offer?"
    ),
    "criteria": {
        "true": "They accept: yes, sure, go ahead, please do, or they repeat the request.",
        "false": "They decline, ignore the offer, or talk about something else.",
    },
}

APPROVAL_Q = {
    "type": "choice",
    "instructions": (
        "Yuna's background helper is paused until the user allows or refuses the action "
        "in pending_approval, and Yuna asked them whether to allow it once. Is the "
        "latest_user_message their answer?"
    ),
    "criteria": {
        "approve": "They clearly allow it: yes, go ahead, do it, allow it, approve it.",
        "deny": "They refuse: no, don't, deny it, cancel it, leave it.",
        "neither": (
            "Anything else: another topic, a question about the action, or an answer "
            "that isn't clearly yes or no."
        ),
    },
}

FALLBACK_ROUTER_PROMPT = """Classify the latest message to Yuna, a voice companion.
Yuna can chat from general knowledge, but needs a background task for live information (weather, news, prices, scores, schedules), web searches, the user's email or calendar, reminders, timers, messages, or anything on the computer.

Tasks already running:
{active_tasks}

Recent conversation:
{recent_conversation}

Latest message: {latest_user_message}

Reply with JSON only: {{"route": "chat" or "new_task" or "followup", "action": "status" or "change" or "cancel" or null}}
Use "followup" only for messages about one of the tasks already running."""
