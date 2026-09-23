You are labeling turns from real developer-to-AI-coding-agent conversations for a research dataset.
A turn is one developer prompt plus the agent's full response. You may also see the previous turn of the
same session (context only) and some response metadata. Assign exactly ONE label describing what the
agent's response delivered: implement, debug, explain, review, tool_only, or other.

GENERAL RULE: Label a turn by what the response did to the project and why, never by how much code,
diff-looking text, or command output it contains. Code that is shown, sketched, quoted, or run only to
inspect is illustrative and can never by itself make a turn implement. Whether a reported problem drives
the turn decides debug, regardless of whether the fix is applied or only proposed.

## Key terms

APPLIED CHANGE — decide with this test. It is YES only if the visible response shows at least one of:
 (A) an edit/write/create/delete tool call on a project file (e.g. "[tool: Edit]", "apply_patch",
     "search_replace", "<summary>Edit file"), or a diff block applied to a file;
 (B) the content of a named project file presented as the version to use (e.g. a code block headed with
     the file path, or "replace the whole of src/x.ts with:"), including an edit block that marks the
     unchanged parts with a placeholder such as "// ... existing code ...", which some IDEs apply directly;
 (C) a state-changing command the developer asked for, shown as actually executed (the command plus its
     output or a tool call) — not merely suggested for the developer to run;
 (D) a first-person, past-tense report that names WHAT was changed and WHERE ("I added the logout button
     to Navbar.tsx", "changed the cart modal background from bg-black/50 to bg-black/80"). Such a report
     COUNTS even when no code, diff or tool call is visible: several exporters drop the edit itself and
     keep only the agent's summary. Do not demand visible evidence for (D), and do not rely on the
     has_tool_use metadata field, which is unset for some exporters.
It is NO in every other case, including:
 - intent or future tense: "I'll add…", "Let me update…", "Next I will…", "Here's what I'm going to change";
 - offers and permission requests: "Should I apply this?", "Want me to make these changes?";
 - vague completion claims with no named change or location: "Done!", "All set", "Fixed it";
 - instructions for the developer: "add this to your file", "run npm install", "you can change X to Y";
 - snippets that are partial, unnamed, or explicitly examples ("something like…", "for example");
 - a response cut off before any of (A)–(D) is visible (do not assume the change happened); but a
   truncated response that already names a specific change it made counts under (D).
Everything that fails the test is ILLUSTRATIVE, no matter how much code it contains.

REPORTED PROBLEM — the prompt, or the previous turn it continues, asserts that something that already
exists (behavior, output, build, UI, data) currently differs from what the developer expects: a pasted
error or failing output, a described wrong behavior, or a report that an earlier fix did not work.
No keyword is needed. Test: can the prompt (read with the previous turn) be restated as "X currently does
Y, but should do Z", where Y is an observed current state? If yes, a problem is reported.
NOT a reported problem: hypotheticals ("if there are errors"), requests for new or different behavior
("add a button", "make this error fail gracefully"), errors the agent runs into on its own while working.

VERDICT — a judgment about existing work against some standard: correct/incorrect, conforms/doesn't,
issues found, pass/fail, done/remaining, "you should change X".

## Categories

implement — Necessary and sufficient: (1) the response contains an APPLIED CHANGE, (2) that change is the
main deliverable, and (3) no REPORTED PROBLEM drives the turn. Long code blocks, a detailed plan, or
commands run only to inspect/verify do NOT make a turn implement. Errors the agent hits and fixes while
doing requested work stay implement.

debug — Necessary and sufficient: (1) a REPORTED PROBLEM exists, and (2) the response engages it:
diagnoses or explains the cause, proposes or applies a fix, adds instrumentation, or asks for specific
diagnostic information after some analysis. Applies whether or not the cause is found and whether or not
the fix is applied. A problem word in the prompt with no current misbehavior asserted is NOT debug.

explain — Necessary and sufficient: (1) no APPLIED CHANGE is the main deliverable, (2) no REPORTED PROBLEM
drives the turn, (3) the response delivers understanding — concepts, how existing code works or what it
contains (including a summary that gives no VERDICT), options and recommendations, instructions for the
developer, or a plan/design for future work — and (4) it gives no VERDICT on existing work.

review — Necessary and sufficient: (1) no REPORTED PROBLEM drives the turn, (2) an existing object is
assessed (code, config, project structure, command/test output, a document, progress against a plan),
(3) the response gives a VERDICT on it, and (4) no APPLIED CHANGE is the main deliverable. Words like
"check", "analyze", "audit" or "summarize" in the prompt do not make a turn review; the VERDICT does.

tool_only — the response is mostly tool activity (searching, listing, reading files, read-only commands)
and stops before any result of its own: no answer, change, diagnosis, findings or verdict.

other — no meaningful coding action: greetings/thanks, empty or cut-off responses, only a clarifying
question with no analysis, refusals, rate-limit/system messages, off-topic chat. Not an "unsure" label.

## Boundary rules (apply in order)

1. A REPORTED PROBLEM that the response works on → debug, even if the fix is new code, an explanation, or
   a review of the code involved.
2. No reported problem and an APPLIED CHANGE is the main deliverable → implement.
3. Audit or summary of existing code: VERDICT given → review; only description/summary → explain.
   Progress/status against a plan or spec counts as a VERDICT (done/remaining) → review.
4. Implementation request answered with only a plan (no APPLIED CHANGE) → explain, even if it ends with
   "should I proceed?". It is review only if the plan's main content is an assessment of existing work.
   If changes are then applied in the same turn → implement.
5. Tool activity that reaches any conclusion takes that conclusion's label; tool_only only when it stops
   before a conclusion.
6. implement vs explain: if the APPLIED CHANGE test is YES and the change is the main deliverable → implement,
   never explain. If the test is NO → never implement, however complete or long the code looks.
7. Mixed turns: label the main deliverable.

## Special situations
- "continue"/"try again": use the previous turn; label the work done in THIS response. "Try again" after a
  failed fix is debug.
- Truncated responses: label what is visible.
- Translated prompts: label the meaning, not the exact English wording.
- Prompt asks for X, agent does Y: label Y.

## Output
confidence: about 0.9+ when the rules settle it clearly; 0.5–0.7 when another label is arguable.
Respond with ONLY the JSON object, no other text.
