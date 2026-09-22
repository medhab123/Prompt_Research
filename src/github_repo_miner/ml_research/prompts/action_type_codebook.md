You are labeling turns from real developer-to-AI-coding-agent conversations for a research dataset.
A turn is one developer prompt plus the agent's full response to it. You will also see the previous
turn of the same session (when there is one) and some response metadata.

Assign exactly ONE label describing **what the agent's response delivered**. The developer's prompt
and the previous turn are context for reading the response; they are not what you label.

Categories: implement, debug, explain, review, tool_only, other.

## 1. Three questions for every turn

1. **What did the developer bring?** A request for new work, a reported problem, a question, a request to check something, or nothing substantive?
2. **What did the response deliver?** Changed code/files, a diagnosis or fix, understanding, findings about existing work, only tool activity, or nothing?
3. **What is the main deliverable?** Judge by what the developer walks away with, not by word count. A 30-line explanation around a 3-line applied fix is a fix; a long code block used only to illustrate a concept is an explanation.

The label comes mostly from question 2. Question 1 matters chiefly for one thing: **was a problem reported?** That is what separates debug from implement/review/explain.

## 2. Category definitions

### implement
**Core:** the agent **produces or changes an artifact of the project** (source code, config, tests, docs, scripts, dependencies, project state) as new work or a requested change, with **no reported problem** driving it.
**Test:** After this turn, does the project have something new or different, or does the developer hold complete code ready to drop in?
**Must have (all):**
- The response contains applied edits (file-edit/write tool calls, diffs, "I've updated `x.py`", "Created `y.ts`"), OR complete, ready-to-use code for what was asked, OR a state-changing command run on request.
- Producing that artifact is the main point of the response.
- The turn is not driven by a reported problem (otherwise debug).
**Includes:** new features, endpoints, components, pages, CLI commands; new files, scaffolding, project setup; refactoring, renaming, moving, restructuring, formatting, removing dead code on request; writing tests for working code or new features; writing/updating docs, READMEs, comments, docstrings, changelogs, commit messages or PR descriptions as files or ready-to-use text; config and infrastructure changes (env files, CI YAML, Dockerfiles, package.json, tsconfig); requested state-changing operations (installing dependencies, running migrations, git commit/push/merge, deploying, generating files with a script); performance or UX improvements asked for without a complaint ("make this faster", "add a loading spinner"); applying changes suggested earlier ("ok, apply those suggestions"); continuing an implementation from the previous turn; the agent's own build/test errors that come up along the way while implementing and are fixed as part of finishing (still implement — the problem was not the reason for the turn).
**Excludes:** anything driven by a reported error or misbehavior (debug); code shown only to illustrate how something works (explain); a plan or description of what should change with no finished code (explain); suggested improvements listed but not applied (review).
**Cues (hints only, never decisive):** "I've added", "Updated", "Created", "Here's the complete implementation", diff blocks, many code blocks, has_generated_code=True.

### debug
**Core:** the turn is **driven by a reported problem**, and the agent **works on that problem**: diagnoses it, explains why it happens, proposes a fix, or applies one.
**Test:** Was something reported as broken or wrong, and is the response about that problem?
**Must have (both):**
1. A reported problem, from the developer's prompt (error message, stack trace, failing test/build output, crash, "doesn't work", "still broken", "wrong output", "button does nothing", "it's slow", "page is blank", a described bug), OR from the previous turn when the current prompt continues it ("still getting the same error", "didn't fix it"), OR from a tool run the agent was asked to investigate ("run the tests and fix what fails").
2. The response addresses that problem: reading code or logs to find the cause, explaining what the error means or why it happens, proposing a fix, applying a fix, adding logging to track it down, or asking for specific diagnostic information after some analysis.
**Includes:** crashes, exceptions, compile/type errors, lint failures blocking a build, failing tests; wrong behavior with no error ("the total is off by one", "styles aren't applied", "the API returns an empty list"); performance problems reported as a complaint; environment/setup failures ("npm install fails with EACCES", "module not found"); "what does this error mean?" with a pasted error (explaining a reported error is debug, not explain); retries after a failed fix; investigating a reported problem even when the cause has not been found yet.
**Excludes:** proactive hardening, validation, or better error handling with nothing reported (implement if applied, review if only suggested); errors the agent hits while doing other requested work and fixes along the way (label the requested work, usually implement); general questions about error-handling concepts with no actual problem (explain); checking whether something might be wrong with no symptom, e.g. "can you check this for bugs?" (review).
**Cues:** pasted error/trace text, "error", "fails", "broken", "not working", "why does…", "root cause", "the issue is", "this happens because", "fixed by".

### explain
**Core:** the main deliverable is **understanding or guidance**. The developer learns something or gets instructions, but the project is not changed and no finished code artifact is delivered.
**Test:** Is the developer mainly getting knowledge, an answer, a recommendation, or a plan, and would they still have to do the work themselves?
**Must have:** a substantive answer in prose (possibly with illustrative snippets); no applied changes and no complete ready-to-use code as the main output; no reported problem driving it (otherwise debug).
**Includes:** concepts ("what is a closure"); how existing code works ("walk me through this function", "what does this regex do"); factual or how-to answers given as steps rather than applied changes; comparisons and recommendations ("Redis vs Postgres?", "which library?"); plans, designs, architecture proposals, plan-mode outputs, lists of files to change; step-by-step instructions for the developer to follow; estimates, feasibility, "is it possible to…"; describing a change in prose without delivering the code.
**Excludes:** explanations of why a reported error happens (debug); assessments of the developer's existing code or work, e.g. "is my code OK?" (review); an explanation wrapping changes actually applied (implement/debug); a response that is only a clarifying question (other).
**Cues:** mostly prose, "X works by", "you can", "here's how", "I'd recommend", "the difference is", "Plan:", few or only short code blocks, no edit tool calls.

### review
**Core:** the agent **assesses existing code, config, output, or project state** and reports findings or a status, **without a reported problem** driving it and **without mainly making changes**.
**Test:** Is the agent judging or checking something that already exists and telling the developer what it found?
**Must have:** an object being assessed (code, a PR or diff, config, test results, a document, project progress); an evaluative output (issues found, "looks correct", ratings, a checklist, pass/fail, a status summary); no reported problem (otherwise debug) and changes are not the main output (otherwise implement).
**Includes:** code review of a file, PR, or diff; critique; best-practice or style audits; security/performance/accessibility audits with no reported incident; "check this for bugs", "is this correct?", "does this look right?", "verify the config" when no symptom was reported; verification turns where the agent runs tests or builds to confirm things work and reports the result ("all 42 tests pass"); progress/status reviews against a plan, TODO list, spec, or requirements ("what's left?"); summarizing what was done in the session or a PR; comparing an implementation to a spec and reporting gaps; suggested improvements listed but not applied.
**Excludes:** investigating a reported problem (debug); applying the review's suggestions as the main output (implement); explaining how code works with no judgment of it (explain); exploration that stops before any findings (tool_only).
**Cues:** "looks good", "I found N issues", "I checked", checkmarks/crosses, "recommendations", "status", "summary of changes", "remaining tasks", ratings or scores.

### tool_only
**Core:** the response is **mostly tool activity** (searching, listing, reading files, running read-only commands) and **gives no result of its own**: no answer, change, diagnosis, or findings.
**Test:** If you removed the tool calls and their raw output, would there be almost nothing left beyond "let me look at…"?
**Must have:** visible tool actions or tool output (usually has_tool_use=True, but read the text); no substantive answer, edit, fix, or findings — the agent's own prose is at most a couple of short narration sentences.
**Includes:** exploration that ends mid-investigation ("Let me read auth.ts… Now let me check the tests…" and then the turn ends); turns that consist only of command output or file contents; requested read-only lookups returned raw with no interpretation; long runs of search/grep/read calls cut off before any conclusion.
**Excludes:** tool use that leads to an edit (implement/debug), to an answer (explain), or to findings or a status (review); tool use investigating a reported problem that reaches any analysis (debug); state-changing commands run on request, such as installs, commits, migrations (implement); no tool activity at all (other).
**Cues:** sequences of "Let me search…", "Reading file…", "Running ls", tool-call markup, grep or listing output, the turn ending abruptly after tool calls.

### other
**Core:** the turn has **no meaningful coding action**; none of the five categories above plausibly applies.
**Test:** Is there nothing here that implements, debugs, explains, reviews, or even explores?
**Includes:** greetings, thanks, praise, acknowledgements answered with a pleasantry; empty responses or responses cut off before any content; a bare "continue"/"go on" where the response is also empty or just "Sure, continuing"; responses that are only a clarifying question with no analysis ("Which file do you mean?"); refusals or capability statements ("I can't access your browser"); system/meta messages (rate limits, "conversation too long", model errors, cancelled requests); off-topic, non-coding chat.
**Excludes:** any turn where a real category plausibly applies, even weakly; clarifying questions that come after real analysis or partial work (label that work); a short response that still delivers something ("Done — renamed it." with an edit is implement).
**Rule:** other is NOT an "I'm not sure" label. If torn between two real categories, pick the more likely one and lower your confidence.

## 3. Boundary rules (apply in order when two labels seem to fit)

1. **debug vs anything:** a reported problem that the response works on is debug, even if the fix is mostly new code, a long explanation, or a review of the code involved.
2. **debug vs review when no cause is found yet:** still debug. The label depends only on whether a problem was reported, not on whether the agent found the cause.
3. **debug vs implement (errors along the way):** errors the agent hits while doing requested new work and fixes along the way do not make it debug. Debug requires the problem to be the reason for the turn.
4. **implement vs explain:** implement if the agent applied a change OR delivered complete, ready-to-use code for what was asked. Explain if the output is prose, illustrative snippets, a plan, or a description the developer still has to turn into code. Test: could they paste or keep it as-is?
5. **implement vs review:** findings or suggestions only is review. Findings and the changes applied, with the changes as the main output, is implement. If the findings are the main output and only small fixes were applied, it is review.
6. **review vs explain:** judging the developer's existing work (correct? good? done?) is review; describing how things work or giving general guidance is explain.
7. **review vs debug (vague prompts):** "check this for bugs" with no symptom is review; "this is buggy, check it" or any described symptom is debug.
8. **tool_only vs the rest:** any conclusion drawn from the tools (answer, edit, findings, diagnosis) gets that conclusion's label; tool_only only applies when the response stops before a conclusion.
9. **tool_only vs other:** any real tool activity is tool_only; no tool activity and no content is other.
10. **Mixed turns:** pick the main deliverable (question 3 in section 1).

## 4. Special situations

- **"continue", "go on", "next", "try again":** use the previous turn. Label the work the agent does in THIS response. "Try again" after a failure is debug; "continue" during a feature build is implement.
- **Truncated responses** ("...[truncated"): label what is visible. Edits visible → implement; only exploration so far → tool_only. Lower your confidence if the cut hides the likely outcome.
- **Agent claims work with no visible evidence** ("Done! I've updated the file."): trust the claim if has_tool_use=True or it is plausible in context (implement or debug); lower your confidence if doubtful.
- **Agent asks permission first** ("Should I go ahead and refactor X?"): after real analysis or a plan → explain (or review if it assessed existing code); with no analysis at all → other.
- **Plan / TODO lists:** a plan for future work → explain; checking progress against an existing plan → review.
- **Translated prompts:** label the meaning, not the exact English wording.
- **Prompt asks for X, agent does Y:** label Y.
- **Several separate tasks in one prompt:** label the task that takes up most of the response.
- **Generated docs, commit messages, emails:** written to files or given as ready-to-use text → implement; advice on what to write → explain.
- **Tests:** writing tests → implement; running tests to verify and reporting results → review; fixing failing tests someone reported → debug.

## 5. Worked examples

| Prompt (summary) | Response (summary) | Label | Why |
|---|---|---|---|
| "Add dark mode" | Edits 3 files; the build fails along the way, the agent fixes an import and finishes | implement | Error came up along the way (rule 3) |
| "Dark mode toggle does nothing" | Reads the component, finds the missing state update, patches it | debug | Reported misbehavior (rule 1) |
| "Dark mode toggle does nothing" | Reads 4 files: "I couldn't find the cause yet — can you share the console output?" | debug | Problem reported, analysis started (rule 2) |
| "Why am I getting TypeError: undefined is not a function?" + trace | Explains the cause, no code | debug | Explaining a reported error |
| "What's the difference between == and ===?" | Prose with examples | explain | Concept, nothing reported |
| "How would you add caching here?" | A 5-step plan with a sketch snippet | explain | Plan plus illustrative snippet (rule 4) |
| "How would you add caching here?" | A complete cache module, ready to paste | implement | Complete, ready-to-use code (rule 4) |
| "Review my auth middleware" | Lists 4 issues with suggestions | review | Assessment, not applied |
| "Review my auth middleware and fix issues" | Lists issues, then applies all fixes with large edits | implement | Applied changes are the main output (rule 5) |
| "Check this function for bugs" | "Found an off-by-one on line 12; here's the corrected line." | review | No symptom reported; the finding is the main output (rules 5, 7) |
| "Run the tests" | Runs them: "All 58 pass" | review | Verification with a reported result |
| "Run the tests and fix failures" | Runs them, 2 fail, fixes both | debug | Asked to investigate tool-reported failures |
| "Find where the API key is used" | grep output only, no comment | tool_only | Raw lookup, no conclusion |
| "Find where the API key is used" | "It's used in config.ts:14 and client.ts:88; the second is hard-coded." | review | Findings about existing code |
| "Where is routing defined?" | "Routing is in app/router.tsx. It uses nested layouts…" | explain | Answer and understanding |
| "Let's implement the settings page" | "Let me look at the existing pages… Reading Profile.tsx…" (then ends) | tool_only | Stopped before any result |
| "Commit and push" | Runs git commands: "Pushed to main" | implement | Requested state-changing operation |
| "thanks!" | "You're welcome! Anything else?" | other | No coding action |
| "fix it" | "Which error are you referring to?" | other | Only a clarifying question, no analysis |
| "continue" | Keeps writing the feature from the previous turn | implement | Label the work done in this response |
| "What's left on the TODO list?" | "Done: auth, DB. Remaining: tests, deploy." | review | Status against a plan |

## 6. Output

- Put the number of the boundary rule that decided a close call at the start of your rationale when one applied (e.g. "R4: ...").
- confidence: about 0.9+ when the definitions settle it clearly; about 0.5–0.7 when another label is arguable or context is missing (truncation, unclear prompt).
- Respond with ONLY the JSON object, no other text.
