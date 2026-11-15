# Mirage, live voice-steered video: MadHacks build plan
14 Nov 2026 · Ishaan (`@Tactacion`)

## What we are building
A video that never stops generating and changes within a fraction of a second when you speak.
You say "make it winter" and it snows. You say "also add a river" and a river appears while the snow stays. You say "remove the river" and only the river goes. Nothing is trained. All of it happens while the model is running.

It builds on Training-Free Interactive World Models, the $50k winner of the 2026 Inference-Time Compute Hackathon. That project showed one move: a single voice command that replaces the scene's prompt over a fixed 3 seconds. We keep that core and add four features on top.

| | Original | Ours |
|---|---|---|
| Commands | One swap: the new prompt replaces the old | Changes stack on each other |
| Undo | Not shown | Undo the last change, or remove one named change |
| Pace | Fixed 3-second ramp | Set by how you say it: "slowly", "suddenly" |
| Strength | All or nothing | "A little", "a lot more" |
| When it starts | After you finish speaking | While you are still speaking |
| Input | Speech recognition on the server | Wispr Flow into a text box, or a built-in live mic |

What a judge sees:
- A full-screen video that is being generated as they watch.
- Their words appear under it, and the scene changes as they speak.
- Each active change sits on screen as a small chip: winter, river, night.
- They say "remove the river" or tap its chip, and only that change goes.
- A number in the corner shows the delay from their word to the pixels.

## Features
Six features, in build order. The first is the original's; the other five are ours.

| # | Feature | You say | What happens | Done when |
|---|---|---|---|---|
| 1 | Live steering | "Make it winter" | The scene turns to winter with no cut | A typed command changes the scene smoothly, 10 times in a row |
| 2 | Stacking | "Also add a river" | The river appears and winter stays | Three changes are active at once and all are visible |
| 3 | Undo last | "Go back" | The most recent change fades out | Undo works three times in a row, back to the starting scene |
| 4 | Remove one | "Remove the river" | Only the river goes; winter and night stay | Any one of three active changes can be removed by name or by tapping its chip |
| 5 | Pace and strength | "Slowly turn to night", "a lot more snow" | The change takes longer, or goes further | "Slowly" and "suddenly" look clearly different; so do "a little" and "a lot" |
| 6 | Early steering | (mid-sentence) "make it win..." | The scene starts changing before you stop talking | The first visible change lands before the sentence ends, on the built-in mic |

If time runs short, cut from the bottom. Features 1 to 4 are the demo. Feature 6 is the one most likely to slip.

Undo covers two different things, and both ship:
- Undo last steps back through history, one change at a time.
- Remove one takes out a single named change from the middle of the stack and keeps everything else.

## How it works
Every feature is the same move underneath: change the target, then blend toward it. That is why six features fit in 24 hours.

From spoken words to changed pixels:
1. Words arrive from the text box or the live mic.
2. The command reader turns the words into one operation: add, remove, undo, set strength, set pace.
3. The scene state applies it. The state is a base scene plus a list of active changes, each with a strength, plus a history for undo.
4. The composer writes the state out as one prompt, and the text encoder turns that prompt into the target embedding.
5. The steering module moves the live conditioning from where it is now toward the target, a step per chunk. A new command mid-move only changes the target.
6. The video model generates the next chunk with that conditioning. Its cache of past frames is left alone, which keeps the scene continuous.
7. A second GPU decodes the chunk and streams the frames to the page.
8. The page shows the video, the words, the chips and the delay.

### Two ways to speak to it

| | Wispr Flow into the text box | Built-in live mic |
|---|---|---|
| How text arrives | The whole sentence at once, after you stop | Word by word while you speak |
| Text quality | Clean and punctuated | Rough, partial words get revised |
| Delay it adds | Wispr targets under 700 ms after you stop | Roughly 150 to 300 ms per word |
| Supports early steering | No | Yes |
| Best for | Precise or unusual commands | The fast demo commands |

Wispr Flow's desktop app inserts the finished text after you stop speaking, not word by word. So early steering needs the live mic. Both inputs feed the same command reader, and a typed command works too.

### Numbers to start from
From the original write-up, on H100 GPUs:

| Item | Value |
|---|---|
| Chunk length | 0.75 s of video |
| Resolution and frame rate | 480 x 832 at 16 fps |
| Denoising steps per chunk | 4 |
| Generation per chunk | about 700 ms on GPU 0 |
| Decode per chunk | about 276 ms on GPU 1 |
| One GPU doing both | 976 ms per chunk, slower than playback |
| Their voice-to-pixels delay | 0.69 s from the end of speech |

The first latency job is to reproduce their 0.69 s using their method of measuring. After that, measure ours from the moment the key word is spoken, and show that number live on the page.

## Who owns what
Four people, four parts, and each part can be built and tested alone until hour 7.

| Person | Owns | First deliverable | Also backs up |
|---|---|---|---|
| Ishaan | Steering: how the prompt embedding is blended, paced, scaled and interrupted. Everything inside the diffusion model. | A typed command changes the scene smoothly, by hour 7 | Teammate 1 on model internals |
| Teammate 1 | Real-time engine: GPUs, the generation loop, the text encoder, streaming, delay measurement | Stock model generating in real time, by hour 3 | Ishaan on speed |
| Teammate 2 | The page and the inputs: video, words, chips, undo, the Wispr Flow text box, the live mic | Page working against a fake server, by hour 4 | Teammate 3 on the demo |
| Teammate 3 | Understanding commands: scene state, operations, undo history, the prompt composer. Also the write-up and pitch. | Command reader passing its phrase tests, by hour 6 | Teammate 2 on voice |

Pick who is teammate 1, 2 and 3 by skill:
- Teammate 1 should be whoever is strongest with PyTorch, GPUs and servers.
- Teammate 2 should be whoever is fastest at web front ends.
- Teammate 3 should be comfortable with LLM APIs and also be the best writer and presenter.

## Ishaan: steering
You own one object, the steerer, and every feature goes through it.
It holds two tensors. `cond_now` is what the model is conditioned on for this chunk. `cond_target` is where it is heading. Each chunk, `cond_now` moves one step toward `cond_target`. Everyone else only calls `set_target` and `get_cond`.

```python
class Steerer:
    def set_target(self, target, ramp_chunks):  # a command landed
        self.start = self.cond_now              # start from wherever we are, even mid-move
        self.target = target
        self.ramp = ramp_chunks                 # how many chunks the move takes
        self.k = 0                              # chunks done so far in this move

    def get_cond(self):                         # the loop calls this once per chunk
        if self.k < self.ramp:
            self.k += 1
            w = ease(self.k / self.ramp)        # 0 to 1, gentle at both ends
            self.cond_now = blend(self.start, self.target, w)
        return self.cond_now
```

`blend` is the original's per-token spherical interpolation, with a plain linear blend when two vectors are nearly the same. `ease` is their smooth start-and-stop curve. Starting each move from `cond_now` is the new part: it is what lets commands interrupt each other.

| Hours | Task | Done when |
|---|---|---|
| 0 to 3 | With teammate 1, find the text-embedding tensor each chunk uses, and confirm it can change between chunks without touching the frame cache | You can swap the prompt by hand in a notebook and the video reacts |
| 3 to 5 | Reproduce the original: one command, old prompt to new, fixed 4-chunk ramp | "Meadow" to "winter meadow" with no cut |
| 5 to 7 | Make moves interruptible with the class above | Two commands one second apart, no cut |
| 7 to 8 | Pace: try ramps of 1, 2, 3 and 4 chunks on five changes, by eye | You have a number for "suddenly" (shortest that never cuts), 4 for default, 12 for "slowly" |
| 8 to 10 | Strength: target = blend(without the change, with it, s) | s = 0.3 and s = 1.0 look clearly different |
| 10 to 12 | Stacking and remove-one, with teammate 3 | Removing the middle of three changes leaves the other two untouched |
| 12 to 16 | Early steering: a partial move on a spotted keyword | A wrong guess is barely visible; a right guess saves time |
| 16 to 20 | Stretch: change the conditioning between denoising steps inside a chunk | Either it saves delay cleanly, or you drop it |

### Three things to watch
- **A second cache.** If the code caches the text's attention keys and values because the prompt is normally fixed, make it recompute them every chunk. That is separate from the frame cache, which you never touch.
- **Removing from the middle.** Deleting one phrase shifts every later token, so a per-token blend then mixes unrelated words. If those transitions look wrong, give each kind of change its own padded span of tokens (season, weather, time of day, two free slots). Then removing one leaves the rest in place.
- **Strength above 1.** For "a lot more", let teammate 3 change the wording ("heavy snow"). Only push s to 1.2 or 1.3 if the picture holds.

### Early steering in detail
Teammate 3's keyword spotter fires on partial text, for example on "winter" before the sentence ends. You then start a move toward that keyword's target, capped at 40% of the way. Teammate 1 precomputes those targets, so no text encoding is needed.
- Final text confirms the guess: continue to 100%.
- Final text says something else: move back and apply the real command.

Tune the cap until a wrong guess reads as a faint shimmer.

### Do not spend time on these
The original team tried them and reports that they failed: swapping the prompt instantly, recomputing the frame cache under the new prompt, editing the cache or reweighting attention, and shrinking the context window.

## Teammate 1: real-time engine
You own the machine that never stops generating, and the number in the corner of the screen.
The loop, with Ishaan's steerer plugged in:

```python
while running:
    cond = steerer.get_cond()                         # Ishaan's module, once per chunk
    latents = model.generate_chunk(cond, frame_cache) # GPU 0; the frame cache is never rebuilt
    decode_queue.put((chunk_id, latents))             # GPU 1 decodes and streams in parallel
```

| Hours | Task | Done when |
|---|---|---|
| 0 to 1 | Start the weights download the minute the clock starts | Download running on the rented machine |
| 1 to 3 | Run stock generation. Log milliseconds per chunk for generation and for decode. Move decode to the second GPU with a queue between them. | Generation takes under 750 ms per chunk, so video plays in real time |
| 3 to 5 | Stream frames to a browser. Start with JPEG frames over a websocket; it is simple and enough for 16 fps. | Live video plays on a laptop on the venue network |
| 5 to 7 | Control socket in, steerer hooked into the loop, text encoder as a service | A typed command reaches the model on the next chunk |
| 7 to 10 | Delay measurement, end to end | The page shows a live number in milliseconds |
| 10 to 12 | Precompute embeddings for the keyword list. Cache every prompt you encode. | A known keyword needs zero encoding time |
| 12 to 16 | Recorder (every session to mp4), reset button, auto-restart | A crash recovers in under 10 seconds |
| 16 to 20 | Soak test: 30 minutes of continuous commands | No stall, no drift in delay |

### The text encoder
A new command means a new prompt, and encoding it takes time. Run the encoder on GPU 1, off the generation path, and return the embedding to the steerer when ready. Cache by prompt text. Precompute the keyword targets so early steering never waits for it.

### Measuring the delay
Stamp each command at five points: word heard, operation decided, embedding ready, first chunk generated with it, first frame shown. Send the stamps to the page with the frames. Report two numbers: from the end of speech (to compare with the original's 0.69 s) and from the key word.

### Fallbacks
- Not real time by hour 3: lower the resolution or the frames per chunk until generation fits in 750 ms.
- Base model will not run by hour 3: switch to AlayaWorld, an open model that already supports changing the prompt mid-generation and ships a browser demo. Ishaan finds the conditioning tensor again there.
- Only one GPU available: accept a smaller picture. The original measured 976 ms per chunk on one GPU at full size.

Keep the machine rented until judging ends. Do not shut it down to save money on the last morning.

## Teammate 2: the page and the inputs
You own everything the judge looks at and everything they speak into.

### What is on the page
- The video, as large as the screen allows.
- The words line under it: what was heard, with the key word highlighted when it triggers a change.
- The chips row: one chip per active change (winter, river, night), each with an x to remove it.
- An undo button, for when voice mishears.
- The delay number in a corner, in milliseconds.
- A text box, always focused, so Wispr Flow has somewhere to paste.
- A mic toggle for the live mic.

| Hours | Task | Done when |
|---|---|---|
| 0 to 4 | Build the page against a fake server that loops a video clip and echoes commands | Every element above is on screen and reacts |
| 4 to 6 | Text box input: send the text the moment it changes | Typing or a Wispr Flow paste sends a command with no button press |
| 6 to 8 | Connect to the real server | A typed command changes the real video |
| 8 to 11 | Live mic with partial results | Words appear on the words line while you are still speaking |
| 11 to 13 | Chips and undo wired to real state | Tapping a chip's x removes that change on screen and in the video |
| 13 to 16 | Delay number, key-word highlight, connection-lost state | The number updates on every command |
| 16 to 20 | Big-screen layout and polish | Readable from three metres away |

### Wispr Flow input
Wispr Flow pastes the finished sentence into whichever text field is focused, after you stop speaking. So keep the text box focused at all times, and send on the input event, not on Enter. Clear the box after sending.
Install it on the demo laptop early and test the hotkey with the page in full screen. Its free desktop tier has a weekly word cap, so check the plan before demo day.

### Live mic input
Use the browser's built-in speech recognition in Chrome with interim results turned on. Send each partial result as it arrives, and mark the last one as final. If it is unreliable on the venue network, switch to a streaming speech-to-text service that returns partial results.
The hall will be loud. Use a headset or a close microphone, and add a push-to-talk key so the page only listens while it is held.

### One rule
The server is the source of truth. The chips show the state the server sends back, never what the page assumes it sent.

## Teammate 3: understanding commands
You own the step between words and a prompt. Stacking, both kinds of undo, pace and strength all live in your code, and none of it needs a GPU to test.

### The scene state
```json
{
  "base": "a wide shot of a meadow with a single oak tree",
  "changes": [
    {"id": "c1", "label": "winter", "kind": "season",  "phrase": "in deep winter, snow on the ground", "strength": 1.0},
    {"id": "c2", "label": "river",  "kind": "extra",   "phrase": "a river running through it",       "strength": 1.0},
    {"id": "c3", "label": "night",  "kind": "time",    "phrase": "at night under a full moon",        "strength": 0.6}
  ],
  "history": []
}
```

Every command becomes one operation on this state:

| Operation | Example words | Effect |
|---|---|---|
| add | "Make it winter", "also add a river" | Appends a change. A new season replaces the old season. |
| remove | "Remove the river", "no more snow" | Deletes the change whose label matches |
| undo | "Go back", "undo" | Restores the previous state from history |
| set strength | "A lot more snow", "a little foggy" | Changes one change's strength |
| new scene | "Show me a beach" | Replaces the base and clears the changes |

Pace is not an operation. It is a field on any command: "slowly" and "suddenly" set how many chunks the move takes.

### Two readers, fast and slow
- **Fast reader:** a keyword table, no model. About 40 entries: winter, snow, night, rain, fog, storm, sunset, fire, autumn, undo, go back, remove, slowly, suddenly, a little, a lot. It runs on partial text for early steering and handles most demo commands in under a millisecond.
- **Slow reader:** one LLM call. It gets the current state and the sentence, and returns one operation as JSON. It handles what the table misses, such as "turn the tree into a lighthouse." Use a fast model and aim for under 400 ms. If it times out, do nothing and show the words in red.

### The composer
It writes the state out as one prompt in a fixed order: base first, then changes sorted by kind. A fixed order keeps the same words in the same positions between prompts, which is what makes Ishaan's blends clean. If he asks for fixed-width slots, pad each kind to an agreed length.

| Hours | Task | Done when |
|---|---|---|
| 0 to 2 | State, operations and history as plain code | add, remove and undo pass by hand |
| 2 to 4 | Keyword table and the fast reader | 25 of the test phrases map correctly |
| 4 to 6 | Composer, plus a list of 40 test phrases with expected operations | All 40 run as automatic tests |
| 6 to 8 | Slow reader | Five unusual commands return the right operation |
| 8 to 12 | Wire into the server with teammate 1; partial-text spotting with Ishaan | Voice stacking, undo and remove-one work on the real video |
| 12 to 16 | Pick three base scenes with Ishaan; write the demo run | A scripted run works five times in a row |
| 16 to 22 | Devpost text, three slides, credits, the backup video | Submitted two hours before the deadline |

### Matching a name to a change
"Remove the river" has to find the right change. Match the spoken word against each label and phrase, and accept close matches such as "stream." If two changes could match, remove the most recent. If nothing matches, do nothing and flash the chips so the user can tap one.

## Contracts
Fix these four message shapes in the first hour, and nobody waits on anybody until hour 7.

| # | From, to | What it carries | Stand-in until the real thing exists |
|---|---|---|---|
| 1 | Page to server | Heard text, partial or final | Teammate 2 types into the text box |
| 2 | Command reader to steerer | A target prompt and a ramp length | Ishaan hard-codes two prompts |
| 3 | Steerer to loop | One conditioning tensor per chunk | Teammate 1 passes a fixed prompt |
| 4 | Server to page | Frames, current state, delay stamps | Teammate 2's fake server loops a clip |

**1. Page to server**
```json
{"type": "partial", "text": "make it win", "t_client": 1731612345123}
{"type": "final",   "text": "make it winter", "t_client": 1731612345890}
{"type": "tap",     "action": "remove", "id": "c2"}
```

**2. Command reader to steerer**
```json
{"prompt": "a wide shot of a meadow..., in deep winter...", "ramp_chunks": 4, "provisional": false, "cap": 1.0}
```
`provisional` is true for an early-steering guess, and `cap` is how far the move may go before it is confirmed.

**3. Steerer to loop**
`steerer.get_cond()` returns the tensor for the next chunk. `steerer.set_target(embedding, ramp_chunks)` is called when contract 2 arrives and the embedding is ready.

**4. Server to page**
```json
{"chunk": 412, "changes": [{"id": "c1", "label": "winter"}], "progress": 0.5,
 "stamps": {"heard": 0, "decided": 4, "embedded": 61, "generated": 790, "shown": 1010}}
```
Frames travel on a separate socket, each tagged with its chunk number. The stamp values here are placeholders, in milliseconds after the word was heard.

## Schedule
Five gates decide the day; hours count from the moment hacking starts.

| Hours | Ishaan | Teammate 1 | Teammate 2 | Teammate 3 |
|---|---|---|---|---|
| 0 to 1 | Agree contracts | Start weights download; agree contracts | Agree contracts | Agree contracts |
| 1 to 3 | Find the conditioning tensor | Stock generation, timing, two GPUs | Fake server and page skeleton | State, operations, history |
| 3 to 5 | Reproduce the original blend | Stream frames to a browser | Page elements; text box input | Keyword table, composer |
| 5 to 7 | Interruptible moves | Control socket, steerer hook, encoder service | Connect to the real server | Test phrases; slow reader |
| 7 to 10 | Pace, then strength | Delay measurement | Live mic with partial words | Wire the reader into the server |
| 10 to 12 | Stacking and remove-one | Precompute keywords, cache prompts | Chips and undo on real state | Partial-text keyword spotting |
| 12 to 16 | Early steering | Sleep 12 to 15, then recorder and reset | Delay number, highlights | Sleep 12 to 15, then scenes and demo run |
| 16 to 20 | Sleep 16 to 19, then stretch or fixes | Soak test | Sleep 16 to 19, then big-screen polish | Devpost, slides, credits |
| 20 to 22 | Rehearse | Rehearse | Rehearse | Cut backup video; submit |
| 22 to 24 | Buffer and table setup | Buffer and table setup | Buffer and table setup | Buffer and table setup |

### Gates

| Gate | Hour | Passes when | If it fails |
|---|---|---|---|
| 1 | 3 | The stock model generates in real time | Shrink the picture, or switch base model (teammate 1's fallbacks) |
| 2 | 7 | A typed command changes the real scene with no cut | Everyone helps until it passes. There is no project without it. |
| 3 | 12 | By voice: stack three changes, undo one, remove one by name | Cut remove-one; keep stacking and undo |
| 4 | 16 | Pace, strength and early steering work | Cut early steering; keep the rest |
| 5 | 20 | Feature freeze, and the backup video is recorded | Nothing new goes in after this, only crash fixes |

### Sleep
Two pairs, three hours each. Someone who knows the model is always awake: Ishaan covers hours 12 to 15, teammate 1 covers hours 16 to 19. Ishaan's stretch task is the first thing to drop if he needs the sleep.

### Rehearsal
Run the full demo five times between hours 20 and 22, twice with someone outside the team speaking. Outsiders say things the keyword table has never seen.

## Risks and fallbacks
The two that can sink the project are getting GPUs and getting the base model to run in real time; settle both before anything else.

| Risk | How you will know | Fallback |
|---|---|---|
| No GPU quota when the clock starts | The rental console refuses two large GPUs | Arrange accounts on two providers beforehand |
| Base model will not run, or setup is undocumented | No video by hour 2. The public repo's README is thin. | Switch to AlayaWorld at hour 3 |
| Not real time | Generation takes over 750 ms per chunk | Smaller picture or fewer frames per chunk |
| Transitions cut or collapse | Visible jumps on some commands | Longer default ramp; avoid those command types in the demo |
| Remove-one disturbs other changes | Removing the river also changes the snow | Fixed-width slots; or cut remove-one and keep undo |
| Voice mishears in a loud hall | Wrong chips appear | Headset, push-to-talk, the words line, and the undo button |
| Slow reader is slow or wrong | Commands lag or do the wrong thing | Demo with keyword-table commands; show unknown words in red |
| Early steering flickers | The scene twitches on wrong guesses | Lower the cap, or cut the feature |
| Venue Wi-Fi drops the stream | Frozen video | Phone hotspot; lower stream quality; backup video ready |
| Scene drifts after many commands | Picture degrades over minutes | Reset button; keep each demo run under three minutes |
| Judges ask what is yours | It will come up | Credit the original first, then list the five features you added |

### On building from an existing project
MadHacks expects projects to be created from scratch during the event. Using an open model and a published method is normal, but say so plainly: name the original project in the pitch, on the Devpost page and on a slide. Ask an organiser at the start if anything is unclear.

## Demo script
Three minutes, and the judge is speaking by the halfway mark.

| Time | You say or do | What the judge sees |
|---|---|---|
| 0:00 | "Nothing on this screen is a recording. It is being generated right now." | A sunny meadow, moving |
| 0:15 | "Make it winter." | Snow arrives; a winter chip appears; the delay number updates |
| 0:30 | "Also add a river." | A river appears and the snow stays; two chips |
| 0:45 | "Slowly turn to night." | A long fade to night; three chips |
| 1:00 | "A lot more snow." | The snow gets heavier |
| 1:10 | "Remove the river." | Only the river goes |
| 1:20 | "Go back." | The river returns |
| 1:30 | Hand over the headset: "Your turn. Say anything." | Their own words change the scene |
| 2:15 | Point at the words line while they speak | The scene starts changing before they finish the sentence |
| 2:30 | "This builds on a project that did one swap over three seconds. We added stacking, two kinds of undo, pace, strength and early steering." | The credits slide |
| 2:50 | "No training. All of it happens while the model runs." | Back to the live video |

Use Wispr Flow for one scripted command so the judge sees it paste and trigger. Use the live mic for the rest, because early steering only works there.

### Questions to expect
- **What did you build, and what did you reuse?** The video model and the blending idea are reused and credited. The steerer that can be interrupted, the scene state, both undos, pace, strength, early steering and the page are ours.
- **Why no training?** There was no data or time for it, and the point is that none is needed.
- **How fast is it?** Read the number off the screen, and say how it is measured.
- **What breaks it?** Be direct: large changes of location, and very fast repeated commands. Name whatever your rehearsals found.
- **What is it for?** Live visuals for performance, games where the world reacts to speech, and storyboarding by voice.

### If something fails on stage
Press reset, say "that is live generation for you," and carry on. If the stream dies, play the backup video and keep talking through it.

## Before the clock starts
Setup and reading are fair game before the event; project code is not.
- [ ] Decide who is teammate 1, 2 and 3.
- [ ] Open a GPU rental account and confirm you can start two 80 GB GPUs on one machine. Quota approval can take a day, so do this first.
- [ ] Open a second provider account as a spare.
- [ ] Agree who pays. Two H100s at $2.60 an hour each for 26 hours is about $135. Add roughly $20 for LLM calls.
- [ ] Get API keys for one fast LLM.
- [ ] Everyone reads the original write-up end to end.
- [ ] Teammate 1 reads the original repository and notes what is missing from its setup steps.
- [ ] Install Wispr Flow on the demo laptop and check its plan's word limit.
- [ ] Bring a headset or close microphone, and a spare.
- [ ] Ask the organisers two things: is building on a credited open project fine, and may model weights be downloaded before the start.

## Open questions
- Which base model exactly, and does its licence allow this use? The write-up names Causal Forcing++; the repository page did not state a licence.
- When does hacking start, so the hour numbers can become clock times?
- Does anyone on the team already have GPU credits?

## Sources
- Training-Free Interactive World Models: the original write-up, method, timings and what failed (https://hrdlabs.org/research/interactive-world-models/)
- SPED repository: the original team's code (https://github.com/bryandong24/SPED)
- Inference-Time Compute Hackathon 2026: the event and its prizes
- AlayaWorld: the fallback base model
- What is Wispr Flow?: text is inserted after you stop speaking
- OpenRelay products: GPU hourly prices
- Inside MadHacks: event format and the from-scratch expectation

## Status

Added on 2026-11-15 at about 00:40 UTC by the docs sweep (issue #124); the plan above is unchanged.

Product name: Mirage, decided by the lead on 2026-11-15 (subtitle "live voice-steered video").

| Gate | Result | Source |
|---|---|---|
| 1. Stock model in real time | Passed on one L40S at 384x672, 2 steps, TAEHV decoder: 225.6 ms per chunk against the 250 ms budget, stable over 70 chunks. 480x832 takes 443 ms, so full size is not real time on this GPU. The live engine's `/stats` showed 191.5 ms generation and 21.9 ms decode per chunk at 16.0 fps on 2026-11-15. | board post #6884481031 (21:29 UTC), PR #31 |
| 2. A typed command changes the real scene | Passed live after PR #57: full winter about 2 s after "make it winter". The change is not cut-free: it needs a re-anchor, which is a cut to the new picture, dissolved over 8 frames (0.5 s) since PR #113. | board post #6885133702 (22:23 UTC), `engine/samples/gate2_live_make_it_winter_pass.jpg` |
| 3. By voice: stack three, undo one, remove one by name | FAIL on the strict test, measured live on 2026-11-15 at about 02:45 UTC (typed finals through the harness, not voice). From a reset, 10 commands 6 s apart (winter, lion, night, river, rain, red car, sunny, desert, tree, jensen huang), 3 undos and "remove the lion": all 14 understood (none unmatched), and the chips were right after every step, undo and remove included. Visible in the frames: winter, lion, night, river, rain, red car, sunny, desert, jensen huang, undo 1 (jensen goes), undo 3 (back to the snowy riverside with lion and car) and remove the lion (only the lion goes). Not visible: "add a tree" (no second tree, and the lion vanished from the picture), and undo 2 (chips lion, river, red car, desert, but the frame shows only the tree, road and river, no lion or car). Delay: pinned or keyword commands 0.48 to 0.6 s, new sentences 1.7 to 3.2 s (Sonnet); jensen huang was stamped shown at 3.2 s but appeared about 5 s after the send. Stacking, undo and remove by name work as state and mostly on screen; past about 5 objects the picture drops some of them. | `engine/samples/gate3_steps_live.jpg` (frames before, +2 s and +5 s per step), `engine/samples/gate3_harness_sheet_live.jpg` |
| 4. Pace, strength and early steering | Measured live on 2026-11-15 at about 02:50 UTC, each from a reset. Pace PARTLY: "make it winter" is full snow 1 s after the send, while "slowly make it winter" ramps over 12 chunks (contract 4 progress 0.08 to 1.0 from +2.3 s to +5 s) and the picture shifts gradually (darker grass, then a lilac sky and frosted green grass), but it never becomes winter: the reader gives it strength 0.6 and no snow appears even 11 s later. Strength SHOWN: "make it snowy" gives deep snow cover, "make it a little snowy" (winter 0.3) gives frosted grass with the grass still showing. "a lot more snow" after winter (winter 1.3) PARTLY: deeper drifts and snow on the branches, but the scene was already white, so the gain is modest. Early steering NOT SHOWN: the partial "make it win" gives no guess at all (whole-word matching), and the partial "make it winter" gives a provisional guess (keyword winter, progress ramp to the 0.4 cap) but no snow or other visible change before the final; the small changes seen in those 2 s match the normal drift seen with no command. | `engine/samples/gate4a_pace_slowly_vs_normal.jpg`, `engine/samples/gate4b_strength_snow.jpg`, `engine/samples/gate4c_early_steering_partial.jpg` |
| 5. Feature freeze and backup video | Not reached. Every session is recorded as an mp4 on the engine machine (`~/recordings`), which can serve as the backup video. | `engine/README.md`, "Recorder" |
