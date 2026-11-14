# contracts

The four message shapes between the parts, from PLAN.md section "Contracts", plus the additions the lead accepted on the board (post #6884057018).
A change to any file here needs agreement on the board from every owner whose part uses that message (AGENTS.md section 4).

| # | From, to | Example files | Schema |
|---|---|---|---|
| 1 | Page to server | `examples/1-*.json` | `schemas/1-page-to-server.schema.json` |
| 2 | Command reader to steerer | `examples/2-target.json` | `schemas/2-command-to-steerer.schema.json` |
| 3 | Steerer to loop | (a Python call, see below) | none |
| 4 | Server to page | `examples/4-*.json` | `schemas/4-server-to-page.schema.json` |

Schemas are JSON Schema draft 2020-12.
They allow additional properties, so a field added later by agreement does not break an existing validator.

Check every example against its schema:

```
pip install jsonschema
python contracts/validate.py
```

Other code can check one message with `errors(contract_number, message)` from `contracts/validate.py`, which returns a list of problems (empty means valid).

## Units

A "chunk" in every contract is one `steerer.get_cond()` call, which is one `generate_chunk` call of the model.
The number of video frames in a chunk is whatever the model produces; do not assume a fixed count.
The real model gives 4 frames per chunk (0.25 s at 16 fps), and its first chunk gives 1 frame.

## 1. Page to server

```json
{"type": "partial", "text": "make it win", "t_client": 1731612345123}
{"type": "final",   "text": "make it winter", "t_client": 1731612345890}
{"type": "tap",     "action": "remove", "id": "c2"}
{"type": "tap",     "action": "undo"}
{"type": "tap",     "action": "reset"}
```

`t_client` is the page's clock in ms since the Unix epoch.
It is required on `partial` and `final`, and optional on `tap`.
A `remove` tap needs the `id` of the change; `undo` and `reset` take no `id`.

## 2. Command reader to steerer

```json
{"prompt": "a wide shot of a meadow..., in deep winter...", "ramp_chunks": 4, "provisional": false, "cap": 1.0}
```

`provisional` is true for an early-steering guess, and `cap` is how far the move may go before it is confirmed.

## 3. Steerer to loop

```python
steerer = Steerer(initial_embedding)
steerer.set_target(embedding, ramp_chunks, cap=1.0)  # contract 2 arrived and its embedding is ready
cond = steerer.get_cond()                            # the loop calls this once per chunk
steerer.progress                                     # 0 to 1, the progress field of contract 4
```

`cap` is optional with default 1.0 and matches the `cap` field of contract 2.
The steerer keeps tensors by reference, so never modify the tensor returned by `get_cond()` in place.
`get_cond()` returns the same object while idle and a new object when the conditioning changed, so `cond is not last_cond` tells whether it changed.

## 4. Server to page

```json
{"chunk": 412, "changes": [{"id": "c1", "label": "winter", "strength": 1.0}], "progress": 0.5,
 "stamps": {"heard": 0, "decided": 4, "embedded": 61, "generated": 790, "shown": 1010},
 "t_client": 1731612345890, "applies_from_chunk": 410,
 "heard": "make it winter", "keyword": "winter", "unmatched": false}
```

One message per chunk, sent when that chunk's first frame is sent.

- `changes`: the full list of active changes, each with `id`, `label` and `strength`.
- `progress`: the steerer's progress of the current move, 0 to 1.
- `stamps`: integer ms after `heard` for the most recent command (see "Wire details"); each field appears once it is known, and it is `{}` before the first command.
- `t_client`: echo of the most recent command's `t_client`, or null.
- `applies_from_chunk`: the first chunk generated with the most recent command's target, or null until that is known.
  The page measures its own "shown" time as the first displayed frame whose chunk is at least `applies_from_chunk`.
- `heard`: the text of the most recent command, or null.
- `keyword`: the key word to highlight in `heard`, or null.
- `unmatched`: true when no operation was found for the most recent command.

The command reader owner (`commands/`) may still adjust these names, which is why the schemas accept extra fields.

## Wire details

These state how the shapes above travel; they are the engine's actual endpoints.

- One HTTP origin serves everything (default port 8000).
- Websocket `/control` carries contract 1 messages in and contract 4 messages out, each as one JSON text message.
  A message that does not validate gets a reply `{"error": "..."}` and the socket stays open.
- Websocket `/frames` carries binary messages, one per video frame:

  | Bytes | Content |
  |---|---|
  | 0 to 3 | chunk number, unsigned 32-bit, big-endian |
  | 4 to 5 | frame index inside that chunk, unsigned 16-bit, big-endian |
  | 6 to end | the JPEG bytes of the frame |

- The server paces frames at the video frame rate, so the page shows each frame when it arrives.
- Stamps are integer milliseconds after `heard`, where `heard` is the server's receive time of the contract 1 message that led to the command.
